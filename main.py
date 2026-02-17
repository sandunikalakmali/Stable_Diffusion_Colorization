#!/usr/bin/env python
# coding: utf-8
"""
Entry point for SD1.5 colorization with L-conditioning.

Dataset structure (yours):
  RGB root: preprocessed_imagenet_filtered_flat/{train,val}/*
  L root:   preprocessed_imagenet_L_filtered_flat/{train,val}/*
Filenames match (same stem).
"""

import argparse
import logging
import os
import shutil
import sys
import traceback
import yaml
import numpy as np
import torch
import torch.distributed as dist

from runners.sd15_colorize_runner import SD15ColorizeRunner

torch.set_printoptions(sci_mode=False)


def dict2namespace(d):
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def setup_logging(level_name: str, file_path: str = None):
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported logging level: {level_name}")

    logger = logging.getLogger()
    logger.handlers = []
    fmt = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if file_path is not None:
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.setLevel(level)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    # ---- Required ----
    parser.add_argument("--config", type=str, required=True, help="Config yml (inside ./configs or absolute)")
    parser.add_argument("--doc", type=str, required=True, help="Run name (folders)")

    # ---- General ----
    parser.add_argument("--seed", type=int, default=None, help="Override config.training.seed")
    parser.add_argument("--exp", type=str, default="exp", help="Root for logs/ckpts/images")
    parser.add_argument("--verbose", type=str, default="info", help="info | debug | warning | critical")
    parser.add_argument("--ni", action="store_true", help="No interaction (auto-overwrite folders)")

    # ---- Mode ----
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--sample_mode", type=str, default="single", choices=["single", "batch"])
    parser.add_argument("--resume", action="store_true")

    # ---- Dataset overrides ----
    parser.add_argument("--rgb_root", type=str, default=None, help="Override config.dataset.rgb_root")
    parser.add_argument("--l_root", type=str, default=None, help="Override config.dataset.l_root")
    parser.add_argument("--train_split", type=str, default=None, help="Override config.dataset.train_split")
    parser.add_argument("--val_split", type=str, default=None, help="Override config.dataset.val_split")

    # ---- Sampling args ----
    parser.add_argument("--sample_method", type=str, default=None, choices=["ddpm", "ddim"])
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--split", type=str, default=None, help="Which split to sample in batch mode (train/val)")
    parser.add_argument("--save_gt", action="store_true", help="Batch mode: also save GT RGB if available")
    parser.add_argument("--limit", type=int, default=None, help="Batch mode: limit number of images")

    # single
    parser.add_argument("--input_L", type=str, default=None, help="Single mode: path to an L image file")
    parser.add_argument("--image_out", type=str, default=None, help="Single mode output path")

    # batch
    parser.add_argument("--image_folder", type=str, default="images", help="Subfolder under exp/image_samples/<doc>/")

    # ---- Device / distributed ----
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id when not using torchrun")
    parser.add_argument("--distributed", action="store_true", help="Enable DDP (torchrun)")
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # ---- Load YAML ----
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join("configs", args.config)
    with open(cfg_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    config = dict2namespace(cfg_dict)

    # ---- Distributed detection ----
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    using_env_dist = env_world_size > 1 or "LOCAL_RANK" in os.environ or "RANK" in os.environ
    args.distributed = bool(args.distributed or using_env_dist)

    if args.distributed and not using_env_dist and args.local_rank < 0:
        logging.warning("--distributed set but no torchrun env detected; running single-process.")
        args.distributed = False

    if args.distributed and not torch.cuda.is_available():
        logging.warning("DDP requested but CUDA not available; falling back to single-process.")
        args.distributed = False

    if args.distributed:
        local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    is_main = (rank == 0)
    args.local_rank = local_rank
    args.rank = rank
    args.world_size = world_size

    # ---- Output folders ----
    if is_main:
        os.makedirs(args.log_path, exist_ok=True)

        if not args.test and not args.sample and not args.resume:
            if os.path.exists(args.log_path):
                overwrite = args.ni
                if not overwrite:
                    resp = input(f"Folder {args.log_path} exists. Overwrite? (Y/N) ")
                    overwrite = (resp.strip().upper() == "Y")
                if overwrite:
                    shutil.rmtree(args.log_path)
                    os.makedirs(args.log_path, exist_ok=True)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)

        with open(os.path.join(args.log_path, "config.yml"), "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)

    if args.distributed and dist.is_initialized():
        dist.barrier()

    setup_logging(args.verbose, os.path.join(args.log_path, "stdout.txt") if is_main else None)

    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if args.distributed else f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # ---- Seed ----
    seed = args.seed if args.seed is not None else getattr(config.training, "seed", 1234)
    seed = seed + (rank if args.distributed else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # ---- Apply dataset overrides ----
    if args.rgb_root is None:
        args.rgb_root = config.dataset.rgb_root
    if args.l_root is None:
        args.l_root = config.dataset.l_root
    if args.train_split is None:
        args.train_split = config.dataset.train_split
    if args.val_split is None:
        args.val_split = config.dataset.val_split

    # ---- Sampling defaults ----
    if args.sample_method is None:
        args.sample_method = config.sampling.method
    if args.num_steps is None:
        args.num_steps = int(config.sampling.num_steps)
    if args.eta is None:
        args.eta = float(config.sampling.eta)
    if args.split is None:
        args.split = getattr(config.sampling, "split", args.val_split)

    # ---- Create sampling folders ----
    if args.sample:
        if args.sample_mode == "single":
            if is_main:
                out_dir = os.path.join(args.exp, "image_single", args.doc)
                os.makedirs(out_dir, exist_ok=True)
        else:
            if is_main:
                out_dir = os.path.join(args.exp, "image_samples", args.doc, args.image_folder)
                os.makedirs(out_dir, exist_ok=True)
        if args.distributed and dist.is_initialized():
            dist.barrier()

    # ---- Attach runtime info ----
    config.device = device
    config.distributed = args.distributed
    config.rank = rank
    config.world_size = world_size
    config.local_rank = local_rank

    if is_main:
        logging.info(f"Using device: {device}")
        if args.distributed:
            logging.info(f"DDP initialized | world_size={world_size}, rank={rank}, local_rank={local_rank}")
        logging.info(f"RGB root: {args.rgb_root}")
        logging.info(f"L root:   {args.l_root}")

    return args, config


def main():
    args, config = parse_args_and_config()

    try:
        runner = SD15ColorizeRunner(args=args, config=config, device=config.device)
        if args.sample:
            if args.sample_mode == "single":
                runner.sample_image()
            else:
                runner.sample()
        elif args.test:
            runner.test()
        else:
            runner.train()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    except Exception:
        logging.error(traceback.format_exc())
    finally:
        if getattr(config, "distributed", False) and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
