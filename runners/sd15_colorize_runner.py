import os
import logging
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from PIL import Image
from tqdm.auto import tqdm

from datasets.dataset import PairedRGBLDataset
from datasets.dataset import LSplitDataset
from modules.lora_utils import enable_lora_on_unet
from modules.lcond_unet import LConditionedUNetWrapper 
from modules.denoisers import build_denoiser
from datasets.utils import tensor_to_rgb_u8, save_rgb_u8, read_l_image_to_tensor, rgb_u8_to_lab, lab_to_rgb_u8, L_tensor_minus1_1_to_L100, list_images_by_stem


class SD15ColorizeRunner:
    def __init__(self, args, config, device):
        self.args = args
        self.cfg = config
        self.device = device
        self.is_main = (getattr(config, "rank", 0) == 0)

        self.model_id = self.cfg.sd.model_id
        self.out_dir = self.args.log_path
        self.ckpt_last = os.path.join(self.out_dir, "ckpt_last.pt")

        self._build()

    def _unwrap(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _build(self):
        # SD components
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")

        # training scheduler (for noise add)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=int(self.cfg.diffusion.num_train_timesteps))

        # freeze VAE/text
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.vae.to(self.device).eval()
        self.text_encoder.to(self.device).eval()


        # LoRA
        unet = enable_lora_on_unet(unet, rank=int(self.cfg.lora.rank))

        # conditioned model
        self.model = LConditionedUNetWrapper(unet)

        # trainable params: LoRA + injector
        for n, p in self.model.named_parameters():
            if ("lora_" in n) or ("injector" in n) or ("lenc.in_proj" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False


        # optimizer
        self.optim = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.optim.weight_decay),
            betas=(float(self.cfg.optim.beta1), float(self.cfg.optim.beta2)),
            eps=float(self.cfg.optim.eps),
        )

        # move / ddp
        if self.cfg.distributed:
            self.model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.cfg.local_rank],
                output_device=self.cfg.local_rank,
                find_unused_parameters=False,
            )
        else:
            self.model.to(self.device)

        # empty prompt embeddings (no-text)
        with torch.no_grad():
            empty = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            empty = {k: v.to(self.device) for k, v in empty.items()}
            self.empty_emb = self.text_encoder(**empty).last_hidden_state  # (1,77,768)

        # train loader (paired RGB/L from split)
        train_split = self.args.train_split
        ds = PairedRGBLDataset(
            rgb_root=self.args.rgb_root,
            l_root=self.args.l_root,
            split=train_split,
            image_size=int(self.cfg.data.image_size),
        )
        if self.cfg.distributed:
            self.train_sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
        else:
            self.train_sampler = None

        self.train_loader = DataLoader(
            ds,
            batch_size=int(self.cfg.training.batch_size),
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            drop_last=True,
        )

        # AMP
        mp = str(self.cfg.training.mixed_precision).lower()
        self.use_amp = (mp in ["fp16", "bf16"])
        self.amp_dtype = torch.float16 if mp == "fp16" else (torch.bfloat16 if mp == "bf16" else torch.float32)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(mp == "fp16"))

        # resume
        self.start_step = 0
        if getattr(self.args, "resume", False) and os.path.isfile(self.ckpt_last):
            self._load(self.ckpt_last)

        if self.is_main:
            trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
            logging.info("Trainable params:\n" + "\n".join(trainable))


    def _save(self, step: int):
        if not self.is_main:
            return
        # state = {
        #     "step": step,
        #     "optim": self.optim.state_dict(),
        #     "injector": self._unwrap().injector.state_dict(),
        #     "lora_attn_processors": self._unwrap().unet.attn_processors.state_dict(),
        # }
        state = {
            "step": step,
            "optim": self.optim.state_dict(),
            "injector": self._unwrap().injector.state_dict(),
            "lora_state_dict": {
                k: v.cpu()
                for k, v in self._unwrap().unet.state_dict().items()
                if "lora_" in k
            },
        }

        torch.save(state, self.ckpt_last)
        torch.save(state, os.path.join(self.out_dir, f"ckpt_{step}.pt"))
        logging.info(f"Saved checkpoint at step {step}")

    def _load(self, path: str):
        state = torch.load(path, map_location="cpu")
        self.optim.load_state_dict(state["optim"])
        self._unwrap().injector.load_state_dict(state["injector"])
        # self._unwrap().unet.attn_processors.load_state_dict(state["lora_attn_processors"])
        if "lora_state_dict" in state:
            self._unwrap().unet.load_state_dict(state["lora_state_dict"], strict=False)
        else:
            logging.warning("Checkpoint has no lora_state_dict; skipping LoRA load.")

        self.start_step = int(state.get("step", 0))
        logging.info(f"Resumed from {path} at step {self.start_step}")

    # -------------------------
    # Training
    # -------------------------
    def train(self):
        max_steps = int(self.cfg.training.max_steps)
        grad_accum = int(self.cfg.training.grad_accum)
        save_every = int(self.cfg.training.save_every)
        log_every = int(self.cfg.training.log_every)

        self.model.train()
        step = self.start_step
        
        pbar = tqdm(total=max_steps,initial=step,desc="train",dynamic_ncols=True,
        disable=not self.is_main,  # only rank0 in DDP
        )

        while step < max_steps:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(step)

            for batch in self.train_loader:
                if step >= max_steps:
                    break

                gt_rgb = batch["gt_rgb"].to(self.device, non_blocking=True)  # (B,3,H,W) [-1,1]
                L = batch["L"].to(self.device, non_blocking=True)            # (B,1,H,W) [-1,1]

                with torch.no_grad():
                    latents = self.vae.encode(gt_rgb).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor  # (B,4,h,w)

                h, w = latents.shape[-2:]
                L_lat = F.interpolate(L, size=(h, w), mode="bilinear", align_corners=False)  # (B,1,h,w)

                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=self.device
                ).long()

                noise = torch.randn_like(latents)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                enc_hid = self.empty_emb.repeat(bsz, 1, 1)

                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    noise_pred = self.model(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=enc_hid,
                        L_latent_1ch=L_lat,
                    )
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                loss = loss / grad_accum
                loss_scalar = loss.detach().item() * grad_accum

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % grad_accum == 0:
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()
                    self.optim.zero_grad(set_to_none=True)

                if self.is_main:
                    pbar.set_postfix(loss=f"{loss_scalar:.6f}")
                    if step % log_every == 0:
                        pbar.write(f"step={step}/{max_steps} loss={loss_scalar:.6f}")


                if (step > 0) and (step % save_every == 0):
                    self._save(step)

                step += 1
                if self.is_main:
                    pbar.update(1)


        if self.is_main:
            pbar.close()
            self._save(step)
            logging.info("Training finished.")

    # -------------------------
    # Single image sampling
    # -------------------------
    @torch.no_grad()
    def sample_image(self):
        if not self.args.input_L or (not os.path.isfile(self.args.input_L)):
            raise FileNotFoundError("--sample_mode single requires --input_L path")

        if os.path.isfile(self.ckpt_last):
            self._load(self.ckpt_last)
        else:
            logging.warning(f"No checkpoint at {self.ckpt_last}. Sampling with current weights.")

        self.model.eval()

        # output path
        if self.args.image_out is not None:
            out_path = self.args.image_out
        else:
            out_path = os.path.join(self.args.exp, "image_single", self.args.doc, "result.png")

        # read L image
        l_u8 = np.array(Image.open(self.args.input_L).convert("L"), dtype=np.uint8)
        L = read_l_image_to_tensor(l_u8).to(self.device)  # (1,H,W) [-1,1]

        img_size = int(self.cfg.data.image_size)
        # enforce size
        L = torch.nn.functional.interpolate(L.unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)

        h, w = img_size // 8, img_size // 8
        L_lat = torch.nn.functional.interpolate(L.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)  # (1,1,h,w)

        denoiser = build_denoiser(self.model_id, method=self.args.sample_method)

        gen = torch.Generator(device=self.device)
        gen.manual_seed(1234 + (self.cfg.rank if self.cfg.distributed else 0))

        latents = torch.randn((1, 4, h, w), device=self.device, generator=gen)
        enc_hid = self.empty_emb.repeat(1, 1, 1)

        final_latents = denoiser.sample(
            model=self.model,
            latents=latents,
            encoder_hidden_states=enc_hid,
            L_latent_1ch=L_lat,
            num_steps=int(self.args.num_steps),
            eta=float(self.args.eta),
            generator=gen,)

        latents_scaled = final_latents / self.vae.config.scaling_factor
        img = self.vae.decode(latents_scaled).sample  # (1,3,H,W) in [-1,1]
        img = (img.clamp(-1, 1) + 1.0) * 0.5          # [0,1]
        rgb_u8 = tensor_to_rgb_u8(img[0])

        # ----------------------------
        # Post-process: replace generated L with conditioning L
        # Keep generated ab as-is
        # ----------------------------
        lab_gen = rgb_u8_to_lab(rgb_u8)                 # (H,W,3)
        ab_gen = lab_gen[..., 1:3].astype(np.float32)       # (H,W,2)

        L100 = L_tensor_minus1_1_to_L100(L)                 # (H,W) in [0,100]

        lab_out = np.zeros_like(lab_gen, dtype=np.float32)
        lab_out[..., 0] = L100
        lab_out[..., 1:3] = ab_gen

        rgb_out_u8 = lab_to_rgb_u8(lab_out)

        if self.is_main:
            save_rgb_u8(rgb_out_u8, out_path)
            logging.info(f"Saved single sampled image to: {out_path}")

        if self.cfg.distributed and dist.is_initialized():
            dist.barrier()

        # if self.is_main:
        #     save_rgb_u8(rgb_u8, out_path)
        #     logging.info(f"Saved single sampled image to: {out_path}")

        # if self.cfg.distributed and dist.is_initialized():
        #     dist.barrier()

    # -------------------------
    # Batch sampling from split (default val)
    # -------------------------
    @torch.no_grad()
    def sample(self):
        # choose split (default config.sampling.split; overridden by --split)
        split = self.args.split

        if os.path.isfile(self.ckpt_last):
            self._load(self.ckpt_last)
        else:
            logging.warning(f"No checkpoint at {self.ckpt_last}. Sampling with current weights.")

        self.model.eval()

        out_dir = os.path.join(self.args.exp, "image_samples", self.args.doc, self.args.image_folder)
        if self.is_main:
            os.makedirs(out_dir, exist_ok=True)
        if self.cfg.distributed and dist.is_initialized():
            dist.barrier()

        # dataset: L from l_root/<split>; optional GT from rgb_root/<split>
        ds = LSplitDataset(
            l_root=self.args.l_root,
            split=split,
            image_size=int(self.cfg.data.image_size),
            rgb_root=(self.args.rgb_root if self.args.save_gt  else None),
        )

        if self.cfg.distributed:
            sampler = DistributedSampler(ds, shuffle=False, drop_last=False)
        else:
            sampler = None

        sample_bsz = int(getattr(self.cfg.sampling, "batch_size", 1))
        
        loader = DataLoader(
            ds,
            batch_size=sample_bsz,
            shuffle=False,
            sampler=sampler,
            num_workers=int(self.cfg.data.num_workers),
            pin_memory=bool(self.cfg.data.pin_memory),
            drop_last=False,
        )

        denoiser = build_denoiser(self.model_id, method=self.args.sample_method)

        img_size = int(self.cfg.data.image_size)
        h, w = img_size // 8, img_size // 8

        save_gt = bool(self.args.save_gt or getattr(self.cfg.sampling, "save_gt", False))

        for batch in loader:
            stems = list(batch["stem"])
            L = batch["L"].to(self.device)  # (B,1,H,W) [-1,1]

            # latent conditioning
            L_lat = torch.nn.functional.interpolate(L, size=(h, w), mode="bilinear", align_corners=False)

            bsz = L_lat.shape[0]
            enc_hid = self.empty_emb.repeat(bsz, 1, 1)

            gen = torch.Generator(device=self.device)
            gen.manual_seed(1234 + (self.cfg.rank if self.cfg.distributed else 0))

            latents = torch.randn((bsz, 4, h, w), device=self.device, generator=gen)

            final_latents = denoiser.sample(
                model=self.model,
                latents=latents,
                encoder_hidden_states=enc_hid,
                L_latent_1ch=L_lat,
                num_steps=int(self.args.num_steps),
                eta=float(self.args.eta),
                generator=gen,
            )

            # decode -> RGB01
            latents_scaled = final_latents / self.vae.config.scaling_factor
            img = self.vae.decode(latents_scaled).sample  # (B,3,H,W) in [-1,1]
            img01 = (img.clamp(-1, 1) + 1.0) * 0.5

            # Post-process each image: replace L only
            for i in range(bsz):
                stem = stems[i]
                gen_rgb_u8 = tensor_to_rgb_u8(img01[i])

                lab_gen = rgb_u8_to_lab(gen_rgb_u8)
                ab_gen = lab_gen[..., 1:3].astype(np.float32)

                L100 = L_tensor_minus1_1_to_L100(L[i])  # conditioning L

                lab_out = np.zeros_like(lab_gen, dtype=np.float32)
                lab_out[..., 0] = L100
                lab_out[..., 1:3] = ab_gen

                rgb_out_u8 = lab_to_rgb_u8(lab_out)

                if self.is_main:
                    pred_path = os.path.join(out_dir, f"{stem}_pred.png")
                    save_rgb_u8(rgb_out_u8, pred_path)

                    if save_gt and ("gt_rgb" in batch) and (batch["gt_rgb"] is not None):
                        gt_rgb = batch["gt_rgb"][i]  # (3,H,W) in [-1,1]
                        gt01 = (gt_rgb.clamp(-1, 1) + 1.0) * 0.5
                        gt_u8 = tensor_to_rgb_u8(gt01)
                        gt_path = os.path.join(out_dir, f"{stem}_gt.png")
                        save_rgb_u8(gt_u8, gt_path)

            if self.is_main:
                logging.info(f"Saved batch of {bsz} samples.")

        if self.cfg.distributed and dist.is_initialized():
            dist.barrier()

