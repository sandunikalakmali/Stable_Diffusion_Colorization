from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Canny edge maps from preprocessed ImageNet L images."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("datasets/preprocessed_imagenet_L_filtered_flat"),
        help="Input root containing split folders (train/val).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/canny_preprocess_L_filterd"),
        help="Output root for Canny edge maps.",
    )
    parser.add_argument(
        "--low-threshold",
        type=int,
        default=100,
        help="Lower threshold for cv2.Canny.",
    )
    parser.add_argument(
        "--high-threshold",
        type=int,
        default=200,
        help="Upper threshold for cv2.Canny.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required for Canny preprocessing. "
            "Install it with: pip install opencv-python-headless"
        ) from exc
    return cv2


def process_split(
    cv2_mod,
    input_root: Path,
    output_root: Path,
    split: str,
    low_threshold: int,
    high_threshold: int,
    overwrite: bool,
) -> dict[str, int]:
    in_dir = input_root / split
    out_dir = output_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "saved": 0, "skipped": 0, "failed": 0}

    if not in_dir.exists():
        print(f"[WARN] Split does not exist, skipping: {in_dir}")
        return stats

    image_paths = list_images(in_dir)
    stats["total"] = len(image_paths)
    print(f"[INFO] {split}: found {stats['total']} images in {in_dir}")

    for src_path in tqdm(image_paths, desc=f"Canny {split}", unit="img"):
        rel_path = src_path.relative_to(in_dir)
        dst_path = out_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not overwrite:
            stats["skipped"] += 1
            continue

        img_l = cv2_mod.imread(str(src_path), cv2_mod.IMREAD_GRAYSCALE)
        if img_l is None:
            stats["failed"] += 1
            continue

        edges = cv2_mod.Canny(img_l, low_threshold, high_threshold)
        ok = cv2_mod.imwrite(str(dst_path), edges)
        if ok:
            stats["saved"] += 1
        else:
            stats["failed"] += 1

    return stats


def main() -> None:
    args = parse_args()
    cv2_mod = require_cv2()

    input_root = args.input_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input root : {input_root}")
    print(f"[INFO] Output root: {output_root}")
    print(
        f"[INFO] Canny thresholds: low={args.low_threshold}, high={args.high_threshold}"
    )

    total_all = {"total": 0, "saved": 0, "skipped": 0, "failed": 0}

    for split in ["train", "val"]:
        stats = process_split(
            cv2_mod=cv2_mod,
            input_root=input_root,
            output_root=output_root,
            split=split,
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
            overwrite=args.overwrite,
        )
        total_all = {k: total_all[k] + stats[k] for k in total_all}
        print(
            f"[SUMMARY] {split}: total={stats['total']} saved={stats['saved']} "
            f"skipped={stats['skipped']} failed={stats['failed']}"
        )

    print(
        f"[DONE] total={total_all['total']} saved={total_all['saved']} "
        f"skipped={total_all['skipped']} failed={total_all['failed']}"
    )


if __name__ == "__main__":
    main()
