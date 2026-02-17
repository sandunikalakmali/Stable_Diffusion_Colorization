from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ----------------------------
# CONFIG (edit these paths)
# ----------------------------
rgb_in_dir = Path("./imagenet_filtered/train")   # filtered RGB (may contain class subfolders)
l_in_dir   = Path("./imagenet_L_filtered/train")     # filtered L (same structure + names)

rgb_out_dir = Path("./preprocessed_imagenet_filtered_flat/train")
l_out_dir   = Path("./preprocessed_imagenet_L_filtered_flat/train")

min_keep = 500          # keep only if min(w,h) >= 500 (before resize)
target_min = 512        # after resize, min(w,h) becomes 512 (aspect ratio preserved)
crop_size = 512         # center crop size (512x512)

rgb_out_dir.mkdir(parents=True, exist_ok=True)
l_out_dir.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".JPEG", ".JPG", ".PNG", ".TIFF"}

# Optional: force a single output extension for both to make it consistent
# If you prefer to keep original extension, set OUT_EXT = None
OUT_EXT = ".png"

rgb_paths = [p for p in rgb_in_dir.rglob("*") if p.is_file() and p.suffix in EXTS]

skipped_missing_pair = 0
skipped_small = 0
saved = 0

for rgb_path in tqdm(rgb_paths, desc="Preprocess paired RGB+L"):
    rel = rgb_path.relative_to(rgb_in_dir)
    l_path = l_in_dir / rel

    if not l_path.exists():
        skipped_missing_pair += 1
        continue

    # Make ONE shared flat name for both outputs (guarantees 1-to-1 matching)
    flat_stem = "_".join(rel.parts).replace(" ", "_")
    if OUT_EXT is None:
        out_rgb = rgb_out_dir / flat_stem
        out_l   = l_out_dir   / flat_stem
    else:
        out_rgb = rgb_out_dir / (Path(flat_stem).with_suffix(OUT_EXT).name)
        out_l   = l_out_dir   / (Path(flat_stem).with_suffix(OUT_EXT).name)

    try:
        # ---------------- RGB ----------------
        with Image.open(rgb_path) as im_rgb:
            im_rgb = im_rgb.convert("RGB")
            w, h = im_rgb.size
            if min(w, h) < min_keep:
                skipped_small += 1
                continue

            scale = target_min / float(min(w, h))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            im_rgb = im_rgb.resize((new_w, new_h), resample=Image.BICUBIC)

            left = (new_w - crop_size) // 2
            top  = (new_h - crop_size) // 2
            im_rgb = im_rgb.crop((left, top, left + crop_size, top + crop_size))

        # ---------------- L ----------------
        with Image.open(l_path) as im_l:
            im_l = im_l.convert("L")
            w2, h2 = im_l.size

            # Safety check: names match, but size might differ—still apply SAME preprocessing rule
            if min(w2, h2) < min_keep:
                skipped_small += 1
                continue

            scale2 = target_min / float(min(w2, h2))
            new_w2 = int(round(w2 * scale2))
            new_h2 = int(round(h2 * scale2))
            im_l = im_l.resize((new_w2, new_h2), resample=Image.BICUBIC)

            left2 = (new_w2 - crop_size) // 2
            top2  = (new_h2 - crop_size) // 2
            im_l = im_l.crop((left2, top2, left2 + crop_size, top2 + crop_size))

        # Save (same filename in both folders)
        im_rgb.save(out_rgb)
        im_l.save(out_l)

        saved += 1

    except Exception:
        # Skip corrupt/unreadable images
        continue

print("Done.")
print(f"Saved pairs: {saved}")
print(f"Skipped (missing L pair): {skipped_missing_pair}")
print(f"Skipped (min side < {min_keep}): {skipped_small}")
print("RGB out:", rgb_out_dir)
print("L out:  ", l_out_dir)
