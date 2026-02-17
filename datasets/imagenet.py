from pathlib import Path
from PIL import Image
from tqdm import tqdm
import math
from collections import Counter

root = Path("./datasets/preprocessed_imagenet_filtered_flat/train")

count_total = 0
count_shorter_lt_512 = 0
count_longer_lt_512 = 0
count_both_lt_512 = 0

min_width = math.inf
min_width_img = None

width_counter = Counter()

exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
img_paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

for img_path in tqdm(img_paths):
    try:
        with Image.open(img_path) as img:
            w, h = img.size
    except Exception:
        continue

    count_total += 1

    # track minimum width
    if w < min_width:
        min_width = w
        min_width_img = img_path

    # count widths
    width_counter[w] += 1

    if min(h, w) < 512:
        count_shorter_lt_512 += 1

    if max(h, w) < 512:
        count_longer_lt_512 += 1

    if h < 512 and w < 512:
        count_both_lt_512 += 1

# ---- stats ----
print(f"Total images: {count_total}")
print(f"Shorter side < 512: {count_shorter_lt_512}")
print(f"Longer side < 512: {count_longer_lt_512}")
print(f"Both sides < 512: {count_both_lt_512}")

print(f"Minimum width found: {min_width}")
print(f"Image with minimum width: {min_width_img}")

# ---- width distribution ----
most_common_width, most_common_count = width_counter.most_common(1)[0]

print(f"Number of unique widths: {len(width_counter)}")
print(f"Most frequent width: {most_common_width}")
print(f"Count of most frequent width: {most_common_count}")
print(f"Percentage of dataset: {100 * most_common_count / count_total:.2f}%")
