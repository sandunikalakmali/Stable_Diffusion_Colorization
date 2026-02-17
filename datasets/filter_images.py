import os
import shutil
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# SETTINGS
src_dir = "imagenet/train"              # Source directory (already extracted)
filtered_dir = "imagenet_filtered/train"   # Where to save the colorful filtered images
l_channel_dir = "imagenet_L_filtered/train"         # Where to save L-channel grayscale images
threshold = 12.0

os.makedirs(filtered_dir, exist_ok=True)
os.makedirs(l_channel_dir, exist_ok=True)

def color_variance_metric(img):
    arr = np.array(img).astype(np.float32)
    if arr.ndim < 3 or arr.shape[2] < 3:
        return 0.0
    diff_rg = arr[:,:,0] - arr[:,:,1]
    diff_rb = arr[:,:,0] - arr[:,:,2]
    diff_gb = arr[:,:,1] - arr[:,:,2]
    return (np.var(diff_rg) + np.var(diff_rb) + np.var(diff_gb)) / 3.0

def convert_to_lab_l_channel(img):
    rgb = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab[:,:,0]  # L channel

for class_name in tqdm(os.listdir(src_dir), desc="Filtering by color variance"):
    class_src_path = os.path.join(src_dir, class_name)
    if not os.path.isdir(class_src_path):
        continue
    class_filtered_path = os.path.join(filtered_dir, class_name)
    class_l_path = os.path.join(l_channel_dir, class_name)
    os.makedirs(class_filtered_path, exist_ok=True)
    os.makedirs(class_l_path, exist_ok=True)

    for fname in os.listdir(class_src_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        fpath = os.path.join(class_src_path, fname)
        try:
            with Image.open(fpath) as img:
                img = img.convert('RGB')
                metric = color_variance_metric(img)
                if metric >= threshold:
                    # Save original to filtered folder
                    save_path = os.path.join(class_filtered_path, fname)
                    img.save(save_path)

                    # Convert to Lab and save L channel
                    L = convert_to_lab_l_channel(img)
                    l_save_path = os.path.join(class_l_path, fname)
                    Image.fromarray(L.astype(np.uint8)).save(l_save_path)
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
