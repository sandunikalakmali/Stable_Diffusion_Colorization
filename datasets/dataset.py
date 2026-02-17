from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.utils import read_l_image_to_tensor, list_images_by_stem, read_rgb_to_tensor, rgb_u8_to_lab

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")



class PairedRGBLDataset(Dataset):
    """
    Loads paired RGB + L given:
      rgb_root/<split>/*
      l_root/<split>/*
    Pairing is by stem (filename without extension).
    """
    def __init__(self, rgb_root: str, l_root: str, split: str, image_size: int):
        self.rgb_dir = Path(rgb_root) / split
        self.l_dir = Path(l_root) / split
        if not self.rgb_dir.is_dir():
            raise FileNotFoundError(f"RGB split dir not found: {self.rgb_dir}")
        if not self.l_dir.is_dir():
            raise FileNotFoundError(f"L split dir not found: {self.l_dir}")

        # Build maps stem -> path
        self.rgb_map = list_images_by_stem(self.rgb_dir)
        self.l_map = list_images_by_stem(self.l_dir)

        # Intersect stems
        stems = sorted(set(self.rgb_map.keys()) & set(self.l_map.keys()))
        if not stems:
            raise RuntimeError(f"No paired stems found between {self.rgb_dir} and {self.l_dir}")

        self.items = [(stem, self.rgb_map[stem], self.l_map[stem]) for stem in stems]

        # # Resize/center-crop already done in preprocessing; still enforce fixed size safely
        # self.rgb_tf = transforms.Compose([
        #     transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.CenterCrop(image_size),
        # ])
        # self.l_tf = transforms.Compose([
        #     transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.CenterCrop(image_size),
        # ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem, rgb_path, l_path = self.items[idx]

        rgb_u8 = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        l_u8 = np.array(Image.open(l_path).convert("L"), dtype=np.uint8)

        gt_rgb = read_rgb_to_tensor(rgb_u8)     # (3,H,W) in [-1,1]
        L = read_l_image_to_tensor(l_u8)        # (1,H,W) in [-1,1] (intensity proxy)

        return {
            "stem": stem,
            "rgb_path": str(rgb_path),
            "l_path": str(l_path),
            "gt_rgb": gt_rgb,
            "L": L
        }



class LSplitDataset(Dataset):
    """
    For batch sampling:
      l_root/<split>/*
    Optionally (if rgb_root provided and exists):
      rgb_root/<split>/*
    Pair by stem.
    """
    def __init__(self, l_root: str, split: str, image_size: int, rgb_root: str=None):
        self.l_dir = Path(l_root) / split
        if not self.l_dir.is_dir():
            raise FileNotFoundError(f"L split dir not found: {self.l_dir}")

        self.rgb_dir = None
        if rgb_root is not None:
            cand = Path(rgb_root) / split
            if cand.is_dir():
                self.rgb_dir = cand

        self.l_map = list_images_by_stem(self.l_dir)
        self.rgb_map = list_images_by_stem(self.rgb_dir)

        self.stems = sorted(self.l_map.keys())

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        l_path = self.l_map[stem]

        l_u8 = np.array(Image.open(l_path).convert("L"), dtype=np.uint8)
        L = read_l_image_to_tensor(l_u8)  # (1,H,W) [-1,1]

        out = {"stem": stem, "l_path": str(l_path), "L": L}

        if stem in self.rgb_map:
            rgb_path = self.rgb_map[stem]
            rgb_u8 = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
            gt_rgb = read_rgb_to_tensor(rgb_u8)
            lab_gt = rgb_u8_to_lab(rgb_u8)               # (H,W,3)
            ab_gt = lab_gt[..., 1:3].astype(np.float32)  # (H,W,2)

            out["rgb_path"] = str(rgb_path)
            out["gt_rgb"] = gt_rgb
            out["ab_gt"] = ab_gt
        else:
            out["rgb_path"] = None
            out["ab_gt"] = None

        return out
