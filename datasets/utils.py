from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage import color

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images_by_stem(folder: Path):
    """
    Returns dict: stem -> path (first occurrence by sorted order)
    """
    if folder is None:
        return {}
    paths = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    m = {}
    for p in paths:
        if p.stem not in m:
            m[p.stem] = p
    return m


def read_rgb_to_tensor(rgb_u8: np.ndarray) -> torch.Tensor:
    """
    rgb_u8: (H,W,3) uint8
    returns: (3,H,W) float in [-1,1]
    """
    t = torch.from_numpy(rgb_u8).permute(2, 0, 1).float() / 255.0
    t = t * 2.0 - 1.0
    return t


def read_l_image_to_tensor(l_u8: np.ndarray) -> torch.Tensor:
    """
    l_u8: (H,W) uint8
    returns: (1,H,W) float in [-1,1]
    (Treat as luminance/intensity proxy; your L images are already prepared.)
    """
    t = torch.from_numpy(l_u8).unsqueeze(0).float() / 255.0
    t = t * 2.0 - 1.0
    return t


def tensor_to_rgb_u8(img_3chw: torch.Tensor) -> np.ndarray:
    """
    img_3chw: (3,H,W) float in [0,1] or [-1,1] depending; we assume [0,1]
    """
    x = img_3chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_rgb_u8(rgb_u8: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_u8).save(path)


def rgb_u8_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """
    rgb_u8: (H,W,3) uint8
    returns lab: (H,W,3) float32 with L in [0,100], a,b ~ [-128,127]
    """
    rgb01 = rgb_u8.astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb01).astype(np.float32)
    return lab


def lab_to_rgb_u8(lab: np.ndarray) -> np.ndarray:
    """
    lab: (H,W,3) float32 (L 0..100)
    returns rgb_u8: (H,W,3) uint8
    """
    rgb01 = color.lab2rgb(lab).astype(np.float32)  # skimage returns [0,1] (may go slightly out)
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    return (rgb01 * 255.0 + 0.5).astype(np.uint8)


def L_tensor_minus1_1_to_L100(L_1hw: torch.Tensor) -> np.ndarray:
    """
    L_1hw: torch (1,H,W) in [-1,1]
    returns: numpy (H,W) float32 in [0,100]
    """
    L01 = (L_1hw.clamp(-1, 1) + 1.0) * 0.5
    L100 = (L01 * 100.0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return L100


# def rescale_ab_to_gt_range(ab_gen: np.ndarray, ab_gt: np.ndarray, eps: float = 1e-6) -> np.ndarray:
#     """
#     ab_gen: (H,W,2) float32
#     ab_gt:  (H,W,2) float32
#     Returns ab_gen mapped channelwise so its min/max match ab_gt min/max.

#     If gen channel range is ~0, falls back to gt mean for that channel.
#     """
#     out = np.empty_like(ab_gen, dtype=np.float32)
#     for c in range(2):
#         g = ab_gen[..., c]
#         t = ab_gt[..., c]

#         gmin, gmax = float(g.min()), float(g.max())
#         tmin, tmax = float(t.min()), float(t.max())

#         if (gmax - gmin) < eps:
#             out[..., c] = float(t.mean())
#         else:
#             out[..., c] = (g - gmin) / (gmax - gmin) * (tmax - tmin) + tmin

#     return out