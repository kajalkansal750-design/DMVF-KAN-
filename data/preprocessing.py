"""
Domain-Agnostic Preprocessing and Medical Augmentation Pipeline
Implements Section 3 (Equation 1) and Table 1 of the paper:
'Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration'

Normalizes heterogeneous chest X-rays by minimizing acquisition-based intensity shifts
while preserving diagnostically important lung anatomical structures.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class DomainAgnosticIntensityNormalizer:
    """
    Standardizes pixel intensity distribution on a per-institution basis (Eq. 1):
    I_norm = (I_raw - \mu_inst) / \sigma_inst
    Centers distribution around zero with unit variance to harmonize cross-domain data.
    """
    def __init__(self, default_mean: float = 0.505, default_std: float = 0.252):
        self.default_mean = default_mean
        self.default_std = default_std
        # Known institution statistics (can be updated dynamically)
        self.inst_stats = {
            "stanford_chexpert": (0.502, 0.254),
            "san_juan_padchest": (0.518, 0.246),
            "jsrt_japan_us": (0.495, 0.260),
            "covid_radiography": (0.488, 0.255),
            "covid_qu_ex": (0.510, 0.248)
        }

    def __call__(self, img_tensor: torch.Tensor, institution: Optional[str] = None) -> torch.Tensor:
        """
        Args:
            img_tensor: [C, H, W] in [0, 1]
            institution: institution identifier key
        Returns:
            Normalized tensor centered around 0 with unit variance
        """
        if institution and institution in self.inst_stats:
            mu, sigma = self.inst_stats[institution]
        else:
            # Compute empirical sample statistics if institution unknown
            mu = img_tensor.mean().item()
            sigma = img_tensor.std().item() + 1e-7
            
        norm_img = (img_tensor - mu) / sigma
        return norm_img


class CLAHETransform:
    """
    Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    Enhances contrast in local radiographic regions (ground-glass opacities, lung fields).
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert to numpy grayscale
        np_img = np.array(img)
        if np_img.ndim == 3:
            np_img = np_img[:, :, 0]
            
        # Fast local adaptive equalization approximation
        # Normalize to 0-255
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        
        # Simple local window contrast adjustment
        h, w = np_img.shape
        th, tw = self.tile_grid_size
        step_h, step_w = h // th, w // tw
        
        eq_img = np_img.copy()
        for i in range(th):
            for j in range(tw):
                r0, r1 = i * step_h, min((i + 1) * step_h, h)
                c0, c1 = j * step_w, min((j + 1) * step_w, w)
                tile = np_img[r0:r1, c0:c1]
                if tile.size > 0:
                    tile_min = np.percentile(tile, 2)
                    tile_max = np.percentile(tile, 98)
                    if tile_max > tile_min:
                        scaled = np.clip((tile - tile_min) / (tile_max - tile_min) * 255.0, 0, 255)
                        eq_img[r0:r1, c0:c1] = scaled.astype(np.uint8)
                        
        return Image.fromarray(eq_img)


def get_transforms(img_size: int = 224, is_train: bool = True):
    """
    Constructs the augmentation and standardization pipeline as listed in Table 1:
    - Random resized crop
    - Horizontal flip
    - CLAHE
    - Domain-specific color jitter
    - Bilinear resize to fixed target resolution
    """
    normalizer = DomainAgnosticIntensityNormalizer()
    clahe = CLAHETransform()
    
    if is_train:
        def train_transform(img: Image.Image, institution: Optional[str] = None) -> torch.Tensor:
            # Step 1: CLAHE local contrast enhancement
            img = clahe(img)
            # Step 2: Spatial random crop & horizontal flip
            img = TF.resize(img, [int(img_size * 1.1), int(img_size * 1.1)], interpolation=TF.InterpolationMode.BILINEAR)
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(img_size, img_size))
            img = TF.crop(img, i, j, h, w)
            if torch.rand(1).item() > 0.5:
                img = TF.hflip(img)
            # Step 3: Domain-specific subtle color/intensity jitter
            if torch.rand(1).item() > 0.5:
                img = TF.adjust_brightness(img, 1.0 + float(torch.randn(1) * 0.1))
                img = TF.adjust_contrast(img, 1.0 + float(torch.randn(1) * 0.1))
            # Step 4: To Tensor [1, H, W] in [0, 1]
            tensor = TF.to_tensor(img)
            if tensor.shape[0] > 1:
                tensor = tensor[0:1, :, :]  # Grayscale
            # Step 5: Domain-Agnostic Intensity Normalization (Eq. 1)
            norm_tensor = normalizer(tensor, institution)
            return norm_tensor
            
        return train_transform
    else:
        def eval_transform(img: Image.Image, institution: Optional[str] = None) -> torch.Tensor:
            img = clahe(img)
            img = TF.resize(img, [img_size, img_size], interpolation=TF.InterpolationMode.BILINEAR)
            tensor = TF.to_tensor(img)
            if tensor.shape[0] > 1:
                tensor = tensor[0:1, :, :]
            norm_tensor = normalizer(tensor, institution)
            return norm_tensor
            
        return eval_transform
