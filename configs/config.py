"""
Configuration settings for DMVF-KAN++ Framework
Based on Table 1 and experimental specifications from:
'Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration'
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # Image & Feature Extraction
    img_size: int = 224
    in_channels: int = 1
    patch_size: int = 16
    d_model: int = 768          # Feature dimension (d)
    d_k: int = 64               # Attention key/query dimension
    d_f: int = 768              # Attention value / feature dimension
    
    # DTA (Diffusion-Time Attention)
    dta_timesteps: int = 100    # T = 100
    dta_init: str = "linear_decay"
    
    # CMCR (Cross-Modal Conditional Routing)
    metadata_dim: int = 64      # d_m = 64
    cmcr_heads: int = 8         # H = 8 channel groups
    cmcr_dropout: float = 0.2
    
    # DACL (Domain-Adaptive Calibration Layer)
    domain_desc_dim: int = 16   # Scanner, view, institution descriptor dim
    dacl_lambda_init: float = 0.7  # \lambda = 0.7 optimal value
    dacl_hidden_dim: int = 256
    
    # MambaVision Backbone
    mambavision_depth: int = 12
    mambavision_hidden_dim: int = 768
    
    # KAN (Kolmogorov-Arnold Network)
    kan_spline_bases: int = 12   # 12 cubic B-spline bases
    kan_grid_range: tuple = (-2.0, 2.0)
    kan_hidden_dim: int = 512
    dropout_rate: float = 0.2    # Dropout rate after DTA and before KAN
    
    # Multi-Label Spline Output Head
    num_classes: int = 14        # K = 14 thoracic pathology labels
    spline_order: int = 3        # Cubic B-splines (order 3)


@dataclass
class TrainingConfig:
    # Optimization
    learning_rate: float = 3e-4  # \alpha = 3 x 10^-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip_norm: float = 1.0
    batch_size: int = 32         # B = 32
    epochs: int = 150            # E = 150
    warmup_epochs: int = 5
    
    # Loss Weights
    beta_co: float = 0.3         # \beta = 0.3 for co-occurrence loss
    eta_fair: float = 0.2        # \eta = 0.2 for FairGrad++ fairness loss
    fairgrad_gamma: float = 0.05 # \gamma_g = 0.05
    label_smoothing: float = 0.05
    
    # Environment & Reproducibility
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda"


# The 14 Standard Thoracic Pathology Labels (CheXpert & PadChest standard space)
CHEST_XRAY_14_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax"
]

# Demographic subgroups for FairGrad++ fairness evaluation
DEMOGRAPHIC_SUBGROUPS = {
    "sex": ["Male", "Female"],
    "age_bands": ["18-39", "40-59", "60-79", ">=80"]
}
