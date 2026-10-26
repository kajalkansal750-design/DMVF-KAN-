"""
DMVF-KAN++: Domain-Calibrated Multi-Label Learning Framework
Implements Section 3, Section 3.2, and Algorithm 1 of the paper:
'Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration'

Integrates:
- DTA: Progression-Aware Feature Refinement via Diffusion-Time Attention (Eq. 2-4)
- CMCR: Cross-Modal Conditional Routing with Bounded Attenuation (Eq. 5-8)
- DACL: Domain-Adaptive Calibration Layer with Observable Descriptors (Eq. 11-13)
- MambaVision: Linear-Complexity Sequence Vision Backbone
- KAN: Kolmogorov-Arnold Network Spline Transformation (Eq. 9-10)
- MLSOH: Multi-Label Spline-Based Output Head (Eq. 14)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import ModelConfig
from models.dta import DiffusionTimeAttention
from models.cmcr import CrossModalConditionalRouting, FiLMModule
from models.dacl import DomainAdaptiveCalibrationLayer, DANNModule
from models.mambavision import MambaVisionBackbone
from models.kan import KolmogorovArnoldNetwork
from models.spline_head import MultiLabelSplineOutputHead


class DMVF_KAN_PlusPlus(nn.Module):
    """
    Proposed DMVF-KAN++ Framework (Figure 1, Algorithm 1).
    Supports full configuration and modular ablation variants (Tables 3b and 3c).
    """
    def __init__(
        self,
        cfg: Optional[ModelConfig] = None,
        variant: str = "full"
    ):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg
        self.variant = variant
        
        # 1. Feature Extraction / DTA
        self.use_dta = variant not in ["baseline", "no_dta"]
        if self.use_dta:
            self.dta = DiffusionTimeAttention(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                in_channels=cfg.in_channels,
                d_model=cfg.d_model,
                d_k=cfg.d_k,
                d_f=cfg.d_f,
                num_timesteps=cfg.dta_timesteps,
                dropout=cfg.dropout_rate
            )
        else:
            # Fallback standard patch embedding when DTA is ablated
            self.patch_proj = nn.Conv2d(cfg.in_channels, cfg.d_f, kernel_size=cfg.patch_size, stride=cfg.patch_size)
            self.patch_norm = nn.LayerNorm(cfg.d_f)
            
        # 2. Metadata-Aware Modulation (CMCR vs FiLM vs None)
        self.use_cmcr = variant in ["full", "b+cmcr", "c+dacl", "d+mlsoh", "no_dta", "no_dacl", "no_mlsoh", "dann_dacl", "coral_dacl"]
        self.use_film = (variant == "film_cmcr")
        if self.use_cmcr:
            self.cmcr = CrossModalConditionalRouting(
                d_f=cfg.d_f,
                metadata_dim=cfg.metadata_dim,
                num_heads=cfg.cmcr_heads,
                dropout=cfg.cmcr_dropout
            )
        elif self.use_film:
            self.film = FiLMModule(d_f=cfg.d_f, metadata_dim=cfg.metadata_dim)

        # 3. Domain-Adaptive Calibration (DACL vs DANN vs CORAL vs None)
        self.use_dacl = variant in ["full", "c+dacl", "d+mlsoh", "no_dta", "no_cmcr", "no_mlsoh", "film_cmcr"]
        self.use_dann = (variant == "dann_dacl")
        if self.use_dacl:
            self.dacl = DomainAdaptiveCalibrationLayer(
                d_f=cfg.d_f,
                domain_dim=cfg.domain_desc_dim,
                hidden_dim=cfg.dacl_hidden_dim,
                lambda_init=cfg.dacl_lambda_init
            )
        elif self.use_dann:
            self.dann = DANNModule(d_f=cfg.d_f)

        # 4. Hierarchical Sequence Backbone (MambaVision)
        self.backbone = MambaVisionBackbone(
            depth=cfg.mambavision_depth,
            hidden_dim=cfg.mambavision_hidden_dim,
            dropout=cfg.dropout_rate
        )
        
        # 5. Kolmogorov-Arnold Network (KAN)
        self.kan = KolmogorovArnoldNetwork(
            in_dim=cfg.mambavision_hidden_dim,
            hidden_dim=cfg.kan_hidden_dim,
            out_dim=cfg.kan_hidden_dim,
            num_spline_bases=cfg.kan_spline_bases,
            dropout=cfg.dropout_rate
        )
        
        # 6. Multi-Label Classification Head (MLSOH vs Linear)
        self.use_mlsoh = variant in ["full", "d+mlsoh", "no_dta", "no_cmcr", "no_dacl", "film_cmcr", "dann_dacl", "coral_dacl"]
        if self.use_mlsoh:
            self.head = MultiLabelSplineOutputHead(
                in_features=cfg.kan_hidden_dim,
                num_classes=cfg.num_classes,
                num_spline_bases=cfg.kan_spline_bases,
                grid_range=cfg.kan_grid_range
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(cfg.dropout_rate),
                nn.Linear(cfg.kan_hidden_dim, cfg.num_classes),
                nn.Sigmoid()
            )

    def get_fairgrad_parameters(self):
        """
        Returns parameters theta on which gradient discrepancy is measured (Eq. 17).
        The paper specifically specifies: 'specifically, the parameters of DACL and the spline output head'.
        """
        params = []
        if self.use_dacl:
            params.extend(list(self.dacl.parameters()))
        if self.use_mlsoh:
            params.extend(list(self.head.parameters()))
        return params

    def forward(
        self,
        x: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        domain_desc: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Executes end-to-end forward path according to Algorithm 1.
        Args:
            x: [B, C, H, W] Input Chest X-Ray
            metadata: [B, metadata_dim] Demographic & acquisition metadata
            domain_desc: [B, domain_desc_dim] Domain descriptor vector
        Returns:
            Dictionary containing predicted probabilities 'probs', intermediate features, etc.
        """
        B = x.shape[0]
        device = x.device
        
        # Handle default metadata and domain descriptors if none provided
        if metadata is None:
            metadata = torch.zeros(B, self.cfg.metadata_dim, device=device)
        if domain_desc is None:
            domain_desc = torch.zeros(B, self.cfg.domain_desc_dim, device=device)
            
        # Step 1: Feature Extraction {Z_t} <- DTA(x_norm)
        if self.use_dta:
            f_dta = self.dta(x)  # [B, N, d_f]
        else:
            p = self.patch_proj(x).flatten(2).transpose(1, 2)
            f_dta = self.patch_norm(p)
            
        # Step 2: Metadata-aware Modulation Z_tilde <- CMCR({Z_t}, u)
        if self.use_cmcr:
            f_cmcr = self.cmcr(f_dta, metadata)
        elif self.use_film:
            f_cmcr = self.film(f_dta, metadata)
        else:
            f_cmcr = f_dta
            
        # Step 3: Domain-Adaptive Calibration h <- DACL(Z_tilde, d)
        if self.use_dacl:
            f_adapted = self.dacl(f_cmcr, domain_desc)
        else:
            f_adapted = f_cmcr
            
        # Step 4: Backbone + KAN
        # f <- MambaVision(h; \theta_MV)
        f_mamba, tokens = self.backbone(f_adapted, return_tokens=True)  # [B, d_hidden], [B, N, d_hidden]
        
        # z <- KAN(f)
        z = self.kan(f_mamba)  # [B, d_kan]
        
        # Step 5: Multi-label Prediction y_hat_k <- \sigma(\sum_j w_kj \phi_j(z))
        if self.use_mlsoh:
            probs = self.head(z)
        else:
            probs = self.head(z)
            
        out = {
            "probs": probs,
            "z": z,
            "f_adapted": f_adapted,
            "f_cmcr": f_cmcr,
            "f_dta": f_dta,
            "tokens": tokens
        }
        
        if self.use_dann:
            out["domain_pred"] = self.dann(f_adapted)
            
        return out
