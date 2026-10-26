"""
Cross-Modal Conditional Routing (CMCR) Module
Implements Section 3.1.2 (Equations 5-8) and Section 3.3 of the paper:
'Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration'

Includes comparison FiLM baseline variant for Table 3c ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalConditionalRouting(nn.Module):
    """
    CMCR: Metadata-conditioned multiplicative feature-modulation mechanism.
    Attenuates metadata-correlated channels via multi-head bounded sigmoid gates
    without additive bias, preventing feature collapse under severe class imbalance.
    """
    def __init__(
        self,
        d_f: int = 768,
        metadata_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        self.d_f = d_f
        self.d_m = metadata_dim
        self.num_heads = num_heads
        
        assert d_f % num_heads == 0, f"d_f ({d_f}) must be divisible by num_heads ({num_heads})"
        self.head_dim = d_f // num_heads
        
        # Complete metadata encoder: \psi(m) = GELU(W_e m + b_e) (Eq. 5)
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, d_f),
            nn.GELU()
        )
        
        # Multi-head conditional modulation parameters W_m^(h), b_m^(h) (Eq. 7)
        self.head_modulators = nn.ModuleList([
            nn.Linear(metadata_dim, self.head_dim)
            for _ in range(num_heads)
        ])
        
        # Learnable residual calibration scalar \gamma \in [0, 1] (Eq. 8)
        # Parameterized as sigmoid(gamma_param) to strictly enforce \gamma \in [0, 1]
        self.gamma_param = nn.Parameter(torch.tensor(0.0))  # sigmoid(0.0) = 0.5 initially
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_f)

    @property
    def gamma(self) -> torch.Tensor:
        """Bounded learnable scalar \gamma in [0, 1]."""
        return torch.sigmoid(self.gamma_param)

    def forward(self, f_DTA: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_DTA: [B, N, d_f] Features from DTA module
            metadata: [B, d_m] Auxiliary demographic & acquisition metadata vector
        Returns:
            f_CMCR_final: [B, N, d_f] Contextually adapted feature sequence
        """
        B, N, d_f = f_DTA.shape
        
        # Multi-head conditional modulation across H groups (Eq. 7)
        # Partition f_DTA into H groups: [B, N, H, head_dim]
        f_groups = f_DTA.view(B, N, self.num_heads, self.head_dim)
        
        modulated_groups = []
        for h in range(self.num_heads):
            # Compute bounded gate \sigma(W_m^(h) m + b_m^(h)) in (0, 1)
            # metadata: [B, d_m] -> gate: [B, 1, head_dim]
            gate_h = torch.sigmoid(self.head_modulators[h](metadata)).unsqueeze(1)
            
            # Element-wise multiplicative attenuation (no additive bias)
            f_h = f_groups[:, :, h, :] * gate_h
            modulated_groups.append(f_h)
            
        # Concatenate along channel dimension (Eq. 7)
        f_CMCR = torch.cat(modulated_groups, dim=-1)  # [B, N, d_f]
        
        # Residual calibration mechanism (Eq. 8):
        # f_CMCR_final = f_DTA + \gamma * (f_CMCR - f_DTA)
        f_CMCR_final = f_DTA + self.gamma * (f_CMCR - f_DTA)
        
        f_CMCR_final = self.norm(self.dropout(f_CMCR_final))
        return f_CMCR_final


class FiLMModule(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) baseline variant for ablation studies (Section 3.3, Table 3c).
    Computes \gamma(m) \odot f + \beta(m) with unbounded scale and additive shift.
    """
    def __init__(self, d_f: int = 768, metadata_dim: int = 64):
        super().__init__()
        self.scale_proj = nn.Linear(metadata_dim, d_f)
        self.shift_proj = nn.Linear(metadata_dim, d_f)

    def forward(self, f: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        # metadata: [B, d_m] -> [B, 1, d_f]
        gamma = self.scale_proj(metadata).unsqueeze(1)
        beta = self.shift_proj(metadata).unsqueeze(1)
        return gamma * f + beta
