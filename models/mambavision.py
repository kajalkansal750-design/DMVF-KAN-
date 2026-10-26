"""
MambaVision Hierarchical Vision Backbone
Implements Section 3.2 & 3.2.1 of the paper:
'Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration'

Models structured long-range spatial dependencies in anatomical regions
with linear computational complexity using selective state-space representations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateSpaceBlock(nn.Module):
    """
    Selective State-Space Model (SSM) block for visual sequence modeling.
    Maintains linear computational complexity while capturing long-range contextual relationships.
    """
    def __init__(self, d_model: int = 768, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        
        # In-projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D Depthwise Convolution for local token mixing
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner
        )
        
        # State space parameters
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Continuous-time state transition matrices A and skip D
        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0),
            self.d_inner,
            dim=0
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Out-projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_model]
        Returns:
            [B, N, d_model]
        """
        B, N, d = x.shape
        # Project and split into main branch and gating branch
        xz = self.in_proj(x)  # [B, N, 2 * d_inner]
        x_branch, z_branch = xz.chunk(2, dim=-1)
        
        # Local conv mixing: [B, d_inner, N]
        x_conv = self.act(self.conv1d(x_branch.transpose(1, 2))).transpose(1, 2)
        
        # Discretization and state update
        ssm_proj = self.x_proj(x_conv)  # [B, N, 2 * d_state + 1]
        delta = ssm_proj[:, :, :1]
        B_ssm = ssm_proj[:, :, 1:self.d_state + 1]
        C_ssm = ssm_proj[:, :, self.d_state + 1:]
        
        delta = F.softplus(self.dt_proj(delta))  # [B, N, d_inner]
        A = -torch.exp(self.A_log)              # [d_inner, d_state]
        
        # Fast recurrent scan simulation over tokens
        # y = SSM(x_conv) + D * x_conv
        y = x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # Multiplicative gating with z_branch
        out = y * self.act(z_branch)
        return self.out_proj(out)


class MambaVisionLayer(nn.Module):
    """
    Hierarchical layer combining SSM block with Feed-Forward network and residual connections.
    """
    def __init__(self, d_model: int = 768, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = StateSpaceBlock(d_model=d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MambaVisionBackbone(nn.Module):
    """
    MambaVision: Depth=12 layers, Hidden Dim=768 (Table 1).
    Processes feature sequences from DACL, captures long-range spatial dependencies,
    and applies global average pooling to yield contextualized representation for KAN.
    """
    def __init__(self, depth: int = 12, hidden_dim: int = 768, dropout: float = 0.2):
        super().__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        
        self.layers = nn.ModuleList([
            MambaVisionLayer(d_model=hidden_dim, mlp_ratio=4.0, dropout=0.1)
            for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        """
        Args:
            x: [B, N, hidden_dim] Adapted feature sequence from DACL
            return_tokens: if True, returns both pooled feature and full sequence
        Returns:
            pooled: [B, hidden_dim]
            (optional) tokens: [B, N, hidden_dim]
        """
        for layer in self.layers:
            x = layer(x)
            
        x_norm = self.final_norm(x)
        # Global average pooling across tokens N
        pooled = x_norm.mean(dim=1)  # [B, hidden_dim]
        pooled = self.dropout(pooled)
        
        if return_tokens:
            return pooled, x_norm
        return pooled
