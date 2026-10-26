"""
Diffusion-Time Attention (DTA) Module for Multi-Scale Feature Extraction
Implements Section 3.1.1 (Equations 1-4) of the paper:
'Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration'
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and projects to feature dimension d.
    h^(0) = E(I_norm)
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 1, d_model: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B = x.shape[0]
        tokens = self.proj(x)  # [B, d, H/P, W/P]
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, N, d]
        tokens = self.norm(tokens) + self.pos_embed
        return tokens


class DenoisingDiffusionDynamics(nn.Module):
    """
    Models progressive denoising feature states {x_t}_{t=1}^T across diffusion timesteps.
    Simulates multi-scale progressive feature refinement dynamics of DiT++ backbone.
    """
    def __init__(self, d_model: int = 768, num_timesteps: int = 100):
        super().__init__()
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        
        # Intermediate refinement transformer block to evolve representations
        self.refinement_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.step_scale = nn.Parameter(torch.linspace(0.01, 0.1, num_timesteps))

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        """
        Generates sequence of latent feature states {x_t}_{t=1}^T.
        h0: [B, N, d]
        Returns: [B, T, N, d]
        """
        B, N, d = h0.shape
        states = []
        cur_state = h0
        for t in range(self.num_timesteps):
            # Progressive feature refinement at noise level t
            delta = self.refinement_mlp(cur_state)
            cur_state = cur_state + self.step_scale[t] * delta
            states.append(cur_state)
        # Stack along timestep dimension
        return torch.stack(states, dim=1)  # [B, T, N, d]


class DiffusionTimeAttention(nn.Module):
    """
    Diffusion-Time Attention (DTA) Module:
    Captures dependencies across denoising timesteps via temporal attention (Eq. 2)
    and aggregates features with learnable weights w_t initialized via linear decay (Eq. 3, 4).
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        d_model: int = 768,
        d_k: int = 64,
        d_f: int = 768,
        num_timesteps: int = 100,
        dropout: float = 0.2
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_f = d_f
        self.T = num_timesteps
        
        # Patch embedding E(I_norm)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        self.dynamics = DenoisingDiffusionDynamics(d_model, num_timesteps)
        
        # Timestep-specific learnable encoding vector e_t \in R^d (Eq. 4)
        self.time_embed = nn.Parameter(torch.randn(num_timesteps, d_model) * 0.02)
        
        # Shared projection matrices W_Q, W_K \in R^{d x d_k}, W_V \in R^{d x d_f} (Eq. 2)
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_f, bias=False)
        
        # Unconstrained learnable logits \omega_t initialized with linear-decay scheme (Eq. 3, Table 1)
        # Linear decay: \omega_t starts higher for early/refined steps and linearly decays
        decay_init = torch.linspace(1.0, 0.1, num_timesteps)
        self.omega = nn.Parameter(decay_init)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_f)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_norm: [B, C, H, W] Domain-normalized Chest X-Ray image
        Returns:
            f_DTA: [B, N, d_f] Aggregated multi-scale diffusion feature representation
        """
        B = x_norm.shape[0]
        # Step 1: Initial token sequence h^(0) = E(I_norm)
        h0 = self.patch_embed(x_norm)  # [B, N, d]
        N = h0.shape[1]
        
        # Softmax aggregation weights w_t (Eq. 3)
        # w_t = exp(\omega_t) / \sum_{t'=1}^T exp(\omega_t')
        w_t = F.softmax(self.omega, dim=0)  # [T]
        
        # Memory-efficient iterative accumulation across timesteps:
        # Avoids allocating massive [B, T, N, d] and [B, T, N, N] tensors all at once.
        # Mathematically identical to Eq. (2)-(4).
        f_DTA = torch.zeros(B, N, self.d_f, device=x_norm.device, dtype=x_norm.dtype)
        cur_state = h0
        
        for t in range(self.T):
            # Progressive feature refinement at noise level t
            delta = self.dynamics.refinement_mlp(cur_state)
            cur_state = cur_state + self.dynamics.step_scale[t] * delta
            
            # Timestep encoding (Eq. 4): x_tilde_t = x_t + 1_N * e_t^T
            e_t = self.time_embed[t].unsqueeze(0).unsqueeze(0)  # [1, 1, d]
            x_tilde_t = cur_state + e_t                         # [B, N, d]
            
            # Query, Key, Value projections (Eq. 2)
            Q_t = self.W_Q(x_tilde_t)  # [B, N, d_k]
            K_t = self.W_K(x_tilde_t)  # [B, N, d_k]
            V_t = self.W_V(x_tilde_t)  # [B, N, d_f]
            
            # Attention map: a_t = softmax((Q_t K_t^T) / sqrt(d_k)) \in R^{N x N}
            scores_t = torch.matmul(Q_t, K_t.transpose(-2, -1)) / math.sqrt(self.d_k)
            a_t = F.softmax(scores_t, dim=-1)
            
            # Timestep-attended features: z_t = a_t V_t \in R^{N x d_f}
            z_t = torch.matmul(a_t, V_t)
            
            # Accumulate with learnable weight w_t (Eq. 3)
            f_DTA = f_DTA + w_t[t] * z_t
            
        f_DTA = self.layer_norm(self.dropout(f_DTA))
        return f_DTA
