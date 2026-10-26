"""
Kolmogorov-Arnold Network (KAN) Module
Implements Section 3.2.1 (Equations 9-10) of the paper:
'Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration'

Uses structured nonlinear spline-based transformations (12 cubic B-spline bases)
to provide interpretable functional decomposition of high-level features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineBasis(nn.Module):
    """
    Evaluates 12 cubic B-spline basis functions over a uniform grid.
    Order k = 3 (cubic).
    """
    def __init__(self, num_bases: int = 12, grid_range: tuple = (-2.0, 2.0)):
        super().__init__()
        self.num_bases = num_bases
        self.grid_range = grid_range
        
        grid_min, grid_max = grid_range
        step = (grid_max - grid_min) / max(num_bases - 3, 1)
        knots = torch.linspace(grid_min - 3 * step, grid_max + 3 * step, num_bases + 4)
        self.register_buffer("knots", knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates B-splines using Cox-de Boor recursion.
        Args:
            x: [..., d_in]
        Returns:
            bases: [..., d_in, num_bases]
        """
        x_exp = x.unsqueeze(-1)  # [..., d_in, 1]
        knots = self.knots
        
        # 0-th order splines (indicator step functions)
        b = ((x_exp >= knots[:-1]) & (x_exp < knots[1:])).float()
        
        # Recursive evaluation up to cubic (order p=1, 2, 3)
        for p in range(1, 4):
            n = b.shape[-1] - 1
            t_i = knots[:n]
            t_ip = knots[p:p + n]
            t_i1 = knots[1:n + 1]
            t_ip1 = knots[p + 1:p + 1 + n]
            
            denom1 = torch.where(t_ip == t_i, torch.ones_like(t_i), t_ip - t_i)
            denom2 = torch.where(t_ip1 == t_i1, torch.ones_like(t_i1), t_ip1 - t_i1)
            
            b = ((x_exp - t_i) / denom1) * b[..., :n] + ((t_ip1 - x_exp) / denom2) * b[..., 1:n + 1]
            
        return b[..., :self.num_bases]


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network Layer:
    Computes \hat{y}_i = \sum_j [ w_base * b(x_j) + \sum_{b=1}^M c_{ijb} * B_b(x_j) ] (Eq. 9-10)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_spline_bases: int = 12,
        grid_range: tuple = (-2.0, 2.0),
        base_act = nn.SiLU
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_spline_bases = num_spline_bases
        
        self.spline_eval = BSplineBasis(num_spline_bases, grid_range)
        
        # Base residual weight (SiLU transformation)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_act = base_act()
        
        # Spline coefficient tensor G: [out_features, in_features, num_spline_bases]
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, num_spline_bases))
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]
        Returns:
            [B, out_features]
        """
        # Base linear term
        base_out = F.linear(self.base_act(x), self.base_weight)  # [B, out_features]
        
        # Spline term:
        # B_spline: [B, in_features, num_spline_bases]
        splines = self.spline_eval(x)
        
        # Matrix form y = G \cdot \Phi_F(x) (Eq. 10)
        # einsum: b: batch, i: out_features, j: in_features, m: spline_bases
        spline_out = torch.einsum("bjm,ijm->bi", splines, self.spline_weight)
        
        return base_out + spline_out + self.bias


class KolmogorovArnoldNetwork(nn.Module):
    """
    Complete KAN module with Dropout (Table 1: Dropout Rate 0.2 before KAN)
    and stacked KANLinear layers for structured nonlinear feature transformation.
    """
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 512,
        out_dim: int = 512,
        num_spline_bases: int = 12,
        dropout: float = 0.2
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.kan1 = KANLinear(in_dim, hidden_dim, num_spline_bases=num_spline_bases)
        self.norm = nn.LayerNorm(hidden_dim)
        self.kan2 = KANLinear(hidden_dim, out_dim, num_spline_bases=num_spline_bases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.kan1(x)
        x = self.norm(x)
        x = self.kan2(x)
        return x
