"""
Multi-Label Spline-Based Output Head (MLSOH)
Implements Section 3.2.2 (Equation 14) and Section 4.3.3 (Figure 6) of the paper:
'Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration'

Assigns one cubic B-spline regressor per thoracic pathology label k in {1, ..., K}.
Produces continuous probabilities y_hat_k in (0, 1) and interpretable response curves.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.kan import BSplineBasis


class MultiLabelSplineOutputHead(nn.Module):
    """
    MLSOH: Assigns one spline regressor per disease label k in {1, ..., K}.
    \hat{y}_k = S_k(z) = \sigma( \sum_{b=1}^M a_{kb} B_b(z) )  (Eq. 14)
    """
    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 14,
        num_spline_bases: int = 12,
        grid_range: tuple = (-2.0, 2.0)
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_spline_bases = num_spline_bases
        
        self.spline_eval = BSplineBasis(num_spline_bases, grid_range)
        
        # Learnable control point weights a_{kb} for each class k and each input dimension j
        # Shape: [num_classes, in_features, num_spline_bases]
        self.control_points = nn.Parameter(
            torch.randn(num_classes, in_features, num_spline_bases) * 0.02
        )
        
        # Linear residual skip connection for stable gradient flow
        self.linear_skip = nn.Linear(in_features, num_classes)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, z: torch.Tensor, return_logits: bool = False):
        """
        Args:
            z: [B, in_features] Pooled feature representation from KAN
            return_logits: if True, returns raw unscaled logits before sigmoid
        Returns:
            probs: [B, num_classes] Predicted probabilities in (0, 1)
        """
        # Evaluate B-spline bases: [B, in_features, num_spline_bases]
        B_splines = self.spline_eval(z)
        
        # \sum_{b=1}^M a_{kb} B_b(z)
        # Einsum: b: batch, k: classes, j: in_features, m: spline_bases
        spline_logits = torch.einsum("bjm,kjm->bk", B_splines, self.control_points)
        
        logits = spline_logits + self.linear_skip(z) + self.bias
        probs = torch.sigmoid(logits)
        
        if return_logits:
            return probs, logits
        return probs

    def get_spline_curve(self, class_idx: int, feature_idx: int, resolution: int = 100):
        """
        Extracts the 1D continuous spline response curve for a specific feature and disease class.
        Used to generate Figure 6 (Spline-Based Feature Contribution Curves).
        """
        device = self.control_points.device
        grid_min, grid_max = self.spline_eval.grid_range
        x_vals = torch.linspace(grid_min, grid_max, resolution, device=device).unsqueeze(-1)  # [R, 1]
        
        with torch.no_grad():
            b_eval = self.spline_eval(x_vals)  # [R, 1, M]
            cp = self.control_points[class_idx, feature_idx, :].unsqueeze(0).unsqueeze(0)  # [1, 1, M]
            y_vals = torch.sum(b_eval * cp, dim=-1).squeeze().cpu().numpy()
            
        x_np = x_vals.squeeze().cpu().numpy()
        # Normalize to [0, 1] for clinical review visualization
        x_norm = (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)
        y_norm = 1.0 / (1.0 + np.exp(-y_vals))  # Sigmoid response
        return x_norm, y_norm
