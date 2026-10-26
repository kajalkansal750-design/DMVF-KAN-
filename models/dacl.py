"""
Domain-Adaptive Calibration Layer (DACL) Module
Implements Section 3.2.1 (Equations 11-13) and Section 3.3 of the paper:
'Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration'

Includes DANN and Deep CORAL baseline variants for Table 3c ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdaptiveCalibrationLayer(nn.Module):
    """
    DACL: Performs feature-level domain correction using observable domain descriptors
    (scanner type, view position PA/AP, institution ID).
    Target-free at inference time without requiring adversarial optimization or target batches.
    """
    def __init__(
        self,
        d_f: int = 768,
        domain_dim: int = 16,
        hidden_dim: int = 256,
        lambda_init: float = 0.7
    ):
        super().__init__()
        self.d_f = d_f
        self.domain_dim = domain_dim
        
        # Two-layer MLP calibration function: \phi_\theta(d) = W_2 \delta(W_1 d + b_1) + b_2 (Eq. 12)
        self.phi_theta = nn.Sequential(
            nn.Linear(domain_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_f)
        )
        
        # Learnable calibration parameter \lambda \in [0, 1] (Eq. 13)
        # Optimal value found via sensitivity analysis = 0.7
        # We parameterize via logit so \lambda = sigmoid(logit)
        logit_init = torch.log(torch.tensor(lambda_init / (1.0 - lambda_init)))
        self.lambda_logit = nn.Parameter(logit_init)

    @property
    def lambda_val(self) -> torch.Tensor:
        """Controlled parameter \lambda \in [0, 1]."""
        return torch.sigmoid(self.lambda_logit)

    def forward(self, f_CMCR: torch.Tensor, domain_desc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_CMCR: [B, N, d_f] Features from CMCR module
            domain_desc: [B, domain_dim] Domain descriptor vector (scanner, view, institution)
        Returns:
            f_adapted: [B, N, d_f] Calibrated feature sequence
        """
        # \phi_\theta(d): [B, d_f] -> broadcast to [B, 1, d_f]
        correction = self.phi_theta(domain_desc).unsqueeze(1)
        
        # Residual blending mechanism (Eq. 13):
        # f_adapted = (1 - \lambda) * f_CMCR + \lambda * [f_CMCR - \phi_\theta(d)]
        #           = f_CMCR - \lambda * \phi_\theta(d)
        f_adapted = (1.0 - self.lambda_val) * f_CMCR + self.lambda_val * (f_CMCR - correction)
        
        return f_adapted


# --- Baselines for Table 3c Ablation Comparison ---

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DANNModule(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN) discriminator baseline (Section 3.3, Table 3c).
    """
    def __init__(self, d_f: int = 768, num_domains: int = 5):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(d_f, 256),
            nn.ReLU(),
            nn.Linear(256, num_domains)
        )

    def forward(self, f: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # Global average pool feature tokens: [B, N, d_f] -> [B, d_f]
        f_pool = f.mean(dim=1)
        reversed_f = GradientReversalFunction.apply(f_pool, alpha)
        return self.discriminator(reversed_f)


def deep_coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Deep CORAL covariance alignment loss (Section 3.3, Table 3c).
    """
    d = source.size(1)
    # Subtraction of mean
    src_mean = torch.mean(source, dim=0, keepdim=True)
    tgt_mean = torch.mean(target, dim=0, keepdim=True)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    
    # Covariance matrices
    cov_src = (src_centered.t() @ src_centered) / (source.size(0) - 1 + 1e-7)
    cov_tgt = (tgt_centered.t() @ tgt_centered) / (target.size(0) - 1 + 1e-7)
    
    # Frobenius norm squared
    loss = torch.sum((cov_src - cov_tgt) ** 2) / (4 * d * d)
    return loss
