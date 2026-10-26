"""
Composite Objective Loss Formulation for DMVF-KAN++
Implements Section 3.2.2 (Equations 15-18) of the paper:
'Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration'

L_total = L_cls + \beta * L_co + \eta * L_fair
- L_cls: Binary cross-entropy with label smoothing (Eq. 15)
- L_co: Dependency-aware disease co-occurrence regularizer (Eq. 16)
- L_fair: FairGrad++ subgroup gradient discrepancy penalty (Eq. 17)
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    """
    Composite Multi-Objective Loss:
    Balances multi-label classification accuracy, clinical disease co-occurrence dependencies,
    and demographic subgroup gradient consistency.
    """
    def __init__(
        self,
        num_classes: int = 14,
        beta_co: float = 0.3,
        eta_fair: float = 0.2,
        gamma_subgroup: float = 0.05,
        label_smoothing: float = 0.05,
        rho_matrix: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta_co
        self.eta = eta_fair
        self.gamma_g = gamma_subgroup
        self.label_smoothing = label_smoothing
        
        # Empirical disease correlation matrix \rho_{pq} \in [-1, 1] (Eq. 16)
        if rho_matrix is None:
            # Default identity prior if uninitialized, updated from training statistics
            rho_matrix = torch.eye(num_classes)
        self.register_buffer("rho", rho_matrix)
        
        # Build pair index list for all unique (p, q) pairs with p < q
        pairs = []
        for p in range(num_classes):
            for q in range(p + 1, num_classes):
                pairs.append((p, q))
        self.pairs = pairs

    def set_correlation_matrix(self, rho_matrix: torch.Tensor):
        """Updates empirical correlation matrix estimated from training set."""
        self.rho.copy_(rho_matrix.to(self.rho.device))

    def compute_classification_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Binary Cross-Entropy Loss with Label Smoothing (Eq. 15).
        targets: [B, K] in {0, 1}
        probs: [B, K] in (0, 1)
        """
        # Apply label smoothing: y_smooth = y * (1 - \epsilon) + 0.5 * \epsilon
        if self.label_smoothing > 0:
            smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smoothed_targets = targets
            
        eps = 1e-7
        p_clamped = torch.clamp(probs, eps, 1.0 - eps)
        bce = -(smoothed_targets * torch.log(p_clamped) + (1.0 - smoothed_targets) * torch.log(1.0 - p_clamped))
        return bce.sum(dim=-1).mean()  # Mean over batch

    def compute_cooccurrence_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Dependency-aware Co-occurrence Regularization (Eq. 16):
        L_co = \sum_{(p,q) \in P} \lambda_{pq} || \hat{y}_p \hat{y}_q - \rho_{pq} y_p y_q ||_2^2
        """
        B = probs.shape[0]
        loss_co = torch.tensor(0.0, device=probs.device)
        
        for p, q in self.pairs:
            pred_co = probs[:, p] * probs[:, q]               # \hat{y}_p \hat{y}_q
            rho_pq = self.rho[p, q]
            true_co = rho_pq * (targets[:, p] * targets[:, q]) # \rho_{pq} y_p y_q
            diff = pred_co - true_co
            loss_co = loss_co + torch.mean(diff ** 2)
            
        num_pairs = max(len(self.pairs), 1)
        return loss_co / num_pairs

    def compute_fairgrad_loss(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        subgroup_ids: Optional[torch.Tensor],
        params_theta: Optional[List[nn.Parameter]] = None,
        head_module: Optional[nn.Module] = None,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        FairGrad++ Subgroup Gradient Discrepancy Loss (Eq. 17):
        L_fair = \sum_{g \in G} \gamma_g || \nabla_\theta L_{cls}^(g) - 1/|G| \sum_{g'} \nabla_\theta L_{cls}^(g') ||_2^2
        """
        if subgroup_ids is None:
            return torch.tensor(0.0, device=probs.device)
            
        unique_groups = torch.unique(subgroup_ids)
        if len(unique_groups) <= 1:
            # If only 1 subgroup is present in the mini-batch, disparity is zero
            return torch.tensor(0.0, device=probs.device)
            
        if head_module is not None and z is not None:
            # Memory-efficient & exact: evaluate head on detached pooled feature z
            target_params = list(head_module.parameters())
        elif params_theta is not None and len(params_theta) > 0:
            target_params = params_theta
        else:
            return torch.tensor(0.0, device=probs.device)
            
        subgroup_grads = []
        for g in unique_groups:
            mask = (subgroup_ids == g)
            if mask.sum() == 0:
                continue
            
            if head_module is not None and z is not None:
                z_g = z[mask].detach()
                probs_g = head_module(z_g)
            else:
                probs_g = probs[mask]
                
            targets_g = targets[mask]
            loss_g = self.compute_classification_loss(probs_g, targets_g)
            
            # Compute gradients w.r.t target parameters (Eq. 17)
            grads = torch.autograd.grad(
                loss_g,
                target_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
            
            # Flatten gradients into a single vector
            grad_vecs = []
            for grad, p in zip(grads, target_params):
                if grad is not None:
                    grad_vecs.append(grad.reshape(-1))
                else:
                    grad_vecs.append(torch.zeros_like(p).reshape(-1))
            flat_grad = torch.cat(grad_vecs)
            subgroup_grads.append(flat_grad)
            
        if len(subgroup_grads) <= 1:
            return torch.tensor(0.0, device=probs.device)
            
        # Stack gradients: [num_active_groups, num_params]
        grad_matrix = torch.stack(subgroup_grads, dim=0)
        mean_grad = torch.mean(grad_matrix, dim=0, keepdim=True)
        
        # Discrepancy: \sum_{g \in G} \gamma_g || \nabla_\theta L_g - mean_grad ||_2^2
        diff = grad_matrix - mean_grad
        loss_fair = self.gamma_g * torch.sum(diff ** 2) / len(subgroup_grads)
        return loss_fair

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        subgroup_ids: Optional[torch.Tensor] = None,
        fairgrad_params: Optional[List[nn.Parameter]] = None,
        head_module: Optional[nn.Module] = None,
        z: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes joint multi-objective loss:
        L_total = L_cls + \beta * L_co + \eta * L_fair (Eq. 18)
        """
        loss_cls = self.compute_classification_loss(probs, targets)
        loss_co = self.compute_cooccurrence_loss(probs, targets)
        
        if self.eta > 0 and subgroup_ids is not None:
            loss_fair = self.compute_fairgrad_loss(
                probs=probs,
                targets=targets,
                subgroup_ids=subgroup_ids,
                params_theta=fairgrad_params,
                head_module=head_module,
                z=z
            )
        else:
            loss_fair = torch.tensor(0.0, device=probs.device)
            
        loss_total = loss_cls + self.beta * loss_co + self.eta * loss_fair
        
        return {
            "loss_total": loss_total,
            "loss_cls": loss_cls,
            "loss_co": loss_co,
            "loss_fair": loss_fair
        }
