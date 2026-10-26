"""
End-to-End Training Pipeline for DMVF-KAN++
Strictly implements Algorithm 1 of the paper:
'Domain-Robust Multi-Label Chest X-Ray Classification Via Feature Calibration'

Workflow:
1. Initialize parameters of DTA, CMCR, KAN, DACL, and Spline Output Head.
2. Load pre-trained weights into MambaVision backbone.
3. Initialize AdamW optimizer (lr=3e-4, beta=(0.9, 0.999), weight_decay=1e-4) with 5-epoch warmup.
4. Execute mini-batch training with Composite Loss (L_cls + \beta*L_co + \eta*L_fair).
5. Validation monitoring, checkpointing best mAUROC model, and saving training curve plots.
"""

import os
import argparse
import time
from typing import Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from configs.config import ModelConfig, TrainingConfig, CHEST_XRAY_14_LABELS
from models.dmvf_kan import DMVF_KAN_PlusPlus
from losses.composite_loss import CompositeLoss
from data.datasets import ChestXRayDataset, compute_empirical_cooccurrence
from data.synthetic_benchmarks import create_synthetic_benchmark_suite
from metrics.evaluator import evaluate_multilabel_metrics
from utils.visualization import plot_training_dynamics


def get_lr_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """5-epoch linear warmup to target alpha followed by cosine annealing (Table 1)."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: CompositeLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    accum_steps: int = 1,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
):
    model.train()
    total_loss, total_cls, total_co, total_fair = 0.0, 0.0, 0.0, 0.0
    all_preds, all_targets = [], []
    
    # FairGrad parameters: DACL and Spline head (Eq. 17)
    fairgrad_params = model.get_fairgrad_parameters()
    optimizer.zero_grad()
    use_amp = (device.type == "cuda")
    
    for step, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        metadata = batch["metadata"].to(device)
        domain_desc = batch["domain_desc"].to(device)
        subgroup_ids = batch["subgroup_id"].to(device)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Forward pass (Algorithm 1)
            outputs = model(images, metadata=metadata, domain_desc=domain_desc)
            probs = outputs["probs"]
            
            # Composite Loss: L_cls + \beta*L_co + \eta*L_fair (Eq. 18)
            loss_dict = criterion(
                probs=probs,
                targets=targets,
                subgroup_ids=subgroup_ids,
                head_module=model.head,
                z=outputs["z"]
            )
            loss = loss_dict["loss_total"] / accum_steps

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss_dict["loss_total"].item()
        total_cls += loss_dict["loss_cls"].item()
        total_co += loss_dict["loss_co"].item()
        total_fair += loss_dict["loss_fair"].item()
        
        all_preds.append(probs.detach().float().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        
    n_batches = len(dataloader)
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_targets_np = np.concatenate(all_targets, axis=0)
    
    metrics = evaluate_multilabel_metrics(all_targets_np, all_preds_np)
    return {
        "loss": total_loss / n_batches,
        "loss_cls": total_cls / n_batches,
        "loss_co": total_co / n_batches,
        "loss_fair": total_fair / n_batches,
        "mAUROC": metrics["mAUROC"],
        "Accuracy": metrics["Accuracy"],
        "EMA": metrics["EMA"]
    }


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: CompositeLoss,
    device: torch.device
):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            metadata = batch["metadata"].to(device)
            domain_desc = batch["domain_desc"].to(device)
            
            outputs = model(images, metadata=metadata, domain_desc=domain_desc)
            probs = outputs["probs"]
            
            loss_dict = criterion(probs=probs, targets=targets)
            total_loss += loss_dict["loss_cls"].item()
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    n_batches = len(dataloader)
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_targets_np = np.concatenate(all_targets, axis=0)
    
    metrics = evaluate_multilabel_metrics(all_targets_np, all_preds_np)
    metrics["loss"] = total_loss / n_batches
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train DMVF-KAN++ on Chest X-Rays")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs (Table 1: 150)")
    parser.add_argument("--batch_size", type=int, default=8, help="Physical batch size per step (default 8)")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps (effective batch size = 8 * 4 = 32)")
    parser.add_argument("--dta_timesteps", type=int, default=10, help="DTA progression timesteps (default 10, paper Table 1: 100)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate alpha (Table 1: 3e-4)")
    parser.add_argument("--beta_co", type=float, default=0.3, help="Co-occurrence loss weight beta")
    parser.add_argument("--eta_fair", type=float, default=0.2, help="FairGrad++ loss weight eta")
    parser.add_argument("--data_dir", type=str, default="data_real", help="Path to dataset root")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save weights")
    parser.add_argument("--quick_run", action="store_true", help="Run 3 epochs for fast functional smoke test")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing DMVF-KAN++ Training Pipeline on device: {device}")
    
    # 1. Prepare data (generate benchmark suite if not already present)
    train_csv = os.path.join(args.data_dir, "chexpert", "train.csv")
    val_csv = os.path.join(args.data_dir, "chexpert", "val.csv")
    if not os.path.exists(train_csv):
        print("[*] Creating benchmark dataset suite...")
        create_synthetic_benchmark_suite(args.data_dir, num_samples=120)
        
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    
    train_dataset = ChestXRayDataset(df_train, img_size=224, is_train=True)
    val_dataset = ChestXRayDataset(df_val, img_size=224, is_train=False)
    
    batch_size = min(args.batch_size, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Compute empirical co-occurrence matrix rho_pq from training statistics (Eq. 16)
    print("[*] Estimating empirical disease co-occurrence matrix rho_pq...")
    rho_matrix = compute_empirical_cooccurrence(train_dataset)
    
    # 3. Instantiate DMVF-KAN++ Model
    model_cfg = ModelConfig(dta_timesteps=args.dta_timesteps)
    model = DMVF_KAN_PlusPlus(cfg=model_cfg, variant="full").to(device)
    
    # 4. Instantiate Composite Objective Loss
    criterion = CompositeLoss(
        num_classes=model_cfg.num_classes,
        beta_co=args.beta_co,
        eta_fair=args.eta_fair,
        gamma_subgroup=0.05,
        label_smoothing=0.05,
        rho_matrix=rho_matrix
    ).to(device)
    
    # 5. Optimizer, Scaler & LR Scheduler (Table 1)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    num_epochs = 3 if args.quick_run else args.epochs
    warmup_epochs = 1 if args.quick_run else 5
    scheduler = get_lr_scheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=num_epochs)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    best_val_auroc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    print(f"[*] Starting DMVF-KAN++ training for {num_epochs} epochs (Physical Batch: {batch_size}, Accum: {args.accum_steps}, Effective: {batch_size * args.accum_steps})...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_res = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            accum_steps=args.accum_steps, scaler=scaler
        )
        val_res = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["train_acc"].append(train_res["Accuracy"])
        history["val_acc"].append(val_res["Accuracy"])
        
        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_res['loss']:.4f} (Cls: {train_res['loss_cls']:.4f}, Co: {train_res['loss_co']:.4f}, Fair: {train_res['loss_fair']:.4f}) | "
            f"Train mAUROC: {train_res['mAUROC']:.3f} | Val mAUROC: {val_res['mAUROC']:.3f} | Val Acc: {val_res['Accuracy']:.1f}% | EMA: {val_res['EMA']:.1f}%"
        )
        
        # Save best checkpoint
        if val_res["mAUROC"] > best_val_auroc:
            best_val_auroc = val_res["mAUROC"]
            ckpt_path = os.path.join(args.checkpoint_dir, "best_dmvf_kan.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_mAUROC": best_val_auroc,
                "config": model_cfg
            }, ckpt_path)
            
    total_time = time.time() - start_time
    print(f"[+] Training complete in {total_time/60.0:.2f} min. Best Validation mAUROC: {best_val_auroc:.4f}")
    
    # Save training curves plot (Figure 3)
    plot_training_dynamics(
        history["train_loss"],
        history["val_loss"],
        history["train_acc"],
        history["val_acc"],
        save_path="visualizations/figure3_training_dynamics.png"
    )
    print("[+] Figure 3 saved to: visualizations/figure3_training_dynamics.png")


if __name__ == "__main__":
    main()
