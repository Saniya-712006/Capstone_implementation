"""
src/training/train.py

The main training loop: Adam + ExponentialLR, AMP (fp16) autocast with a
GradScaler, gradient accumulation, and gradient-norm clipping -- all four
matching Table 5 / Section 3.8 of the report. Runs validation once per epoch,
logs every epoch's losses + val RMSE through the ResultsLogger, and saves a
checkpoint whenever validation RMSE improves.
"""

import os
from collections import defaultdict
from typing import Dict

import torch

from configs.config import Config
from src.data.dataset import move_batch_to_device
from src.models.physchem_cal import PhysChemCAL
from src.training.evaluate import evaluate
from src.training.losses import compute_losses
from src.utils.checkpoint import save_checkpoint
from src.utils.results_logger import ResultsLogger


def train(model: PhysChemCAL, data: Dict[str, object], config: Config, device: torch.device,
          results_logger: ResultsLogger, checkpoint_dir: str) -> Dict[str, float]:
    """Train `model` for config.EPOCHS epochs on data["train_loader"], validating each epoch on data["val_loader"].

    Args:
        model: a PhysChemCAL instance, already moved to `device`.
        data: dict returned by src.data.dataset.get_dataloaders() (train_loader,
            val_loader, test_loader, label_mean, label_std, ...).
        config: hyperparameters (learning rate, epochs, loss weights, ...).
        device: torch device to train on.
        results_logger: appends per-epoch metrics to results/YYYY-MM-DD.md.
        checkpoint_dir: where the best-so-far model is saved
            (checkpoint_dir/best_model.pt), overwritten every time val RMSE improves.
    Returns:
        dict with "best_val_rmse" and "best_epoch".
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    label_mean, label_std = data["label_mean"], data["label_std"]
    train_loader = data["train_loader"]

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.LR_GAMMA)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_rmse = float("inf")
    best_epoch = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_losses = defaultdict(list)
        optimizer.zero_grad()
        accumulated = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(
                    batch["atom_ftr"], batch["bond_ftr"], batch["pos"], batch["masses"], batch["mask_matrices"]
                )
                losses = compute_losses(outputs, batch, config)
                loss = losses["total_loss"] / config.ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            accumulated += 1

            for k, v in losses.items():
                epoch_losses[k].append(v.item())

            if accumulated == config.ACCUMULATION_STEPS:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accumulated = 0

        if accumulated > 0:  # flush a partial accumulation window at epoch end
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()

        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        val_metrics = evaluate(model, data["val_loader"], device, label_mean, label_std)
        avg_losses["val_rmse"] = val_metrics["rmse"]

        results_logger.log_epoch(epoch, config.EPOCHS, avg_losses)
        print(f"[train] epoch {epoch}/{config.EPOCHS} "
              f"c={avg_losses['c_loss']:.4f} o={avg_losses['o_loss']:.4f} "
              f"co={avg_losses['co_loss']:.4f} conf={avg_losses['conf_loss']:.4f} "
              f"total={avg_losses['total_loss']:.4f} val_rmse={val_metrics['rmse']:.4f}")

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            save_checkpoint(
                os.path.join(checkpoint_dir, "best_model.pt"), model, optimizer, epoch,
                label_mean, label_std, best_val_rmse,
            )

    results_logger.log_best(best_epoch, best_val_rmse)
    return {"best_val_rmse": best_val_rmse, "best_epoch": best_epoch}
