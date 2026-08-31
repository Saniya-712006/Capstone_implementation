"""
src/training/train.py

The main training loop: Adam + ExponentialLR, AMP (fp16) autocast with a
GradScaler, gradient accumulation, and gradient-norm clipping -- all four
matching Table 5 / Section 3.8 of the report. Runs validation once per epoch,
logs every epoch's losses + val RMSE through the ResultsLogger, and saves a
checkpoint every epoch.

Two checkpoints are kept, for two different purposes:
  - latest_model.pt: overwritten every single epoch, regardless of whether
    val RMSE improved. This is what --resume reads -- resuming needs the
    most recent state, not the best one, or you lose whatever epochs
    happened after the last validation improvement.
  - best_model.pt: overwritten only when val RMSE improves. This is what
    final evaluation and Phase-3 read -- you want the best-generalizing
    weights there, not necessarily the most recent ones.

Resuming (--resume in main.py) restores model + optimizer + LR-scheduler
state and continues numbering epochs from where the checkpoint left off, up
to config.EPOCHS total -- it does not restart the count or the LR schedule.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from configs.config import Config
from src.data.dataset import move_batch_to_device
from src.models.physchem_cal import PhysChemCAL
from src.training.evaluate import evaluate
from src.training.losses import compute_losses
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.git_sync import push_results
from src.utils.results_logger import ResultsLogger

_LIVE_BATCH_CAP = 8  # how many SMILES from the last batch the dashboard gallery will render


def _write_live_batch(results_dir: str, epoch: int, smiles: List[str]) -> None:
    """Record the last training batch's SMILES so the dashboard can show 'what it's training on right
    now' instead of a fixed demo molecule list. Cheap: the SMILES are already in memory from the last
    batch (no extra forward pass here), and this writes once per epoch, not per batch. Rides along with
    the existing push_results() cadence since it lives inside results_dir -- no extra git calls added.
    Best-effort: a failure here must never interrupt training.
    """
    try:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "live_batch.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"epoch": epoch, "smiles": smiles[:_LIVE_BATCH_CAP]}, f)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[train] could not write live_batch.json (non-fatal): {e}")


def _try_restore(path: str, model: PhysChemCAL, optimizer: torch.optim.Optimizer,
                  scheduler: torch.optim.lr_scheduler.LRScheduler, device: torch.device):
    """Attempt to restore model/optimizer/scheduler state from one checkpoint file.

    Returns (start_epoch, best_val_rmse, best_epoch, loaded_epoch) on success,
    or None on ANY failure -- missing file, truncated/corrupt .pt (e.g. from a
    process killed mid-write before the atomic-rename fix in checkpoint.py),
    unexpected schema, etc. Never raises: the caller decides what to do next
    (try a fallback checkpoint, or start fresh) rather than this function
    crashing the whole training run over a bad checkpoint.
    """
    try:
        payload = load_checkpoint(path, model, optimizer, map_location=str(device))
        scheduler_state = payload.get("extra", {}).get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        start_epoch = payload["epoch"] + 1
        best_val_rmse = payload.get("best_val_rmse", float("inf"))
        best_epoch = payload.get("extra", {}).get("best_epoch", 0)
        return start_epoch, best_val_rmse, best_epoch, payload["epoch"]
    except Exception as e:
        print(f"[train] could not load checkpoint {path}: {e}")
        return None


def train(model: PhysChemCAL, data: Dict[str, object], config: Config, device: torch.device,
          results_logger: ResultsLogger, checkpoint_dir: str,
          resume_path: Optional[str] = None, push_every: int = 0) -> Dict[str, float]:
    """Train `model` up to config.EPOCHS epochs on data["train_loader"], validating each epoch on data["val_loader"].

    Args:
        model: a PhysChemCAL instance, already moved to `device`.
        data: dict returned by src.data.dataset.get_dataloaders() (train_loader,
            val_loader, test_loader, label_mean, label_std, ...).
        config: hyperparameters (learning rate, epochs, loss weights, ...).
        device: torch device to train on.
        results_logger: appends per-epoch metrics to results/YYYY-MM-DD.md.
        checkpoint_dir: where latest_model.pt / best_model.pt are saved.
        resume_path: if given, restore model/optimizer/scheduler/epoch from
            this checkpoint (typically checkpoint_dir/latest_model.pt from a
            previous, interrupted run) and continue from epoch+1 instead of
            epoch 1. None (default) starts fresh.
        push_every: if > 0, git-push the results/ folder every this many
            epochs (best-effort -- see src/utils/git_sync.py). 0 disables it.
    Returns:
        dict with "best_val_rmse" and "best_epoch".
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    label_mean, label_std = data["label_mean"], data["label_std"]
    train_loader = data["train_loader"]
    results_dir = results_logger.results_dir

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.LR_GAMMA)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_val_rmse = float("inf")
    best_epoch = 0

    if resume_path is not None:
        restored = _try_restore(resume_path, model, optimizer, scheduler, device)
        if restored is None:
            # A checkpoint saved right as a session died can be truncated/corrupt.
            # Try the sibling best_model.pt before giving up -- losing a few
            # epochs to a fallback beats crashing the whole run outright.
            fallback = os.path.join(os.path.dirname(resume_path) or ".", "best_model.pt")
            if os.path.abspath(fallback) != os.path.abspath(resume_path) and os.path.exists(fallback):
                print(f"[train] {resume_path} failed to load -- trying fallback {fallback}")
                restored = _try_restore(fallback, model, optimizer, scheduler, device)

        if restored is not None:
            start_epoch, best_val_rmse, best_epoch, loaded_epoch = restored
            results_logger.log_note(
                f"Resumed from checkpoint (epoch {loaded_epoch}) -- continuing at epoch {start_epoch}."
            )
            print(f"[train] resumed: epoch {loaded_epoch} -> continuing at epoch {start_epoch}")
            if start_epoch > config.EPOCHS:
                print(f"[train] checkpoint epoch {loaded_epoch} already >= --epochs {config.EPOCHS}; nothing to do.")
                return {"best_val_rmse": best_val_rmse, "best_epoch": best_epoch}
        else:
            msg = (f"Could not load `{resume_path}` or a fallback -- starting fresh from epoch 1 "
                   f"instead of crashing the run.")
            results_logger.log_note(f"WARNING: {msg}")
            print(f"[train] WARNING: {msg}")

    for epoch in range(start_epoch, config.EPOCHS + 1):
        model.train()
        epoch_losses = defaultdict(list)
        optimizer.zero_grad()
        accumulated = 0
        last_batch_smiles: List[str] = []

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            last_batch_smiles = batch["smiles"]

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
        _write_live_batch(results_dir, epoch, last_batch_smiles)
        print(f"[train] epoch {epoch}/{config.EPOCHS} "
              f"c={avg_losses['c_loss']:.4f} o={avg_losses['o_loss']:.4f} "
              f"co={avg_losses['co_loss']:.4f} conf={avg_losses['conf_loss']:.4f} "
              f"frag={avg_losses['frag_loss']:.4f} "
              f"total={avg_losses['total_loss']:.4f} val_rmse={val_metrics['rmse']:.4f}")

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            save_checkpoint(
                os.path.join(checkpoint_dir, "best_model.pt"), model, optimizer, epoch,
                label_mean, label_std, best_val_rmse,
                extra={"scheduler_state_dict": scheduler.state_dict(), "best_epoch": best_epoch},
            )

        # latest_model.pt: every epoch, regardless of improvement -- this is what --resume reads.
        # Carries best_epoch/best_val_rmse too (not just this epoch's own state) so a *second*
        # resume from latest_model.pt doesn't lose track of which earlier epoch was actually best.
        save_checkpoint(
            os.path.join(checkpoint_dir, "latest_model.pt"), model, optimizer, epoch,
            label_mean, label_std, best_val_rmse,
            extra={"scheduler_state_dict": scheduler.state_dict(), "best_epoch": best_epoch},
        )

        if push_every > 0 and epoch % push_every == 0:
            push_results(results_dir, epoch=epoch)

    results_logger.log_best(best_epoch, best_val_rmse)
    if push_every > 0:
        push_results(results_dir, epoch=config.EPOCHS)
    return {"best_val_rmse": best_val_rmse, "best_epoch": best_epoch}
