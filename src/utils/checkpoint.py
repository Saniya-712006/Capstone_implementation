"""
src/utils/checkpoint.py

Save/load helpers for model weights plus the label normalisation statistics
(train-set mean/std) they must always travel with -- a checkpoint without its
normalisation stats can't correctly denormalise predictions back to pEC50
units, so the two are always bundled into a single .pt file rather than
tracked separately.
"""

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                     epoch: int, label_mean: float, label_std: float,
                     best_val_rmse: float, extra: Optional[Dict[str, Any]] = None) -> None:
    """Write model/optimizer state plus label-normalisation stats to `path`.

    Args:
        path: destination file, e.g. checkpoints/physchemcal_epoch15.pt.
        model: the PhysChemCAL model whose state_dict is saved.
        optimizer: current optimizer (state saved for resuming training); pass
            None to skip (e.g. when saving a "best model for inference only" copy).
        epoch: epoch number this checkpoint was taken at.
        label_mean, label_std: training-set label normalisation stats, needed
            at inference time to denormalise predictions back to pEC50 units.
        best_val_rmse: best validation RMSE seen so far (for logging/resume).
        extra: any additional small picklable metadata to stash alongside.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "label_mean": label_mean,
        "label_std": label_std,
        "best_val_rmse": best_val_rmse,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model: torch.nn.Module,
                     optimizer: Optional[torch.optim.Optimizer] = None,
                     map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a checkpoint written by save_checkpoint() into `model` (and `optimizer` if given).

    Returns the raw payload dict so callers can also read epoch/label_mean/
    label_std/best_val_rmse/extra without a second disk read.
    """
    payload = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
