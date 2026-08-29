"""
src/training/evaluate.py

Runs the model in eval mode over a DataLoader and reports RMSE in the
original pEC50 scale (predictions are denormalised with the training set's
label_mean/label_std before the error is computed, per Section 3.9 of the
report). Used both for the periodic validation-set check during training and
for the final held-out OOD test evaluation.
"""

import math
from typing import Dict

import torch

from src.data.dataset import move_batch_to_device
from src.models.physchem_cal import PhysChemCAL


@torch.no_grad()
def evaluate(model: PhysChemCAL, loader, device, label_mean: float, label_std: float) -> Dict[str, float]:
    """Compute RMSE of the object/causal branch prediction (o_pred) against raw pEC50 labels over an entire DataLoader.

    Args:
        model: PhysChemCAL model; this function puts it in eval() mode and
            restores it to train() mode afterwards if it was training before.
        loader: a DataLoader built by src/data/dataset.py's get_dataloaders().
        device: torch device to run on.
        label_mean, label_std: training-set label normalisation stats, used
            to denormalise o_pred back to the real pEC50 scale before
            computing the error.
    Returns:
        dict with "rmse" and "n" (number of molecules evaluated).
    """
    was_training = model.training
    model.eval()

    total_sq_err = 0.0
    n = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            batch["atom_ftr"], batch["bond_ftr"], batch["pos"], batch["masses"], batch["mask_matrices"]
        )
        pred_denorm = outputs["o_pred"] * label_std + label_mean
        sq_err = (pred_denorm - batch["raw_labels"]) ** 2
        total_sq_err += sq_err.sum().item()
        n += batch["raw_labels"].numel()

    if was_training:
        model.train()

    rmse = math.sqrt(total_sq_err / n) if n > 0 else float("nan")
    return {"rmse": rmse, "n": n}
