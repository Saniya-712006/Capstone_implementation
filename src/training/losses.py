"""
src/training/losses.py

The four training loss terms from Table 2 of the report, combined into
total_loss = LAMBDA_C*c_loss + LAMBDA_O*o_loss + LAMBDA_CO*co_loss + LAMBDA_CONF*conf_loss.

  c_loss:    Var(c_pred) -> pushed toward 0. Since labels are z-normalised
             (train-set mean 0), a context branch with zero variance is
             already predicting the flat, uninformative "mean" answer -- no
             separate target value is needed, minimising variance IS the
             "predict nothing useful" objective.
  o_loss:    MSE(o_pred, labels) -- the real task loss, causal branch only.
  co_loss:   MSE(co_pred, labels) -- combined branch, only present in
             training mode (co_pred is None in eval mode; see CALHead).
  conf_loss: Huber loss between predicted inter-atomic distances (from each
             PhysNet snapshot) and the reference distances (from the real,
             ETKDG-derived `pos`), restricted to atom pairs exactly 3 bonds
             apart (the H_ADJ3 scheme -- these pairs constrain dihedral
             angles, the hardest-to-recover 3D degree of freedom).
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from configs.config import Config


def conformational_loss(conformations: List[torch.Tensor], pos: torch.Tensor,
                         adj3: torch.Tensor) -> torch.Tensor:
    """Average Huber loss, across every PhysNet snapshot, between predicted and reference inter-atomic distances on exactly-3-bonds-apart atom pairs.

    Args:
        conformations: list of [A, 3] position snapshots from PhysNet (one per Newton step).
        pos: [A, 3] reference (real, ETKDG-derived) 3D coordinates.
        adj3: [A, A] binary mask, 1 where two atoms are exactly 3 bonds apart.
    Returns:
        scalar tensor; 0.0 if the batch has no 3-hop atom pairs at all
        (possible for a batch of very small/smoke-test molecules).
    """
    pair_i, pair_j = adj3.nonzero(as_tuple=True)
    if pair_i.numel() == 0:
        return pos.new_zeros(())

    ref_dist = (pos[pair_i] - pos[pair_j]).norm(dim=-1)

    total = pos.new_zeros(())
    for q in conformations:
        pred_dist = (q[pair_i] - q[pair_j]).norm(dim=-1)
        total = total + F.huber_loss(pred_dist, ref_dist)
    return total / max(len(conformations), 1)


def compute_losses(outputs: Dict[str, Optional[torch.Tensor]], batch: Dict[str, object],
                    config: Config) -> Dict[str, torch.Tensor]:
    """Compute all four loss terms plus the weighted total for one batch.

    Args:
        outputs: dict returned by PhysChemCAL.forward() (c_pred, o_pred,
            co_pred, conformations, ...).
        batch: dict returned by the collate_fn in src/data/dataset.py
            (labels, pos, adj3, ...).
        config: hyperparameters, for the LAMBDA_* loss weights.
    Returns:
        dict with keys c_loss, o_loss, co_loss, conf_loss, total_loss (all scalar tensors).
    """
    labels = batch["labels"]
    c_pred, o_pred, co_pred = outputs["c_pred"], outputs["o_pred"], outputs["co_pred"]

    c_loss = c_pred.var(unbiased=False)
    o_loss = F.mse_loss(o_pred, labels)
    co_loss = F.mse_loss(co_pred, labels) if co_pred is not None else o_pred.new_zeros(())
    conf_loss = conformational_loss(outputs["conformations"], batch["pos"], batch["adj3"])

    total_loss = (
        config.LAMBDA_C * c_loss
        + config.LAMBDA_O * o_loss
        + config.LAMBDA_CO * co_loss
        + config.LAMBDA_CONF * conf_loss
    )

    return {
        "c_loss": c_loss,
        "o_loss": o_loss,
        "co_loss": co_loss,
        "conf_loss": conf_loss,
        "total_loss": total_loss,
    }
