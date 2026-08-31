"""
src/training/losses.py

The four training loss terms from Table 2 of the report, plus one added
later (frag_loss), combined into
total_loss = LAMBDA_C*c_loss + LAMBDA_O*o_loss + LAMBDA_CO*co_loss
           + LAMBDA_CONF*conf_loss + LAMBDA_FRAG*frag_loss.

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
  frag_loss: mean within-BRICS-fragment variance of att_o -- a soft nudge
             (not a hard constraint) encouraging CAL's causal/context split
             to respect real chemical fragment boundaries instead of
             cutting an acetyl group, ring, etc. in half. Added in response
             to reviewer feedback questioning whether CAL's split is
             chemically valid; see fragment_purity_loss() below.
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


def fragment_purity_loss(att_o: torch.Tensor, fragment_id: torch.Tensor) -> torch.Tensor:
    """Mean, across every BRICS fragment in the batch, of Var(att_o) among that fragment's atoms.

    Low when CAL's causal attention agrees within each chemical fragment
    (an acetyl group's atoms all lean causal or all lean scaffold together);
    high when it's splitting fragments into "half causal, half scaffold".
    This is a soft penalty, not a constraint -- minimising it pulls the
    model toward fragment-respecting splits, but a fragment can still be
    split if doing so genuinely helps o_loss/co_loss enough to be worth the
    penalty (see the option-2-vs-option-3 discussion this was built from:
    a hard per-fragment vote would remove the model's ability to express
    that kind of exception entirely).

    Implemented as a single vectorised scatter (index_add_) over all
    fragments in the batch, not a Python loop -- same pattern as the
    index_add_-based force aggregation in src/models/physnet.py.

    Args:
        att_o: [A, 1] or [A] per-atom causal attention weight.
        fragment_id: [A] long tensor, globally-unique fragment id per atom
            (already offset across molecules by batch_graphs()).
    Returns:
        scalar tensor. A fragment of size 1 contributes exactly 0 (a single
        value has zero variance from itself), which is the correct/expected
        behaviour, not a special case that needs guarding against.
    """
    att_o = att_o.view(-1)
    n_fragments = int(fragment_id.max().item()) + 1

    counts = att_o.new_zeros(n_fragments)
    counts.index_add_(0, fragment_id, torch.ones_like(att_o))
    sums = att_o.new_zeros(n_fragments)
    sums.index_add_(0, fragment_id, att_o)
    sums_sq = att_o.new_zeros(n_fragments)
    sums_sq.index_add_(0, fragment_id, att_o * att_o)

    means = sums / counts
    mean_of_squares = sums_sq / counts
    # clamp(min=0): floating-point roundoff can otherwise push this a hair below 0.
    variances = (mean_of_squares - means * means).clamp(min=0)
    return variances.mean()


def compute_losses(outputs: Dict[str, Optional[torch.Tensor]], batch: Dict[str, object],
                    config: Config) -> Dict[str, torch.Tensor]:
    """Compute all four loss terms plus the weighted total for one batch.

    Args:
        outputs: dict returned by PhysChemCAL.forward() (c_pred, o_pred,
            co_pred, att_o, conformations, ...).
        batch: dict returned by the collate_fn in src/data/dataset.py
            (labels, pos, adj3, fragment_id, ...).
        config: hyperparameters, for the LAMBDA_* loss weights.
    Returns:
        dict with keys c_loss, o_loss, co_loss, conf_loss, frag_loss,
        total_loss (all scalar tensors).
    """
    labels = batch["labels"]
    c_pred, o_pred, co_pred = outputs["c_pred"], outputs["o_pred"], outputs["co_pred"]

    c_loss = c_pred.var(unbiased=False)
    o_loss = F.mse_loss(o_pred, labels)
    co_loss = F.mse_loss(co_pred, labels) if co_pred is not None else o_pred.new_zeros(())
    conf_loss = conformational_loss(outputs["conformations"], batch["pos"], batch["adj3"])
    frag_loss = fragment_purity_loss(outputs["att_o"], batch["fragment_id"])

    total_loss = (
        config.LAMBDA_C * c_loss
        + config.LAMBDA_O * o_loss
        + config.LAMBDA_CO * co_loss
        + config.LAMBDA_CONF * conf_loss
        + config.LAMBDA_FRAG * frag_loss
    )

    return {
        "c_loss": c_loss,
        "o_loss": o_loss,
        "co_loss": co_loss,
        "conf_loss": conf_loss,
        "frag_loss": frag_loss,
        "total_loss": total_loss,
    }
