"""
src/models/cal_head.py

CAL (Causal Attention Learning, Sui et al. KDD-2022) head -- Section 3.7 /
Figure 3 of the report. Takes the atom-level embeddings hv [A, hv_dim] from
PhysChemEncoder and splits each atom into a causal ("object") share and a
scaffold/context share via one shared softmax attention, then produces three
molecule-level predictions used by the three training losses.

Architectural note (documented here deliberately, not hidden): the design
docs (interface_planning.md / integration_final.md) sketch a richer version
of this head with per-branch edge attention and a dedicated GCN layer before
pooling. The report's own Section 3.7.1-3.7.3 + Figure 3 -- i.e. what was
actually built to produce the results in Chapter 6 -- describes a simpler
head: a linear+ReLU projection per branch, attention-weighted MEAN pooling
straight to molecule level (via mol_vertex_w), LayerNorm, then a 2-layer MLP
decoder. This module follows the report's simpler, already-validated version
to keep the head lightweight (matching the "reduce effort" direction agreed
on for Stage 2 of PhysChem) -- flag this choice if the richer edge-attention
variant is wanted later.

Three branches:
  - context (c_pred): trained to be non-predictive (pushed toward the
    z-normalised label mean, i.e. ~0) -- forces the attention to keep causal
    signal OUT of att_c.
  - object (o_pred): the primary prediction, trained with the real task loss.
    This is the only branch used at inference.
  - combined (co_pred, TRAINING ONLY): xo + a randomly-shuffled xc from
    elsewhere in the batch (xc detached, so no gradient reinforces the
    context branch through this path) -- the backdoor-adjustment
    intervention that forces xo to be scaffold-invariant.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.data.mask_matrices import MaskMatrices


def _mean_pool(atom_features: torch.Tensor, mask_matrices: MaskMatrices) -> torch.Tensor:
    """Attention-weighted atom features -> per-molecule mean, using the mol_vertex_w membership mask (Section 3.7.2: 'mean-pooled to molecule level using the mol_vertex_w mask')."""
    sums = mask_matrices.mol_vertex_w @ atom_features           # [M, hv_dim]
    counts = mask_matrices.mol_vertex_w.sum(dim=1, keepdim=True).clamp(min=1.0)  # [M, 1]
    return sums / counts


class BranchDecoder(nn.Module):
    """2-layer MLP decoder shared shape for all three CAL branches: LayerNorm -> Linear -> ReLU -> Dropout -> LayerNorm -> Linear(->1). LayerNorm (not BatchNorm) is used throughout specifically because it is batch-size-agnostic -- BatchNorm1d crashes on a trailing batch of size 1, which is exactly the bug the report documents fixing."""

    def __init__(self, hv_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hv_dim),
            nn.Linear(hv_dim, hv_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hv_dim),
            nn.Linear(hv_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [M, hv_dim] -> [M] scalar prediction (z-normalised label scale)."""
        return self.net(x).squeeze(-1)


class CALHead(nn.Module):
    """Causal-attention split + three-branch prediction over per-molecule pooled embeddings."""

    def __init__(self, hv_dim: int, dropout: float):
        super().__init__()
        self.node_att_mlp = nn.Linear(hv_dim, 2)  # shared attention: logits for [context, object] per atom

        self.context_proj = nn.Linear(hv_dim, hv_dim)
        self.object_proj = nn.Linear(hv_dim, hv_dim)

        self.context_norm = nn.LayerNorm(hv_dim)
        self.object_norm = nn.LayerNorm(hv_dim)

        self.context_head = BranchDecoder(hv_dim, dropout)
        self.object_head = BranchDecoder(hv_dim, dropout)
        self.combined_head = BranchDecoder(hv_dim, dropout)

    def forward(self, hv: torch.Tensor, mask_matrices: MaskMatrices) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            hv: [A, hv_dim] atom embeddings from PhysChemEncoder.
            mask_matrices: batch connectivity (mol_vertex_w used for pooling).
        Returns:
            dict with:
              c_pred [M]: context-branch prediction (always computed -- c_loss needs it every step).
              o_pred [M]: object/causal-branch prediction -- THE prediction used at inference.
              co_pred [M] or None: combined-branch prediction, computed only
                  when self.training is True (matches report Section 3.9:
                  eval mode returns only o_pred-relevant outputs).
              att_c [A, 1], att_o [A, 1]: per-atom attention weights (sum to
                  1 per atom); att_o is what Phase-3 counterfactual
                  generation reads to find "causal atoms".
        """
        node_att = torch.softmax(self.node_att_mlp(hv), dim=-1)  # [A, 2]
        att_c = node_att[:, 0:1]
        att_o = node_att[:, 1:2]

        xc_atoms = att_c * torch.relu(self.context_proj(hv))  # [A, hv_dim]
        xo_atoms = att_o * torch.relu(self.object_proj(hv))   # [A, hv_dim]

        xc_mol = self.context_norm(_mean_pool(xc_atoms, mask_matrices))  # [M, hv_dim]
        xo_mol = self.object_norm(_mean_pool(xo_atoms, mask_matrices))   # [M, hv_dim]

        c_pred = self.context_head(xc_mol)
        o_pred = self.object_head(xo_mol)

        co_pred = None
        if self.training:
            m = xc_mol.shape[0]
            perm = torch.randperm(m, device=hv.device)
            xc_shuffled = xc_mol[perm].detach()  # backdoor adjustment: intervened context, no gradient into context branch
            co_pred = self.combined_head(xo_mol + xc_shuffled)

        return {
            "c_pred": c_pred,
            "o_pred": o_pred,
            "co_pred": co_pred,
            "att_c": att_c,
            "att_o": att_o,
        }
