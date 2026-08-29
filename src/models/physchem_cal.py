"""
src/models/physchem_cal.py

Top-level model: wraps PhysChemEncoder (Stage 1 only) + CALHead into the
single module main.py / training / inference actually construct and call.
Mirrors the report's "PhyschemCAL for DrugOOD Wrapper" (Section 5.4.3): the
train/eval mode switch is handled implicitly by nn.Module's self.training
flag (which CALHead reads to decide whether to compute co_pred), so callers
never need to pass a separate flag -- just model.train() / model.eval() as usual.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.data.mask_matrices import MaskMatrices
from src.models.cal_head import CALHead
from src.models.physchem_encoder import PhysChemEncoder


class PhysChemCAL(nn.Module):
    """PhysChem Stage-1 encoder feeding directly into the CAL head -- the complete model this project trains and evaluates."""

    def __init__(self, atom_ftr_dim: int, bond_ftr_dim: int, hv_dim: int, he_dim: int,
                 pq_dim: int, n_layer: int, n_iteration: int, tau: float, rela_chunk: int,
                 cal_dropout: float):
        super().__init__()
        self.encoder = PhysChemEncoder(
            atom_ftr_dim=atom_ftr_dim, bond_ftr_dim=bond_ftr_dim, hv_dim=hv_dim, he_dim=he_dim,
            pq_dim=pq_dim, n_layer=n_layer, n_iteration=n_iteration, tau=tau, rela_chunk=rela_chunk,
        )
        self.cal_head = CALHead(hv_dim=hv_dim, dropout=cal_dropout)

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, pos: torch.Tensor,
                masses: torch.Tensor, mask_matrices: MaskMatrices
                ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            atom_ftr [A,34], bond_ftr [E,10], pos [A,3], masses [A,1]: from one collated batch (see src/data/dataset.py).
            mask_matrices: batch connectivity structure for that same batch.
        Returns dict with:
            c_pred, o_pred, co_pred (co_pred is None in eval mode), att_c, att_o -- see CALHead.forward.
            conformations: list of PhysNet position snapshots, for conf_loss.
            hv: final atom embeddings [A, hv_dim] (exposed for Phase-3 counterfactual guidance, which reads att_o but may also want hv directly for debugging).
        """
        hv, conformations = self.encoder(atom_ftr, bond_ftr, pos, masses, mask_matrices)
        outputs = self.cal_head(hv, mask_matrices)
        outputs["conformations"] = conformations
        outputs["hv"] = hv
        return outputs
