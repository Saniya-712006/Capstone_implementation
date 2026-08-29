"""
src/models/physchem_encoder.py

Wires together Initializer -> [PhysNet -> ChemNet] x N_LAYER into the full
PhysChem Stage-1 encoder (Section 3.1 "Pipeline overview" of the report).

This is deliberately Stage 1 ONLY -- the FingerprintGenerator
(GlobalReadout + GRUUnion, "Stage 2" in interface_planning.md) is NOT
implemented here. As agreed and documented in interface_planning.md /
integration_final.md's "Key Design Decisions" table, CAL's own
attention-weighted pooling (src/models/cal_head.py) replaces it -- nothing in
the loss ever consumes a Stage-2 fingerprint, so building it would be dead
computation. The only thing this encoder hands off to CAL is the final
atom-level hv [A, hv_dim].
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from src.data.mask_matrices import MaskMatrices
from src.models.chemnet import ChemNet
from src.models.initializer import Initializer
from src.models.physnet import PhysNet


class PhysChemEncoder(nn.Module):
    """Stage-1-only PhysChem encoder: Initializer, then N_LAYER alternations of PhysNet (physics) and ChemNet (triplet-attention chemistry)."""

    def __init__(self, atom_ftr_dim: int, bond_ftr_dim: int, hv_dim: int, he_dim: int,
                 pq_dim: int, n_layer: int, n_iteration: int, tau: float, rela_chunk: int):
        super().__init__()
        self.n_layer = n_layer
        self.initializer = Initializer(atom_ftr_dim, bond_ftr_dim, hv_dim, he_dim, pq_dim)
        # Independent PhysNet/ChemNet weights per outer layer (not shared across
        # layers) -- matches "the two alternate for N_LAYER rounds" as distinct
        # learned steps, not a single recurrent cell applied twice.
        self.phys_layers = nn.ModuleList([
            PhysNet(hv_dim, he_dim, n_iteration, tau, rela_chunk) for _ in range(n_layer)
        ])
        self.chem_layers = nn.ModuleList([
            ChemNet(hv_dim) for _ in range(n_layer)
        ])

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, pos: torch.Tensor,
                masses: torch.Tensor, mask_matrices: MaskMatrices
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            atom_ftr: [A, 34], bond_ftr: [E, 10], pos: [A, 3], masses: [A, 1].
            mask_matrices: batch connectivity structure.
        Returns:
            hv_final [A, hv_dim]: physics+chemistry-informed atom embeddings, handed to the CAL head.
            conformations: list of length n_layer * n_iteration, each [A, 3],
                the q snapshots from every Newton step, used by conf_loss.
        """
        hv, he, p, q = self.initializer(atom_ftr, bond_ftr, pos, mask_matrices)

        conformations: List[torch.Tensor] = []
        for phys, chem in zip(self.phys_layers, self.chem_layers):
            p, q, snapshots = phys(hv, he, p, q, masses, mask_matrices)
            conformations.extend(snapshots)
            hv, he = chem(hv, he, q, mask_matrices)

        return hv, conformations
