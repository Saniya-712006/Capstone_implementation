"""
src/models/chemnet.py

ChemNet: the triplet-attention chemical message-passing half of PhysChem
Stage 1 (hld_physchem.md "Triplet Message Passing", report Section 3.5).

Unlike a standard pairwise GNN, the message that updates atom A also encodes
the *angle* between every pair of A's outgoing bonds (using the current 3D
positions q from PhysNet) -- this is what lets the model distinguish, e.g., a
120-degree aromatic carbon from a 109.5-degree saturated one, which look
identical to a pairwise-only GNN.

Implementation follows the exact memory fix documented in Table 3 / Section
5.3.3 of the report: instead of a global [2E, 2E] edge-edge tensor, edges are
grouped by their shared source ("hub") atom and processed one small group at
a time (each group's pairwise tensor is at most [k, k, feature_dim] where k
is that atom's degree, typically <= 4) -- mathematically identical to the
naive [2E, 2E] version because triplet interactions are exactly zero for any
pair of edges that don't share a source atom.

Simplification made explicit here: only atom hidden states (hv) are updated
by this module; bond hidden states (he) are passed through unchanged between
ChemNet calls. he already carries local bond chemistry from the Initializer
and is what PhysNet's bond-force MLP reads directly -- the design docs give a
precise formula for the atom-level triplet update but not for an he update,
so rather than invent one, he is left untouched here.
"""

from typing import Tuple

import torch
import torch.nn as nn

from src.data.mask_matrices import MaskMatrices


class ChemNet(nn.Module):
    """One round of triplet-attention message passing followed by a GRU fusion into the atom hidden states."""

    def __init__(self, hv_dim: int, dist_dim: int = 8, angle_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.hv_dim = hv_dim
        self.dist_proj = nn.Linear(1, dist_dim)
        self.angle_proj = nn.Linear(1, angle_dim)
        self.bond_attn = nn.Linear(hv_dim, 1)  # scores an edge from its "B" endpoint's hidden state

        msg_in_dim = hv_dim + dist_dim + hv_dim + hv_dim + angle_dim  # [B | dist(A,B) | A | C | angle(B,A,C)]
        self.message_mlp = nn.Sequential(
            nn.Linear(msg_in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hv_dim),
        )
        self.gru = nn.GRUCell(hv_dim, hv_dim)

    def forward(self, hv: torch.Tensor, he: torch.Tensor, q: torch.Tensor,
                mask_matrices: MaskMatrices) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hv: [A, hv_dim] atom hidden states (updated by this call).
            he: [E, he_dim] bond hidden states (returned unchanged, see module docstring).
            q: [A, 3] current 3D positions from PhysNet, used to compute real bond angles.
            mask_matrices: batch connectivity (begin_idx/end_idx group edges by source atom).
        Returns:
            hv_new [A, hv_dim], he (unchanged).
        """
        A = hv.shape[0]
        begin_idx, end_idx = mask_matrices.begin_idx, mask_matrices.end_idx

        mv = torch.zeros((A, self.hv_dim), device=hv.device, dtype=hv.dtype)

        order = torch.argsort(begin_idx)
        begin_sorted = begin_idx[order]
        unique_atoms, counts = torch.unique_consecutive(begin_sorted, return_counts=True)

        offset = 0
        for atom_idx, k in zip(unique_atoms.tolist(), counts.tolist()):
            edge_slice = order[offset:offset + k]
            offset += k

            neighbour_idx = end_idx[edge_slice]              # [k] the atoms this hub is bonded to
            hv_neighbours = hv[neighbour_idx]                 # [k, hv_dim]  ("B"/"C" candidates)
            q_neighbours = q[neighbour_idx]                   # [k, 3]
            q_hub = q[atom_idx]                                # [3]

            direction = q_neighbours - q_hub.unsqueeze(0)      # [k, 3]
            dist = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [k, 1]
            unit_dir = direction / dist

            # angle(j, l) between bond hub->B_j and hub->B_l; j==l gives cos=1 (self-pair, degree-1 atoms naturally fall out of this with no special-casing).
            cos_angle = unit_dir @ unit_dir.T                  # [k, k]
            angle_ftr = self.angle_proj(cos_angle.unsqueeze(-1))          # [k, k, angle_dim]
            dist_ftr = self.dist_proj(dist).unsqueeze(1).expand(k, k, -1)  # [k, k, dist_dim] (dist to B_j, broadcast over l)

            hv_B = hv_neighbours.unsqueeze(1).expand(k, k, -1)   # [k, k, hv_dim]
            hv_C = hv_neighbours.unsqueeze(0).expand(k, k, -1)   # [k, k, hv_dim]
            hv_A = hv[atom_idx].view(1, 1, -1).expand(k, k, -1)  # [k, k, hv_dim]

            msg_in = torch.cat([hv_B, dist_ftr, hv_A, hv_C, angle_ftr], dim=-1)  # [k, k, msg_in_dim]
            pair_msg = self.message_mlp(msg_in)                              # [k, k, hv_dim]

            edge_scores = self.bond_attn(hv_neighbours).squeeze(-1)  # [k]
            edge_weights = torch.softmax(edge_scores, dim=0)          # [k]
            weighted = pair_msg * edge_weights.view(k, 1, 1)

            mv[atom_idx] = weighted.reshape(k * k, self.hv_dim).max(dim=0).values

        hv_new = self.gru(mv, hv)
        return hv_new, he
