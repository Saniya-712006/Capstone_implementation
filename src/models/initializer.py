"""
src/models/initializer.py

PhysChem Stage-1, step 1 ("Initializer" in hld_physchem.md / Section 3.3 of
the report): turns raw atom/bond feature tensors into the (hv, he, p, q)
state the PhysNet+ChemNet loop starts from.

Three operations, in order:
  1. Linear+Tanh projection of atom/bond features into hidden space.
  2. A small residual GCN over the atom graph (using the MaskMatrices
     vertex_edge_w1/w2 masks) so each atom's seed state already knows about
     its 1-hop neighbourhood before physics/chemistry begins.
  3. A 2-layer LSTM reading each molecule's atoms as a sequence (in RDKit
     atom order) to produce the initial momentum p and to seed q -- q is
     ultimately overwritten by the molecule's real 3D coordinates (`pos`),
     since PhysNet should start from a physically real geometry, not a
     learned guess; p (momentum / "push direction") has no ground truth, so
     it stays learned.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data.mask_matrices import MaskMatrices


class ResidualGCNLayer(nn.Module):
    """One residual graph-conv step: aggregate neighbour features through the bond graph, then concatenate (not replace) with the input, so information from earlier layers is never overwritten."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, mask_matrices: MaskMatrices) -> torch.Tensor:
        """Aggregate each atom's bonded neighbours' features and apply a ReLU-activated linear transform.

        Args:
            h: [A, hidden_dim] current atom hidden states.
            mask_matrices: batch connectivity (uses vertex_edge_w1/w2 to reach 1-hop neighbours through bonds).
        Returns:
            [A, hidden_dim] aggregated + transformed neighbour features (to be concatenated with `h` by the caller).
        """
        # neighbour-of-atom-a via bonds: w1[a,e]=1 if a begins edge e, w2[a,e]=1 if a' ends edge e.
        # (w1 @ w2^T)[a, a'] counts directed edges a->a', i.e. an adjacency matrix.
        adjacency = mask_matrices.vertex_edge_w1 @ mask_matrices.vertex_edge_w2.T  # [A, A]
        degree = adjacency.sum(dim=1, keepdim=True).clamp(min=1.0)
        agg = (adjacency @ h) / degree
        return torch.relu(self.linear(agg))


class Initializer(nn.Module):
    """Projects raw atom/bond features to hidden space, refines atoms with a residual GCN, and produces the initial momentum p (learned) and position q (= real 3D coordinates)."""

    def __init__(self, atom_ftr_dim: int, bond_ftr_dim: int, hv_dim: int, he_dim: int,
                 pq_dim: int = 3, n_gcn_layers: int = 2):
        super().__init__()
        self.hv_dim = hv_dim
        self.atom_proj = nn.Linear(atom_ftr_dim, hv_dim)
        self.bond_proj = nn.Linear(bond_ftr_dim, he_dim)

        self.gcn_layers = nn.ModuleList([ResidualGCNLayer(hv_dim) for _ in range(n_gcn_layers)])
        # residual concat: [input, layer1_out, ..., layerK_out] -> back down to hv_dim
        concat_dim = hv_dim * (n_gcn_layers + 1)
        self.gcn_project = nn.Linear(concat_dim, hv_dim)

        self.lstm = nn.LSTM(input_size=hv_dim, hidden_size=pq_dim, num_layers=2,
                             batch_first=True, bidirectional=False)
        self.p_out = nn.Linear(pq_dim, pq_dim)

    def forward(self, atom_ftr: torch.Tensor, bond_ftr: torch.Tensor, pos: torch.Tensor,
                mask_matrices: MaskMatrices) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            atom_ftr: [A, 34] raw atom features.
            bond_ftr: [E, 10] raw bond features.
            pos: [A, 3] real 3D coordinates from the cached conformer.
            mask_matrices: batch connectivity structure.
        Returns:
            hv [A, hv_dim], he [E, he_dim], p [A, 3] (learned initial momentum),
            q [A, 3] (= pos, the real starting geometry PhysNet will simulate forward from).
        """
        hv = torch.tanh(self.atom_proj(atom_ftr))
        he = torch.tanh(self.bond_proj(bond_ftr))

        layer_outputs = [hv]
        h = hv
        for layer in self.gcn_layers:
            h = layer(h, mask_matrices)
            layer_outputs.append(h)
        hv = torch.relu(self.gcn_project(torch.cat(layer_outputs, dim=-1)))

        p = self._compute_momentum(hv, mask_matrices)
        q = pos
        return hv, he, p, q

    def _compute_momentum(self, hv: torch.Tensor, mask_matrices: MaskMatrices) -> torch.Tensor:
        """Run the 2-layer LSTM per-molecule over each molecule's atoms (in their existing RDKit order) to get the initial momentum p.

        Molecules in the flat batch have variable atom counts, so this pads
        each molecule's atom sequence to the batch's max length, runs a
        packed LSTM, then un-pads and concatenates back into the same flat
        [A, pq_dim] atom ordering used everywhere else.
        """
        device = hv.device
        counts = mask_matrices.mol_vertex_w.sum(dim=1).long()  # [M] atoms per molecule
        M = counts.shape[0]
        max_len = counts.max().item()

        padded = torch.zeros((M, max_len, self.hv_dim), device=device, dtype=hv.dtype)
        offset = 0
        for m in range(M):
            n = counts[m].item()
            padded[m, :n] = hv[offset:offset + n]
            offset += n

        packed = pack_padded_sequence(padded, counts.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=max_len)

        flat = []
        for m in range(M):
            n = counts[m].item()
            flat.append(unpacked[m, :n])
        p_seq = torch.cat(flat, dim=0)  # [A, pq_dim]
        return self.p_out(p_seq)
