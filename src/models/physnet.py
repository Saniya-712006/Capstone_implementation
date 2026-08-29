"""
src/models/physnet.py

PhysNet: the differentiable Newtonian-mechanics half of PhysChem Stage 1
(hld_physchem.md Step 2, report Section 3.4). At each of N_ITERATION steps:
  1. compute a learned bond ("spring") force along every bond direction,
  2. compute a learned long-range repulsion/attraction force between every
     pair of atoms in the same molecule (chunked over atom rows to avoid an
     O(A^2) memory blowout -- this is the exact fix documented in Table 3 of
     the report),
  3. apply the Newtonian update  p += F*tau ; q += (p/m)*tau,
  4. snapshot q into the running conformations list (used later by
     conf_loss).

All force parameters are learned MLPs, not hard-coded physics -- the model
discovers what forces make chemical sense from training signal alone. The
bond-force MLP's final layer is zero-initialised so the simulation starts
with literally no forces and learns them from scratch, per the original
paper's design choice (this prevents a large random perturbation at the
very start of training).
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from src.data.mask_matrices import MaskMatrices


class BondForceMLP(nn.Module):
    """Learned scalar force magnitude for a directed bond, from its two endpoint atoms' hidden states plus the bond's own hidden state."""

    def __init__(self, hv_dim: int, he_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hv_dim + he_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init the final layer: no forces at the start of training.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hv_begin: torch.Tensor, hv_end: torch.Tensor, he: torch.Tensor) -> torch.Tensor:
        """Returns [E, 1] scalar force magnitude per directed edge."""
        return self.net(torch.cat([hv_begin, hv_end, he], dim=-1))


class RelationalForceMLP(nn.Module):
    """Learned scalar long-range force magnitude between an atom pair, from both atoms' hidden states plus their current distance."""

    def __init__(self, hv_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hv_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hv_a: torch.Tensor, hv_b: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hv_a: [..., hv_dim] hidden states of the "row" atoms in this chunk.
            hv_b: [..., hv_dim] hidden states of the "column" (all) atoms, broadcast-compatible with hv_a.
            dist: [..., 1] pairwise distances.
        Returns:
            [..., 1] scalar force magnitude per pair.
        """
        return self.net(torch.cat([hv_a, hv_b, dist], dim=-1))


class PhysNet(nn.Module):
    """Runs N_ITERATION Newtonian simulation steps, updating (p, q) each step and returning the final state plus every intermediate q as a conformation snapshot."""

    def __init__(self, hv_dim: int, he_dim: int, n_iteration: int, tau: float, rela_chunk: int):
        super().__init__()
        self.n_iteration = n_iteration
        self.tau = tau
        self.rela_chunk = rela_chunk
        self.bond_force = BondForceMLP(hv_dim, he_dim)
        self.rela_force = RelationalForceMLP(hv_dim)

    def forward(self, hv: torch.Tensor, he: torch.Tensor, p: torch.Tensor, q: torch.Tensor,
                masses: torch.Tensor, mask_matrices: MaskMatrices
                ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            hv: [A, hv_dim] atom hidden states (read-only here; ChemNet updates these).
            he: [E, he_dim] bond hidden states (read-only here).
            p: [A, 3] current momentum.
            q: [A, 3] current position.
            masses: [A, 1] atomic mass / 100.
            mask_matrices: batch connectivity (begin_idx/end_idx for bond force, batch for same-molecule masking).
        Returns:
            p, q after n_iteration Newton steps, and a list of n_iteration
            position snapshots (one per step, each [A, 3]) for the
            conformational loss.
        """
        snapshots = []
        for _ in range(self.n_iteration):
            f_bond = self._bond_force(hv, he, q, mask_matrices)
            f_rela = self._relational_force(hv, q, masses, mask_matrices)
            f_total = f_bond + f_rela

            p = p + f_total * self.tau
            q = q + (p / masses.clamp(min=1e-6)) * self.tau
            snapshots.append(q)
        return p, q, snapshots

    def _bond_force(self, hv: torch.Tensor, he: torch.Tensor, q: torch.Tensor,
                     mask_matrices: MaskMatrices) -> torch.Tensor:
        """Sum of learned spring-like forces from every bond incident on each atom (see hld_physchem.md 'Bond Forces')."""
        begin_idx, end_idx = mask_matrices.begin_idx, mask_matrices.end_idx
        direction = q[end_idx] - q[begin_idx]                          # [E, 3]
        dist = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)    # [E, 1]
        unit_dir = direction / dist

        magnitude = self.bond_force(hv[begin_idx], hv[end_idx], he)    # [E, 1]
        contribution = magnitude * unit_dir                            # [E, 3]

        A = hv.shape[0]
        f_bond = torch.zeros((A, 3), device=hv.device, dtype=hv.dtype)
        f_bond.index_add_(0, begin_idx, contribution)
        return f_bond

    def _relational_force(self, hv: torch.Tensor, q: torch.Tensor, masses: torch.Tensor,
                           mask_matrices: MaskMatrices) -> torch.Tensor:
        """Long-range attraction/repulsion between every same-molecule atom pair, processed in row-chunks of `rela_chunk` atoms to keep peak memory at O(chunk x A) instead of O(A^2) (Table 3 of the report)."""
        A = hv.shape[0]
        batch = mask_matrices.batch  # [A]
        f_rela = torch.zeros((A, 3), device=hv.device, dtype=hv.dtype)

        for start in range(0, A, self.rela_chunk):
            end = min(start + self.rela_chunk, A)
            chunk_size = end - start

            q_chunk = q[start:end]                      # [c, 3]
            hv_chunk = hv[start:end]                     # [c, hv_dim]
            batch_chunk = batch[start:end]                # [c]

            diff = q_chunk.unsqueeze(1) - q.unsqueeze(0)  # [c, A, 3]
            dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [c, A, 1]
            unit_dir = diff / dist

            hv_a = hv_chunk.unsqueeze(1).expand(-1, A, -1)  # [c, A, hv_dim]
            hv_b = hv.unsqueeze(0).expand(chunk_size, -1, -1)  # [c, A, hv_dim]
            magnitude = self.rela_force(hv_a, hv_b, dist)  # [c, A, 1]

            same_mol = (batch_chunk.unsqueeze(1) == batch.unsqueeze(0)).unsqueeze(-1).float()  # [c, A, 1]
            not_self = torch.ones((chunk_size, A, 1), device=hv.device)
            not_self[torch.arange(chunk_size), start + torch.arange(chunk_size)] = 0.0
            mask = same_mol * not_self

            mass_b = masses.view(1, A, 1)  # scale contribution by the partner atom's mass
            contribution = magnitude * unit_dir * mask * mass_b  # [c, A, 3]

            f_rela[start:end] = contribution.sum(dim=1)

        return f_rela
