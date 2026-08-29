"""
src/data/mask_matrices.py

Turns a single RDKit Mol (already 3D-embedded, as stored in the cached
DrugOOD .pkl files -- see README.md "Data setup") into the tensors the
PhysChem encoder and CAL head need, and batches several such molecules into
one flat, block-diagonal MaskMatrices structure.

This mirrors the featurisation scheme fixed in hld_physchem.md /
integration_final.md exactly:
  - atom_ftr: 34-dim one-hot (symbol 11 + degree 11 + H-count 5 + valence 6 + aromatic 1)
  - bond_ftr: 10-dim one-hot (bond type 4 + conjugated 1 + in-ring 1 + stereo 4)
  - every undirected bond is stored as two directed edges (begin->end and
    end->begin), matching the "E = total directed bonds" convention used
    throughout the design docs and by CAL's edge_index.
  - a batch of M molecules is packed into one flat atom array and one flat
    bond array; MaskMatrices carries the block-diagonal masks that let all
    downstream attention/message-passing be written as dense matmuls with no
    Python-level looping over molecules.

Molecules never get re-embedded here: the incoming Mol object's existing
conformer (produced once, offline, by ETKDG + MMFF94) is read as-is.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem

# ---------------------------------------------------------------------------
# Featurisation vocabularies (fixed -- changing these changes ATOM_FTR_DIM /
# BOND_FTR_DIM and therefore the model's first linear layer shapes).
# ---------------------------------------------------------------------------
ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "I", "other"]     # 11
DEGREES = list(range(11))                                                        # 11 (0-10)
H_COUNTS = list(range(5))                                                        # 5  (0-4)
VALENCES = list(range(6))                                                        # 6  (0-5)
# 11 + 11 + 5 + 6 + 1(aromatic) = 34

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]                                                                                  # 4
STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]                                                                                  # 4
# 4 + 1(conjugated) + 1(in ring) + 4 = 10


def _one_hot(value, choices: list) -> List[float]:
    """Standard one-hot: returns len(choices) floats, with the last slot acting as an 'other' bucket if `value` isn't found."""
    vec = [0.0] * len(choices)
    idx = choices.index(value) if value in choices else len(choices) - 1
    vec[idx] = 1.0
    return vec


def atom_features(atom: "Chem.Atom") -> List[float]:
    """Build the 34-dim feature vector for one heavy atom: symbol + degree + H-count + valence + aromaticity."""
    symbol = atom.GetSymbol()
    return (
        _one_hot(symbol, ATOM_SYMBOLS)
        + _one_hot(min(atom.GetDegree(), 10), DEGREES)
        + _one_hot(min(atom.GetTotalNumHs(), 4), H_COUNTS)
        + _one_hot(min(atom.GetValence(Chem.ValenceType.IMPLICIT), 5), VALENCES)
        + [float(atom.GetIsAromatic())]
    )


def bond_features(bond: "Chem.Bond") -> List[float]:
    """Build the 10-dim feature vector for one bond: bond type + conjugation + ring membership + stereo."""
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + _one_hot(bond.GetStereo(), STEREO_TYPES)
    )


@dataclass
class MolGraph:
    """Per-molecule tensors produced by mol_to_graph(), before batching."""

    atom_ftr: torch.Tensor    # [A_i, 34]
    bond_ftr: torch.Tensor    # [E_i, 10]
    pos: torch.Tensor         # [A_i, 3]  heavy-atom 3D coordinates from the cached conformer
    masses: torch.Tensor      # [A_i, 1]  atomic mass / 100
    begin_idx: torch.Tensor   # [E_i]     local (0-based, this molecule only) begin-atom index per directed edge
    end_idx: torch.Tensor     # [E_i]     local end-atom index per directed edge
    adj3: torch.Tensor        # [A_i, A_i] binary mask: 1 where two atoms are exactly 3 bonds apart
    smiles: str
    label: float


def _three_hop_mask(adj: np.ndarray) -> np.ndarray:
    """Build the H_ADJ3 mask used by the conformational loss: 1 where two atoms are exactly 3 bonds apart, 0 elsewhere (including the diagonal and 1/2-hop pairs)."""
    n = adj.shape[0]
    adj1 = (adj > 0).astype(np.float32)
    adj2 = (adj1 @ adj1 > 0).astype(np.float32)
    adj3 = (adj2 @ adj1 > 0).astype(np.float32)
    eye = np.eye(n, dtype=np.float32)
    only3 = adj3 * (1 - adj2) * (1 - adj1) * (1 - eye)
    return only3


def mol_to_graph(mol: "Chem.Mol", label: float, smiles: Optional[str] = None) -> Optional[MolGraph]:
    """Convert one already-3D-embedded RDKit Mol (as read from the cached .pkl) into a MolGraph.

    Strips explicit hydrogens from the graph topology (their count is already
    encoded in each heavy atom's feature vector) while keeping the heavy-atom
    3D coordinates from the existing conformer -- no re-embedding happens
    here. Returns None for molecules that can't be used (RDKit parse failure,
    zero heavy atoms, or zero bonds), so the batching step can silently skip them.
    """
    if mol is None:
        return None
    try:
        mol_heavy = Chem.RemoveHs(mol)
    except Exception:
        return None

    n_atoms = mol_heavy.GetNumAtoms()
    if n_atoms == 0 or mol_heavy.GetNumBonds() == 0:
        return None
    if mol_heavy.GetNumConformers() == 0:
        return None

    conf = mol_heavy.GetConformer()
    pos = torch.tensor(
        [list(conf.GetAtomPosition(i)) for i in range(n_atoms)], dtype=torch.float32
    )

    atom_ftrs = [atom_features(a) for a in mol_heavy.GetAtoms()]
    atom_ftr = torch.tensor(atom_ftrs, dtype=torch.float32)

    masses = torch.tensor(
        [[a.GetMass() / 100.0] for a in mol_heavy.GetAtoms()], dtype=torch.float32
    )

    begin_idx, end_idx, bond_ftrs = [], [], []
    adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    for bond in mol_heavy.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        begin_idx += [u, v]
        end_idx += [v, u]
        bond_ftrs += [bf, bf]
        adj[u, v] = 1.0
        adj[v, u] = 1.0

    if len(begin_idx) == 0:
        return None

    bond_ftr = torch.tensor(bond_ftrs, dtype=torch.float32)
    begin_idx_t = torch.tensor(begin_idx, dtype=torch.long)
    end_idx_t = torch.tensor(end_idx, dtype=torch.long)
    adj3 = torch.tensor(_three_hop_mask(adj), dtype=torch.float32)

    return MolGraph(
        atom_ftr=atom_ftr,
        bond_ftr=bond_ftr,
        pos=pos,
        masses=masses,
        begin_idx=begin_idx_t,
        end_idx=end_idx_t,
        adj3=adj3,
        smiles=smiles or "",
        label=float(label),
    )


@dataclass
class MaskMatrices:
    """Batched, block-diagonal connectivity structure for M molecules flattened into A total atoms / E total directed bonds.

    Carries both representations used across the codebase: the dense
    attention-bias matrices PhysChem's own modules expect (vertex_edge_*,
    mol_vertex_*), and the plain index tensors (begin_idx/end_idx, batch)
    that gather/scatter-based ops (Newton forces, CAL's pooling) use -- both
    are built from the same flat atom ordering in batch_graphs() below, so
    they are guaranteed consistent by construction.
    """

    vertex_edge_w1: torch.Tensor  # [A, E]  1 where atom a is the begin-atom of edge e
    vertex_edge_w2: torch.Tensor  # [A, E]  1 where atom a is the end-atom of edge e
    vertex_edge_b1: torch.Tensor  # [A, E]  0 / -1e9 attention bias for w1
    vertex_edge_b2: torch.Tensor  # [A, E]  0 / -1e9 attention bias for w2
    mol_vertex_w: torch.Tensor    # [M, A]  1 where atom a belongs to molecule m
    mol_vertex_b: torch.Tensor    # [M, A]  0 / -1e9 attention bias for mol-level readout
    begin_idx: torch.Tensor       # [E]     global begin-atom index per directed edge
    end_idx: torch.Tensor         # [E]     global end-atom index per directed edge
    batch: torch.Tensor           # [A]     molecule id (0..M-1) each atom belongs to

    def to(self, device) -> "MaskMatrices":
        """Return a copy of this MaskMatrices with every tensor moved to `device` (e.g. 'cuda') -- called once per batch in the training/eval loops."""
        return MaskMatrices(
            vertex_edge_w1=self.vertex_edge_w1.to(device),
            vertex_edge_w2=self.vertex_edge_w2.to(device),
            vertex_edge_b1=self.vertex_edge_b1.to(device),
            vertex_edge_b2=self.vertex_edge_b2.to(device),
            mol_vertex_w=self.mol_vertex_w.to(device),
            mol_vertex_b=self.mol_vertex_b.to(device),
            begin_idx=self.begin_idx.to(device),
            end_idx=self.end_idx.to(device),
            batch=self.batch.to(device),
        )


def batch_graphs(graphs: List[MolGraph]):
    """Concatenate a list of per-molecule MolGraphs into one flat batch.

    Returns:
        atom_ftr [A,34], bond_ftr [E,10], pos [A,3], masses [A,1],
        adj3 [A,A] (block-diagonal), labels [M], smiles_list (len M),
        mask_matrices (MaskMatrices).

    Every molecule's begin_idx/end_idx is offset by the cumulative atom count
    of the molecules before it, so indices stay valid into the flat, batched
    atom array (this is the "Danger Zone 1" offset step called out in
    integration_final.md).
    """
    atom_ftrs, bond_ftrs, poss, masses_list, adj3_blocks, labels, smiles_list = (
        [], [], [], [], [], [], []
    )
    begin_idxs, end_idxs, batch_ids = [], [], []

    atom_offset = 0
    for m_id, g in enumerate(graphs):
        n_atoms = g.atom_ftr.shape[0]
        atom_ftrs.append(g.atom_ftr)
        bond_ftrs.append(g.bond_ftr)
        poss.append(g.pos)
        masses_list.append(g.masses)
        adj3_blocks.append(g.adj3)
        labels.append(g.label)
        smiles_list.append(g.smiles)

        begin_idxs.append(g.begin_idx + atom_offset)
        end_idxs.append(g.end_idx + atom_offset)
        batch_ids.append(torch.full((n_atoms,), m_id, dtype=torch.long))

        atom_offset += n_atoms

    atom_ftr = torch.cat(atom_ftrs, dim=0)
    bond_ftr = torch.cat(bond_ftrs, dim=0)
    pos = torch.cat(poss, dim=0)
    masses = torch.cat(masses_list, dim=0)
    begin_idx = torch.cat(begin_idxs, dim=0)
    end_idx = torch.cat(end_idxs, dim=0)
    batch = torch.cat(batch_ids, dim=0)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    A = atom_ftr.shape[0]
    E = bond_ftr.shape[0]
    M = len(graphs)

    # Block-diagonal adj3 [A, A].
    adj3 = torch.zeros((A, A), dtype=torch.float32)
    offset = 0
    for block in adj3_blocks:
        n = block.shape[0]
        adj3[offset:offset + n, offset:offset + n] = block
        offset += n

    # Dense vertex<->edge masks [A, E]. A and E are small per training batch
    # (hundreds, not thousands), so dense tensors here are fine -- this
    # matches the design docs' own MaskMatrices convention.
    vertex_edge_w1 = torch.zeros((A, E), dtype=torch.float32)
    vertex_edge_w2 = torch.zeros((A, E), dtype=torch.float32)
    edge_range = torch.arange(E)
    vertex_edge_w1[begin_idx, edge_range] = 1.0
    vertex_edge_w2[end_idx, edge_range] = 1.0
    neg_inf = -1e9
    vertex_edge_b1 = (1.0 - vertex_edge_w1) * neg_inf
    vertex_edge_b2 = (1.0 - vertex_edge_w2) * neg_inf

    mol_vertex_w = torch.zeros((M, A), dtype=torch.float32)
    mol_vertex_w[batch, torch.arange(A)] = 1.0
    mol_vertex_b = (1.0 - mol_vertex_w) * neg_inf

    mask_matrices = MaskMatrices(
        vertex_edge_w1=vertex_edge_w1,
        vertex_edge_w2=vertex_edge_w2,
        vertex_edge_b1=vertex_edge_b1,
        vertex_edge_b2=vertex_edge_b2,
        mol_vertex_w=mol_vertex_w,
        mol_vertex_b=mol_vertex_b,
        begin_idx=begin_idx,
        end_idx=end_idx,
        batch=batch,
    )

    return atom_ftr, bond_ftr, pos, masses, adj3, labels_t, smiles_list, mask_matrices
