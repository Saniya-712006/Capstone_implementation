"""
src/inference/predict.py

The one place in this codebase that embeds a molecule's 3D geometry at
runtime (ETKDG + MMFF94), because it is the one place that legitimately needs
to: turning an arbitrary, never-before-seen SMILES string into a prediction.
Everywhere else (training/eval), molecules arrive pre-embedded in the cached
.pkl -- see src/data/dataset.py's module docstring for why that split exists.

Two entrypoints:
  predict_smiles():          SMILES in, denormalised pEC50 out (batched,
                              graceful NaN for anything that fails to parse/embed).
  predict_with_attention():  same, but for a single molecule, also returning
                              the per-atom att_o causal-attention weights --
                              this is what src/explain/counterfactual.py
                              (Phase 3) calls to find a query molecule's
                              "causal atoms" before generating counterfactuals.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from src.data.mask_matrices import MolGraph, batch_graphs, mol_to_graph
from src.models.physchem_cal import PhysChemCAL

_ETKDG_MAX_ITERS = 200
_MMFF_MAX_ITERS = 500


def smiles_to_3d_mol(smiles: str, seed: int = 42) -> Optional["Chem.Mol"]:
    """Parse a raw SMILES string and embed a single 3D conformer via ETKDG + MMFF94 (mirrors the offline preprocessing pipeline exactly, just run for one molecule instead of the whole dataset).

    Returns None if RDKit can't parse the SMILES or embedding fails --
    callers treat that as "skip this molecule", never as a hard error.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    params.maxIterations = _ETKDG_MAX_ITERS
    if AllChem.EmbedMolecule(mol, params) == -1:
        return None

    try:
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=_MMFF_MAX_ITERS)
    except Exception:
        pass  # optimisation failing is non-fatal -- the ETKDG geometry alone is still usable.

    return mol


def _graph_for_smiles(smiles: str, seed: int = 42) -> Optional[MolGraph]:
    """SMILES -> 3D Mol -> MolGraph in one call, or None if either step fails."""
    mol = smiles_to_3d_mol(smiles, seed=seed)
    if mol is None:
        return None
    return mol_to_graph(mol, label=0.0, smiles=smiles)  # label is a placeholder, unused at inference


@torch.no_grad()
def predict_smiles(model: PhysChemCAL, smiles_list: List[str], label_mean: float, label_std: float,
                    device: torch.device, batch_size: int = 32) -> List[float]:
    """Run the full SMILES -> pEC50 pipeline for a list of molecules.

    Args:
        model: trained PhysChemCAL (put in eval mode by this function).
        smiles_list: raw SMILES strings.
        label_mean, label_std: training-set label normalisation stats (from
            the checkpoint) used to denormalise predictions.
        device: torch device to run the model on.
        batch_size: molecules per forward pass (candidates are batched rather
            than predicted one at a time -- see Phase-3 counterfactual notes
            on why this matters for throughput).
    Returns:
        list the same length as smiles_list; float('nan') at any index whose
        SMILES failed to parse or embed.
    """
    was_training = model.training
    model.eval()

    graphs: List[Optional[MolGraph]] = [_graph_for_smiles(s) for s in smiles_list]
    valid_positions = [i for i, g in enumerate(graphs) if g is not None]
    results = [float("nan")] * len(smiles_list)

    for start in range(0, len(valid_positions), batch_size):
        chunk_positions = valid_positions[start:start + batch_size]
        chunk_graphs = [graphs[i] for i in chunk_positions]

        atom_ftr, bond_ftr, pos, masses, adj3, _labels, _smiles, mask_matrices = batch_graphs(chunk_graphs)
        atom_ftr, bond_ftr, pos, masses = (t.to(device) for t in (atom_ftr, bond_ftr, pos, masses))
        mask_matrices = mask_matrices.to(device)

        outputs = model(atom_ftr, bond_ftr, pos, masses, mask_matrices)
        preds = (outputs["o_pred"] * label_std + label_mean).cpu().tolist()

        for pos_idx, pred in zip(chunk_positions, preds):
            results[pos_idx] = pred

    if was_training:
        model.train()
    return results


@torch.no_grad()
def predict_with_attention(model: PhysChemCAL, smiles: str, label_mean: float, label_std: float,
                            device: torch.device, seed: int = 42
                            ) -> Tuple[float, Optional[np.ndarray]]:
    """Predict pEC50 for one molecule AND return its per-atom causal attention (att_o).

    Returns:
        (prediction, att_o) where att_o is a 1-D numpy array of length
        (heavy atom count), one causal-attention weight per atom in RDKit
        atom order (matching the order mol_to_graph's atom_features loop
        used, so index i corresponds directly to heavy atom i of the
        RDKit Mol). Returns (nan, None) if the SMILES fails to parse/embed.
    """
    was_training = model.training
    model.eval()

    graph = _graph_for_smiles(smiles, seed=seed)
    if graph is None:
        if was_training:
            model.train()
        return float("nan"), None

    atom_ftr, bond_ftr, pos, masses, adj3, _labels, _smiles, mask_matrices = batch_graphs([graph])
    atom_ftr, bond_ftr, pos, masses = (t.to(device) for t in (atom_ftr, bond_ftr, pos, masses))
    mask_matrices = mask_matrices.to(device)

    outputs = model(atom_ftr, bond_ftr, pos, masses, mask_matrices)
    pred = (outputs["o_pred"] * label_std + label_mean).item()
    att_o = outputs["att_o"].squeeze(-1).cpu().numpy()

    if was_training:
        model.train()
    return pred, att_o
