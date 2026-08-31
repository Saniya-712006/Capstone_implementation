"""
src/dashboard/molecule_viz.py

Renders a molecule's 2D structure as a PNG (returned as a base64 data URI,
ready to drop into an <img src="..."> tag) with atoms colored to show what
CAL is doing:

  render_causal_attention(): heatmap by att_o (blue = scaffold, red = causal)
      -- the "here's what CAL thinks matters" picture.
  render_changed_atoms(): a single highlight color on whichever atoms differ
      from a query molecule -- the "here's what Phase-3 actually edited"
      picture, used side-by-side with the query's attention picture so a
      viewer can see whether the edited atoms were the causal ones.
"""

import base64
from typing import Dict, Optional, Set

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from src.dashboard.colors import attention_to_rgb

_IMG_SIZE = 350
_CHANGED_ATOM_COLOR = (1.0, 0.65, 0.0)  # orange -- distinct from the blue/red attention heatmap


def _draw_to_data_uri(mol: "Chem.Mol", highlight_colors: Optional[Dict[int, tuple]]) -> str:
    """Shared rendering step: 2D-depict `mol`, optionally color specific atoms, return a base64 PNG data URI."""
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(_IMG_SIZE, _IMG_SIZE)
    if highlight_colors:
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=list(highlight_colors.keys()),
            highlightAtomColors=highlight_colors,
        )
    else:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_causal_attention(smiles: str, att_o) -> Optional[str]:
    """Render `smiles` with every atom colored by its att_o weight (coolwarm: blue=scaffold, red=causal).

    Args:
        smiles: the molecule to draw.
        att_o: array-like of per-atom causal weights, length = heavy atom
            count, in the same RDKit atom order mol_to_graph used (this is
            exactly what src/inference/predict.py's predict_with_attention returns).
    Returns:
        base64 PNG data URI, or None if `smiles` fails to parse.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    colors = {i: attention_to_rgb(float(att_o[i])) for i in range(mol.GetNumAtoms())}
    return _draw_to_data_uri(mol, colors)


def render_changed_atoms(smiles: str, changed_atoms: Set[int]) -> Optional[str]:
    """Render `smiles` (a counterfactual candidate) with the atoms that differ from the query highlighted in one fixed color.

    Args:
        smiles: the counterfactual candidate's SMILES.
        changed_atoms: query-molecule atom indices that changed (from
            Counterfactual.changed_atoms) -- note these indices are valid on
            the *query* molecule, so this is meant to be called on the query
            itself for the "before" picture; the candidate's own structure
            is rendered plainly (its atom indices don't correspond 1:1 with
            the query's after an edit, so highlighting "changed atoms" only
            makes unambiguous sense on the query side).
    Returns:
        base64 PNG data URI, or None if `smiles` fails to parse.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    colors = {i: _CHANGED_ATOM_COLOR for i in changed_atoms if i < mol.GetNumAtoms()}
    return _draw_to_data_uri(mol, colors or None)


def render_plain(smiles: str) -> Optional[str]:
    """Render `smiles` with no highlighting at all -- used for a counterfactual candidate's own structure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _draw_to_data_uri(mol, None)
