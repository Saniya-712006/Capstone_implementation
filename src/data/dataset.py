"""
src/data/dataset.py

Loads a cached, pre-3D-embedded DrugOOD .pkl (schema: list of
{mol, smiles, label, split, group_id} records, exactly what
process_*.py / the notebooks in Capstone_implementation already produce --
see README.md "Data setup"), turns each record into a MolGraph via
mask_matrices.mol_to_graph, and exposes torch DataLoaders for the
train / ood_val / ood_test splits.

This module deliberately does NOT fall back to building a .pkl from raw
DrugOOD JSON -- that preprocessing step is done locally by the team ahead of
time (per project decision) and only the resulting .pkl is ever handed to
this pipeline, via --pkl-path pointing at a mounted Google Drive folder.

Label normalisation: stats (mean/std) are always computed from the TRAIN
split only, then applied to val/test too, and are returned alongside the
loaders so callers (train.py, checkpoint saving) can denormalise predictions
back to pEC50 units later.
"""

import pickle
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.mask_matrices import MolGraph, batch_graphs, mol_to_graph

DEFAULT_SPLITS = ("train", "ood_val", "ood_test")


def move_batch_to_device(batch: Dict[str, object], device) -> Dict[str, object]:
    """Move every tensor field of a collated batch dict (and its MaskMatrices) to `device`; leaves the `smiles` list untouched (it's plain Python strings, nothing to move)."""
    moved = dict(batch)
    for key in ("atom_ftr", "bond_ftr", "pos", "masses", "adj3", "labels", "raw_labels"):
        moved[key] = batch[key].to(device)
    moved["mask_matrices"] = batch["mask_matrices"].to(device)
    return moved


def load_pkl_records(pkl_path: str) -> List[dict]:
    """Load the raw list-of-dict records from a cached DrugOOD .pkl file (no featurisation yet)."""
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    if not isinstance(records, list):
        raise ValueError(
            f"Expected {pkl_path} to contain a list of records "
            f"(schema: mol/smiles/label/split/group_id), got {type(records)}"
        )
    return records


def records_to_graphs(records: List[dict], split_name: str,
                       limit: Optional[int] = None) -> List[MolGraph]:
    """Filter `records` to one split, optionally truncate (smoke test), and featurise each surviving record into a MolGraph.

    Records whose RDKit Mol fails featurisation (mol_to_graph returns None --
    e.g. zero bonds, missing conformer) are silently skipped and counted, not
    raised on, matching the robustness behaviour already documented for the
    original collate function.

    Args:
        records: full list loaded by load_pkl_records().
        split_name: one of the DrugOOD split keys, e.g. "train", "ood_val", "ood_test".
        limit: if set, only the first `limit` matching records are used
            (this is exactly what --smoke-test wires up -- see main.py).
    """
    matching = [r for r in records if r.get("split") == split_name]
    if limit is not None:
        matching = matching[:limit]

    graphs = []
    n_failed = 0
    for rec in matching:
        g = mol_to_graph(rec["mol"], rec["label"], smiles=rec.get("smiles"))
        if g is None:
            n_failed += 1
            continue
        graphs.append(g)

    print(f"[data] split='{split_name}': {len(graphs)} usable molecules "
          f"({n_failed} skipped: failed featurisation) out of {len(matching)} requested.")
    return graphs


def compute_label_stats(graphs: List[MolGraph]) -> Tuple[float, float]:
    """Compute (mean, std) of raw labels across `graphs`; std is floored at 1e-6 to avoid divide-by-zero on a degenerate (e.g. 1-molecule smoke) split."""
    labels = torch.tensor([g.label for g in graphs], dtype=torch.float32)
    mean = labels.mean().item()
    std = max(labels.std().item(), 1e-6)
    return mean, std


class DrugOODGraphDataset(Dataset):
    """Thin torch Dataset wrapper around a pre-built list of MolGraph objects."""

    def __init__(self, graphs: List[MolGraph]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> MolGraph:
        return self.graphs[idx]


def make_collate_fn(label_mean: float, label_std: float):
    """Build a collate_fn closed over the (train-set) label normalisation stats, so every batch's `labels` tensor comes back z-normalised and ready for the loss functions.

    The returned batch is a dict (not a namedtuple/Data object) with keys:
        atom_ftr, bond_ftr, pos, masses, adj3, mask_matrices,
        labels (z-normalised, for the loss), raw_labels (original pEC50
        scale, for RMSE reporting), smiles (list[str], len M).
    """

    def collate(graph_list: List[MolGraph]) -> Dict[str, object]:
        atom_ftr, bond_ftr, pos, masses, adj3, raw_labels, smiles_list, mask_matrices = (
            batch_graphs(graph_list)
        )
        labels = (raw_labels - label_mean) / label_std
        return {
            "atom_ftr": atom_ftr,
            "bond_ftr": bond_ftr,
            "pos": pos,
            "masses": masses,
            "adj3": adj3,
            "mask_matrices": mask_matrices,
            "labels": labels,
            "raw_labels": raw_labels,
            "smiles": smiles_list,
        }

    return collate


def get_dataloaders(pkl_path: str, batch_size: int, smoke_test: bool = False,
                     smoke_n: int = 20, splits: Tuple[str, ...] = DEFAULT_SPLITS,
                     num_workers: int = 0) -> Dict[str, object]:
    """Build train/val/test DataLoaders (plus label stats) from one cached .pkl file.

    This is the single entrypoint main.py calls for data setup. Handles:
      - reading the .pkl once,
      - featurising each split (truncated to `smoke_n` per split if
        `smoke_test` is True -- this is the entire smoke-vs-full-dataset
        mechanism, no separate smoke data file needed),
      - computing label normalisation stats from the train split only,
      - wrapping each split in a DataLoader with a shared collate_fn.

    Returns:
        dict with keys "train_loader", "val_loader", "test_loader",
        "label_mean", "label_std", and per-split molecule counts under
        "n_train"/"n_val"/"n_test" (handy for results_logger.log_config()).
    """
    records = load_pkl_records(pkl_path)
    limit = smoke_n if smoke_test else None

    train_graphs = records_to_graphs(records, splits[0], limit=limit)
    val_graphs = records_to_graphs(records, splits[1], limit=limit)
    test_graphs = records_to_graphs(records, splits[2], limit=limit)

    if len(train_graphs) == 0:
        raise RuntimeError(
            f"No usable training molecules found in {pkl_path} for split '{splits[0]}'. "
            f"Check the .pkl was built with the expected split names."
        )

    label_mean, label_std = compute_label_stats(train_graphs)
    collate_fn = make_collate_fn(label_mean, label_std)

    # drop_last=True avoids a stray batch-size-1 final batch (documented as a
    # LayerNorm edge case in the report); disabled in smoke-test mode instead,
    # since smoke_n can be smaller than batch_size and drop_last=True there
    # would silently produce zero batches.
    train_drop_last = (not smoke_test) and len(train_graphs) > batch_size
    train_loader = DataLoader(
        DrugOODGraphDataset(train_graphs), batch_size=batch_size, shuffle=True,
        drop_last=train_drop_last, collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        DrugOODGraphDataset(val_graphs), batch_size=batch_size, shuffle=False,
        drop_last=False, collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        DrugOODGraphDataset(test_graphs), batch_size=batch_size, shuffle=False,
        drop_last=False, collate_fn=collate_fn, num_workers=num_workers,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "label_mean": label_mean,
        "label_std": label_std,
        "n_train": len(train_graphs),
        "n_val": len(val_graphs),
        "n_test": len(test_graphs),
    }
