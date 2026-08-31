"""
generate_dashboard.py

Run the dashboard entirely locally -- no Colab, no Kaggle, no notebook.

  - Training curves only need `results_<name>_<platform>/` to exist locally,
    which just means the repo is cloned/pulled -- results are pushed to git
    every 3 epochs regardless of which platform trained them.
  - The causal-attention gallery and Phase-3 sections additionally need an
    actual checkpoint file present on disk (e.g. downloaded once from Google
    Drive) -- pass its path with --checkpoint. Without one, those sections
    are skipped with a note, same graceful-degradation behaviour as the
    notebook version.

Since there's no Jupyter kernel to render IPython.display.HTML() into
locally, this writes the same HTML build_dashboard() produces to a file and
opens it in your default browser instead.

Usage:
    python generate_dashboard.py --results-dir results_saniya_colab
    python generate_dashboard.py --results-dir results_saniya_colab --checkpoint C:\\path\\to\\best_model.pt
    python generate_dashboard.py --results-dir results_saniya_colab --checkpoint <path> \\
        --phase3-query "CC(=O)Oc1ccccc1C(=O)O"
"""

import argparse
import os
import webbrowser

from configs.config import DEFAULT_CONFIG
from src.dashboard.report import build_dashboard

_DEFAULT_EXAMPLE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "c1ccc2c(c1)ccc1ccccc12",      # anthracene
]


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the CLI flags: which results/checkpoint to read, which molecules to show, and where to write the output HTML."""
    p = argparse.ArgumentParser(description="Generate the PhysChemCAL dashboard as a local HTML file (no notebook needed).")
    p.add_argument("--results-dir", required=True,
                    help="e.g. results_saniya_colab -- must exist locally (clone/pull the repo first).")
    p.add_argument("--checkpoint", default=None,
                    help="Path to a checkpoint .pt (e.g. downloaded from Drive). Omit to see training curves only.")
    p.add_argument("--example-smiles", nargs="+", default=None,
                    help="SMILES for the causal-attention gallery (default: aspirin/ibuprofen/anthracene; "
                         "only used if --checkpoint is given).")
    p.add_argument("--phase3-query", nargs="+", default=None,
                    help="SMILES to run Phase-3 counterfactuals on -- expensive, opt-in, requires --checkpoint.")
    p.add_argument("--out", default="dashboard.html", help="Output HTML file path (default: dashboard.html).")
    p.add_argument("--no-open", action="store_true", help="Write the file but don't auto-open a browser tab.")
    return p


def main() -> None:
    """Parse args, build the dashboard HTML, write it to disk, and open it in the default browser (unless --no-open)."""
    args = build_arg_parser().parse_args()

    example_smiles = args.example_smiles
    if example_smiles is None and args.checkpoint:
        example_smiles = _DEFAULT_EXAMPLE_SMILES

    html = build_dashboard(
        results_dir=args.results_dir,
        checkpoint_path=args.checkpoint,
        example_smiles=example_smiles,
        phase3_query_smiles=args.phase3_query,
        config=DEFAULT_CONFIG,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[dashboard] wrote {args.out}")

    if not args.no_open:
        webbrowser.open(f"file://{os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
