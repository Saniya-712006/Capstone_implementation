"""
generate_dashboard.py

Run the dashboard entirely locally -- no Colab, no Kaggle, no notebook.

  - Training curves only need `results_<name>_<platform>/` to exist locally,
    which just means the repo is cloned/pulled -- results are pushed to git
    every 3 epochs regardless of which platform trained them.
  - The causal-attention gallery and Phase-3 sections additionally need an
    actual checkpoint file present on disk. Two ways to get one there:
    --checkpoint (a path you already downloaded yourself), or
    --drive-folder-link (a public "anyone with the link" Drive folder
    containing best_model.pt -- fetched automatically via gdown). Without
    either, those sections are skipped with a note, same
    graceful-degradation behaviour as the notebook version.

Since there's no Jupyter kernel to render IPython.display.HTML() into
locally, this writes the same HTML build_dashboard() produces to a file and
opens it in your default browser instead.

By default this is a one-shot run: it writes dashboard.html once and exits.
Pass --watch to keep it running in the foreground -- it re-renders the same
--out file every --watch-interval seconds (best-effort `git pull` for fresh
results_dir data, plus a fresh Drive re-download if --drive-folder-link is
set), so you just reload the browser tab to see new data. Ctrl+C to stop.

Usage:
    python generate_dashboard.py --results-dir results_saniya_colab
    python generate_dashboard.py --results-dir results_saniya_colab --checkpoint C:\\path\\to\\best_model.pt
    python generate_dashboard.py --results-dir results_saniya_colab --drive-folder-link "https://drive.google.com/drive/folders/XXXX?usp=sharing"
    python generate_dashboard.py --results-dir results_saniya_colab --checkpoint <path> --phase3-query "CC(=O)Oc1ccccc1C(=O)O"
    python generate_dashboard.py --results-dir results_saniya_colab --drive-folder-link "https://drive.google.com/drive/folders/XXXX?usp=sharing" --watch
"""

import argparse
import datetime
import glob
import os
import shutil
import subprocess
import time
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
                    help="Path to a checkpoint .pt already on disk. Omit to see training curves only.")
    p.add_argument("--drive-folder-link", default=None,
                    help="Alternative to --checkpoint: a public (\"anyone with the link\") Google Drive "
                         "folder URL containing best_model.pt -- downloaded automatically via gdown. "
                         "Always re-fetched fresh (never cached/skipped), so this reflects the latest "
                         "checkpoint on Drive every time you run this, not a stale local copy.")
    p.add_argument("--drive-cache-dir", default=".drive_checkpoint_cache",
                    help="Where --drive-folder-link's contents get downloaded to (default: .drive_checkpoint_cache).")
    p.add_argument("--example-smiles", nargs="+", default=None,
                    help="SMILES for the causal-attention gallery (default: aspirin/ibuprofen/anthracene; "
                         "only used if --checkpoint is given).")
    p.add_argument("--phase3-query", nargs="+", default=None,
                    help="SMILES to run Phase-3 counterfactuals on -- expensive, opt-in, requires --checkpoint.")
    p.add_argument("--out", default="dashboard.html", help="Output HTML file path (default: dashboard.html).")
    p.add_argument("--no-open", action="store_true", help="Write the file but don't auto-open a browser tab.")
    p.add_argument("--watch", action="store_true",
                    help="Keep running and re-render --out every --watch-interval seconds "
                         "(git pull + fresh Drive re-download each time) instead of exiting after one write. "
                         "Just reload the browser tab to see new data. Ctrl+C to stop.")
    p.add_argument("--watch-interval", type=int, default=60,
                    help="Seconds between re-renders in --watch mode (default: 60).")
    return p


def fetch_checkpoint_from_drive(folder_link: str, cache_dir: str) -> str:
    """Download best_model.pt out of a public Drive folder via gdown, return its local path.

    Always wipes `cache_dir` first and re-downloads from scratch -- gdown's
    `resume=True` skips re-downloading a file if the local one is already
    the same size, but a retrained checkpoint's file size barely changes
    between epochs (same tensor shapes every time), so resume could easily
    serve up a stale checkpoint without any indication it did so. A fresh
    ~16MB download every run is a small, honest price for never being wrong
    about which epoch you're actually looking at.
    """
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)

    import gdown  # imported lazily -- only needed for this optional path
    print(f"[dashboard] downloading checkpoint from Drive: {folder_link}")
    gdown.download_folder(url=folder_link, output=cache_dir, quiet=False, resume=False)

    matches = glob.glob(os.path.join(cache_dir, "**", "best_model.pt"), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No best_model.pt found anywhere under {cache_dir} after downloading {folder_link} -- "
            f"check the link points at (or contains) a checkpoint folder, and that it's shared as "
            f"\"anyone with the link\"."
        )
    return matches[0]


def _git_pull_best_effort() -> None:
    """Best-effort `git pull`, only called in --watch mode so each refresh can pick up new results_dir
    data pushed by a live training run elsewhere. Never raises -- not being a git repo, no remote
    configured, or a network hiccup should just skip the pull, not kill the watch loop.
    """
    try:
        result = subprocess.run(["git", "pull"], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"[dashboard] git pull skipped (not fatal): {result.stderr.strip()[:200]}")
    except Exception as e:
        print(f"[dashboard] git pull skipped due to error: {e}")


def _render_once(args: argparse.Namespace) -> None:
    """One fetch+build+write cycle: optional git pull, optional fresh Drive download, rebuild the HTML, write it to --out."""
    if args.watch:
        _git_pull_best_effort()

    checkpoint_path = args.checkpoint
    if args.drive_folder_link:
        checkpoint_path = fetch_checkpoint_from_drive(args.drive_folder_link, args.drive_cache_dir)

    example_smiles = args.example_smiles
    if example_smiles is None and checkpoint_path:
        example_smiles = _DEFAULT_EXAMPLE_SMILES

    html = build_dashboard(
        results_dir=args.results_dir,
        checkpoint_path=checkpoint_path,
        example_smiles=example_smiles,
        phase3_query_smiles=args.phase3_query,
        config=DEFAULT_CONFIG,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[dashboard] wrote {args.out} at {stamp}")


def main() -> None:
    """Parse args, render once, open the browser (unless --no-open), then either exit or loop re-rendering (--watch)."""
    args = build_arg_parser().parse_args()

    if args.watch and args.phase3_query:
        print(f"[dashboard] warning: --phase3-query reruns the (expensive) Phase-3 explainer every "
              f"{args.watch_interval}s under --watch -- consider dropping one of the two flags.")

    _render_once(args)

    if not args.no_open:
        webbrowser.open(f"file://{os.path.abspath(args.out)}")

    if not args.watch:
        return

    print(f"[dashboard] watch mode: re-rendering {args.out} every {args.watch_interval}s -- "
          f"just reload the browser tab to see new data. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(args.watch_interval)
            _render_once(args)
    except KeyboardInterrupt:
        print("[dashboard] watch stopped.")


if __name__ == "__main__":
    main()
