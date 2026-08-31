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
results_dir data), so you just reload the browser tab to see new data.
Ctrl+C to stop. The Drive checkpoint itself refreshes on its own, slower
cadence (--checkpoint-refresh-seconds, default 3600 = hourly) since it's a
~16MB download and checkpoints don't change nearly as often as --watch-interval
ticks -- every fetch overwrites the same local file in place, never piling
up copies.

Usage:
    python generate_dashboard.py --results-dir results_saniya_colab
    python generate_dashboard.py --results-dir results_saniya_colab --checkpoint C:\\path\\to\\best_model.pt
    python generate_dashboard.py --results-dir results_saniya_colab --drive-folder-link "https://drive.google.com/drive/folders/XXXX?usp=sharing"
    python generate_dashboard.py --results-dir results_saniya_colab --checkpoint <path> --phase3-query "CC(=O)Oc1ccccc1C(=O)O"
    python generate_dashboard.py --results-dir results_saniya_colab --drive-folder-link "https://drive.google.com/drive/folders/XXXX?usp=sharing" --watch
"""

import argparse
import datetime
import os
import subprocess
import time
import webbrowser
from typing import Dict

from configs.config import DEFAULT_CONFIG
from src.dashboard.report import build_dashboard


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the CLI flags: which results/checkpoint to read, which molecules to show, and where to write the output HTML."""
    p = argparse.ArgumentParser(description="Generate the PhysChemCAL dashboard as a local HTML file (no notebook needed).")
    p.add_argument("--results-dir", required=True,
                    help="e.g. results_saniya_colab -- must exist locally (clone/pull the repo first).")
    p.add_argument("--checkpoint", default=None,
                    help="Path to a checkpoint .pt already on disk. Omit to see training curves only.")
    p.add_argument("--drive-folder-link", default=None,
                    help="Alternative to --checkpoint: a public (\"anyone with the link\") Google Drive "
                         "folder URL containing best_model.pt -- downloaded automatically via gdown, "
                         "always overwriting the same local file in place (never accumulates copies).")
    p.add_argument("--drive-cache-dir", default=".drive_checkpoint_cache",
                    help="Where --drive-folder-link's best_model.pt gets downloaded to (default: .drive_checkpoint_cache).")
    p.add_argument("--checkpoint-refresh-seconds", type=int, default=3600,
                    help="In --watch mode, minimum seconds between re-downloading the Drive checkpoint "
                         "(default: 3600 = once an hour). Training runs for far longer than one epoch "
                         "per minute, so re-fetching a ~16MB file every --watch-interval is wasted "
                         "bandwidth; the training-curves section still refreshes every --watch-interval "
                         "regardless, via the cheap git pull.")
    p.add_argument("--example-smiles", nargs="+", default=None,
                    help="SMILES to force into the causal-attention gallery. Omit this to auto-pick: "
                         "the model's actual last training batch (from results_dir/live_batch.json) "
                         "if a training run has written one, else a small fixed demo list.")
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


_CHECKPOINT_FILENAME = "best_model.pt"


def fetch_checkpoint_from_drive(folder_link: str, cache_dir: str) -> str:
    """Download only best_model.pt out of a public Drive folder, to one fixed local path, via gdown.

    Lists the folder's contents first (skip_download=True -- no bytes moved
    yet) to find best_model.pt's file id, then downloads just that one file
    straight to `cache_dir/best_model.pt` every time. gdown.download() with a
    fixed output path overwrites that exact file in place -- there's no
    per-call subfolder or timestamped copy, so repeated calls (e.g. every
    hour under --watch) never accumulate anything on disk, and latest_model.pt
    (unused here) is never fetched at all.
    """
    os.makedirs(cache_dir, exist_ok=True)
    target_path = os.path.join(cache_dir, _CHECKPOINT_FILENAME)

    import gdown  # imported lazily -- only needed for this optional path
    print(f"[dashboard] checking Drive folder for {_CHECKPOINT_FILENAME}: {folder_link}")
    entries = gdown.download_folder(url=folder_link, skip_download=True, quiet=True)
    match = next((e for e in entries if os.path.basename(e.path) == _CHECKPOINT_FILENAME), None)
    if match is None:
        raise FileNotFoundError(
            f"No {_CHECKPOINT_FILENAME} found in Drive folder {folder_link} -- check the link points at "
            f"(or contains) a checkpoint folder, and that it's shared as \"anyone with the link\"."
        )

    print(f"[dashboard] downloading {_CHECKPOINT_FILENAME} -> {target_path}")
    gdown.download(id=match.id, output=target_path, quiet=False, resume=False)
    return target_path


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


def _render_once(args: argparse.Namespace, state: Dict[str, object]) -> None:
    """One fetch+build+write cycle: optional git pull, optional Drive download (rate-limited by
    --checkpoint-refresh-seconds via `state`), rebuild the HTML, write it to --out.
    """
    if args.watch:
        _git_pull_best_effort()

    checkpoint_path = args.checkpoint
    if args.drive_folder_link:
        now = time.monotonic()
        last_fetch = state.get("checkpoint_fetched_at")
        due = last_fetch is None or (now - last_fetch) >= args.checkpoint_refresh_seconds
        if due:
            state["checkpoint_path"] = fetch_checkpoint_from_drive(args.drive_folder_link, args.drive_cache_dir)
            state["checkpoint_fetched_at"] = now
        else:
            remaining = int(args.checkpoint_refresh_seconds - (now - last_fetch))
            print(f"[dashboard] checkpoint refresh not due yet -- reusing local copy (next check in {remaining}s)")
        checkpoint_path = state["checkpoint_path"]

    html = build_dashboard(
        results_dir=args.results_dir,
        checkpoint_path=checkpoint_path,
        example_smiles=args.example_smiles,
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

    state: Dict[str, object] = {}
    _render_once(args, state)

    if not args.no_open:
        webbrowser.open(f"file://{os.path.abspath(args.out)}")

    if not args.watch:
        return

    print(f"[dashboard] watch mode: re-rendering {args.out} every {args.watch_interval}s "
          f"(checkpoint re-fetched at most every {args.checkpoint_refresh_seconds}s) -- "
          f"just reload the browser tab to see new data. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(args.watch_interval)
            _render_once(args, state)
    except KeyboardInterrupt:
        print("[dashboard] watch stopped.")


if __name__ == "__main__":
    main()
