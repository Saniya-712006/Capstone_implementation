"""
src/utils/git_sync.py

One function: best-effort periodic `git add results/ && commit && push` from
inside a long-running training loop, so progress is visible on GitHub (and
the results/<date>.md log survives) even if the Colab/Kaggle session dies
before the run finishes. Deliberately never raises -- a network hiccup or a
missing git identity must never crash a training run that's hours into
epoch 47. Git identity and remote auth (PAT token, if the repo is private)
must already be configured before training starts; see the "Authenticate to
GitHub" step near the top of notebooks/run_colab.ipynb -- a push attempted
before that is just logged as a failure and training continues.

Checkpoints are deliberately NOT pushed here -- they're tens of MB each and
git has no efficient binary diffing, so pushing one every N epochs would
bloat the repo's history fast. Point --checkpoint-dir at a Drive/Kaggle-Input
path instead for cross-session checkpoint persistence (see README).
"""

import subprocess


def push_results(results_dir: str = "results/", epoch: int = None) -> bool:
    """Stage, commit, and push `results_dir` only. Returns True on a successful push, False on any failure (logged, never raised).

    Args:
        results_dir: path to the results folder relative to the current
            working directory (main.py runs with cwd = repo root, since the
            notebook %cd's into it before invoking python).
        epoch: current epoch number, purely for the commit message.
    """
    try:
        subprocess.run(["git", "add", results_dir], check=True, capture_output=True, text=True, timeout=30)

        label = f"epoch {epoch}" if epoch is not None else "periodic"
        commit = subprocess.run(
            ["git", "commit", "-m", f"Training progress push ({label})"],
            capture_output=True, text=True, timeout=30,
        )
        if commit.returncode != 0 and "nothing to commit" in (commit.stdout + commit.stderr).lower():
            print(f"[git] nothing new in {results_dir} to push at {label}")
            return True
        if commit.returncode != 0:
            print(f"[git] commit failed, skipping push: {commit.stderr.strip()[:300]}")
            return False

        # Another session (a teammate, or a push from outside this training run) may have moved
        # origin/main since we started -- a plain `git push` would then be rejected with "fetch
        # first". Pull (merge, not rebase -- keeps this simple and safe to retry) before pushing so
        # a push race resolves itself instead of silently failing every epoch until someone notices.
        pull = subprocess.run(["git", "pull", "--no-edit", "--no-rebase"], capture_output=True, text=True, timeout=60)
        if pull.returncode != 0:
            print(f"[git] pull before push failed, skipping push this round: {pull.stderr.strip()[:300]}")
            return False

        push = subprocess.run(["git", "push"], capture_output=True, text=True, timeout=120)
        if push.returncode != 0:
            print(f"[git] push failed (training continues regardless): {push.stderr.strip()[:300]}")
            return False

        print(f"[git] pushed {results_dir} to origin ({label})")
        return True
    except Exception as e:
        print(f"[git] periodic push skipped due to error: {e}")
        return False
