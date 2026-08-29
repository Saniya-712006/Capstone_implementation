"""
src/utils/results_logger.py

Writes every run's output into results/YYYY-MM-DD.md -- one Markdown file per
calendar day, with each run appended as its own timestamped section rather
than overwriting the previous run's numbers. This is what main.py calls after
training/eval/counterfactual generation finishes; committing results/ to git
(`git add results/ && git commit -m "..."`) is a manual step the user does
after a run, per their workflow -- this module only ever appends locally.
"""

import datetime
import json
import os
from typing import Any, Dict, Optional


class ResultsLogger:
    """Appends structured run output to a single per-day Markdown log file.

    One instance is created per run (see main.py). Every `log_*` call appends
    a section to results/<today's date>.md; if that file doesn't exist yet
    (first run of the day) it is created with a top-level date heading first.
    """

    def __init__(self, results_dir: str, run_name: str):
        """
        Args:
            results_dir: path to the results/ folder (created if missing).
            run_name: short identifier for this run (e.g. "train_smoke_test",
                "phase3_counterfactuals") used as the section heading so
                multiple runs on the same day stay distinguishable.
        """
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir
        self.run_name = run_name
        self.today = datetime.date.today().isoformat()
        self.file_path = os.path.join(results_dir, f"{self.today}.md")
        self.run_started_at = datetime.datetime.now().strftime("%H:%M:%S")
        self._write_run_header()

    def _ensure_day_heading(self) -> None:
        """Create the file with a top-level '# YYYY-MM-DD' heading if it doesn't exist yet."""
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(f"# Results log -- {self.today}\n\n")

    def _append(self, text: str) -> None:
        self._ensure_day_heading()
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")

    def _write_run_header(self) -> None:
        """Start a new '## <run_name> (HH:MM:SS)' section for this run."""
        self._append(f"\n## {self.run_name} ({self.run_started_at})\n")

    def log_config(self, config_dict: Dict[str, Any]) -> None:
        """Dump the CLI args / Config used for this run as a fenced JSON block, so every run's log is self-contained enough to reproduce."""
        pretty = json.dumps(config_dict, indent=2, default=str)
        self._append(f"\n**Config:**\n```json\n{pretty}\n```\n")

    def log_epoch(self, epoch: int, total_epochs: int, metrics: Dict[str, float]) -> None:
        """Append one training-epoch row, e.g. epoch=15/100, metrics={'c_loss':0.0,...,'val_rmse':1.2545}."""
        metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self._append(f"- epoch {epoch}/{total_epochs}: {metric_str}\n")

    def log_best(self, epoch: int, val_rmse: float) -> None:
        """Record which epoch produced the best validation RMSE so far -- checked at the end of training."""
        self._append(f"\n**Best validation RMSE:** {val_rmse:.4f} (epoch {epoch})\n")

    def log_test_metrics(self, metrics: Dict[str, float]) -> None:
        """Append the final held-out OOD test metrics, once training is complete."""
        self._append("\n**Final OOD test metrics:**\n")
        for k, v in metrics.items():
            self._append(f"- {k}: {v:.4f}\n")

    def log_counterfactuals(self, query_smiles: str, query_pred: float,
                             causal_atom_count: int, counterfactuals: list,
                             alignment_score: Optional[float] = None) -> None:
        """Append one Phase-3 counterfactual-explanation result block for a single query molecule.

        Args:
            query_smiles: the molecule that was explained.
            query_pred: model's predicted pEC50 for the query.
            causal_atom_count: number of atoms with att_o above the causal threshold.
            counterfactuals: list of dicts, each with keys
                smiles, pred, similarity, direction ("up"/"down").
            alignment_score: fraction of counterfactual-changed atoms that
                overlapped CAL's causal atoms (cross-validation score from
                Stage 13 of integration_final.md); None if not computed.
        """
        self._append(f"\n### Counterfactuals for `{query_smiles}`\n")
        self._append(f"- predicted pEC50: {query_pred:.4f}\n")
        self._append(f"- causal atoms (att_o > threshold): {causal_atom_count}\n")
        if alignment_score is not None:
            self._append(f"- CAL-ExMol alignment score: {alignment_score:.4f}\n")
        self._append("\n| direction | SMILES | pred pEC50 | similarity |\n")
        self._append("|---|---|---|---|\n")
        for cf in counterfactuals:
            self._append(
                f"| {cf['direction']} | `{cf['smiles']}` | {cf['pred']:.4f} | {cf['similarity']:.3f} |\n"
            )

    def log_note(self, text: str) -> None:
        """Free-text note appended verbatim -- for anything that doesn't fit the structured helpers above."""
        self._append(f"\n{text}\n")
