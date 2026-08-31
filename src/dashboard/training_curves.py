"""
src/dashboard/training_curves.py

Parses the per-epoch lines already written by src/utils/results_logger.py
(e.g. "- epoch 3/30: c_loss=0.0001 | o_loss=0.8524 | ... | val_rmse=1.3934")
out of every results_<name>_<platform>/*.md file, and plots all five loss
terms plus val_rmse over epoch as one matplotlib figure, returned as a
base64 PNG data URI.

Reads every .md file in the results dir (not just today's), so a run that
spans multiple days via --resume still produces one continuous curve --
epoch number is the x-axis, not which day or which "## run" section a line
happened to land in.
"""

import base64
import glob
import io
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: no display backend needed/available in Colab/Kaggle
import matplotlib.pyplot as plt

_EPOCH_LINE = re.compile(r"^- epoch (\d+)/(\d+): (.+)$")
_METRIC = re.compile(r"(\w+)=(-?[\d.]+)")


def parse_results_logs(results_dir: str) -> List[Tuple[int, Dict[str, float]]]:
    """Extract every "- epoch X/Y: k=v | k=v | ..." line from all .md files in `results_dir`.

    Args:
        results_dir: e.g. "results_saniya_colab" -- a folder of dated .md files.
    Returns:
        list of (epoch_number, {metric_name: value}) sorted by epoch number,
        deduplicated so a later occurrence of the same epoch (e.g. from a
        resumed run whose log line got re-written) overwrites an earlier one.
        Empty list if the folder doesn't exist or has no matching lines yet.
    """
    by_epoch: Dict[int, Dict[str, float]] = {}
    if not os.path.isdir(results_dir):
        return []

    for path in sorted(glob.glob(os.path.join(results_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                match = _EPOCH_LINE.match(line.strip())
                if not match:
                    continue
                epoch = int(match.group(1))
                metrics = {k: float(v) for k, v in _METRIC.findall(match.group(3))}
                by_epoch[epoch] = metrics

    return sorted(by_epoch.items())


def render_training_curves(results_dir: str) -> Optional[str]:
    """Plot c/o/co/conf/frag loss (left axis) and val_rmse (right axis) over epoch, from every log in `results_dir`.

    Returns:
        base64 PNG data URI, or None if no epoch data has been logged yet
        (e.g. training hasn't started, or this platform's results folder is empty).
    """
    epochs_data = parse_results_logs(results_dir)
    if not epochs_data:
        return None

    epochs = [e for e, _ in epochs_data]
    loss_keys = ["c_loss", "o_loss", "co_loss", "conf_loss", "frag_loss"]
    colors = {"c_loss": "#7e22ce", "o_loss": "#15803d", "co_loss": "#2563eb",
              "conf_loss": "#c2410c", "frag_loss": "#db2777"}

    fig, ax_loss = plt.subplots(figsize=(9, 4.5))
    for key in loss_keys:
        series = [m.get(key) for _, m in epochs_data]
        if all(v is None for v in series):
            continue  # metric didn't exist yet for older logs (e.g. frag_loss before it was added)
        ax_loss.plot(epochs, series, label=key, color=colors.get(key), linewidth=1.6)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(loc="upper left", fontsize=8, ncol=2)
    ax_loss.grid(alpha=0.25)

    val_rmse = [m.get("val_rmse") for _, m in epochs_data]
    if any(v is not None for v in val_rmse):
        ax_rmse = ax_loss.twinx()
        ax_rmse.plot(epochs, val_rmse, label="val_rmse", color="#dc2626", linewidth=2.2, linestyle="--")
        ax_rmse.set_ylabel("val_rmse (pEC50 units)", color="#dc2626")
        ax_rmse.tick_params(axis="y", colors="#dc2626")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
