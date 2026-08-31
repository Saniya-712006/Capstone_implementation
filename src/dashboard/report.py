"""
src/dashboard/report.py

Assembles everything in src/dashboard/ into one self-contained HTML string
-- meant to be displayed inline via `IPython.display.HTML(build_dashboard(...))`
in a Colab or Kaggle cell, not hosted anywhere (see the "where do I view it"
discussion in the project history: no server, no localhost, no tunnel --
just render it where you're already looking, on whichever platform you
currently happen to be running on).

Every section degrades gracefully instead of failing:
  - no results logged yet          -> training-curves section shows a note, not an error
  - no checkpoint given/found      -> attention/counterfactual sections show a note
  - phase3_query_smiles not given  -> Phase-3 section is simply omitted (it's
                                       the expensive step; opt-in per call, not automatic)
"""

import datetime
import statistics
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import torch

from configs.config import Config, DEFAULT_CONFIG
from src.dashboard.molecule_viz import render_causal_attention, render_changed_atoms, render_plain
from src.dashboard.training_curves import render_training_curves
from src.explain.counterfactual import generate_counterfactuals
from src.inference.predict import predict_with_attention
from src.models.physchem_cal import PhysChemCAL
from src.utils.checkpoint import load_checkpoint

_CSS = """
<style>
  .pcc-dash { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; color: #1f2937; background: #ffffff; padding: 16px; }
  .pcc-dash h1 { font-size: 20px; margin: 0 0 4px 0; }
  .pcc-dash .pcc-meta { color: #6b7280; font-size: 12px; margin-bottom: 20px; }
  .pcc-dash h2 { font-size: 16px; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; margin-top: 28px; }
  .pcc-dash .pcc-note { color: #6b7280; font-style: italic; font-size: 13px; }
  .pcc-dash .pcc-grid { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; }
  .pcc-dash .pcc-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; width: 220px; background: #fafafa; }
  .pcc-dash .pcc-card img { width: 100%; border-radius: 4px; background: #fff; }
  .pcc-dash .pcc-card .pcc-smiles { font-family: monospace; font-size: 10px; color: #6b7280; word-break: break-all; margin-top: 4px; }
  .pcc-dash .pcc-card .pcc-pred { font-weight: 600; margin-top: 4px; }
  .pcc-dash .pcc-badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px; font-weight: 600; margin-top: 4px; }
  .pcc-dash .pcc-badge-up { background: #dcfce7; color: #15803d; }
  .pcc-dash .pcc-badge-down { background: #fee2e2; color: #b91c1c; }
  .pcc-dash .pcc-query-block { border-left: 4px solid #7e22ce; padding-left: 14px; margin-top: 14px; }
</style>
"""


def load_model_for_dashboard(checkpoint_path: Optional[str], config: Config, device: torch.device):
    """Build a fresh PhysChemCAL and load `checkpoint_path` into it.

    Returns (model, label_mean, label_std) on success, or (None, None, None)
    if checkpoint_path is None, doesn't exist, or fails to load (corrupt
    file, etc.) -- this lets build_dashboard() skip model-dependent sections
    cleanly instead of crashing the whole report over a missing checkpoint.
    """
    if not checkpoint_path:
        return None, None, None
    model = PhysChemCAL(
        atom_ftr_dim=config.ATOM_FTR_DIM, bond_ftr_dim=config.BOND_FTR_DIM,
        hv_dim=config.HV_DIM, he_dim=config.HE_DIM, pq_dim=config.PQ_DIM,
        n_layer=config.N_LAYER, n_iteration=config.N_ITERATION, tau=config.TAU,
        rela_chunk=config.RELA_CHUNK, cal_dropout=config.CAL_DROPOUT,
    ).to(device)
    try:
        payload = load_checkpoint(checkpoint_path, model, map_location=str(device))
    except Exception as e:
        print(f"[dashboard] could not load checkpoint {checkpoint_path}: {e} -- skipping model-dependent sections.")
        return None, None, None
    model.eval()
    return model, payload["label_mean"], payload["label_std"]


def _section_training_curves(results_dir: str) -> str:
    img = render_training_curves(results_dir)
    if img is None:
        return f'<h2>Training curves</h2><p class="pcc-note">No epoch data logged yet in "{results_dir}" -- run some training first.</p>'
    return f'<h2>Training curves</h2><img src="{img}" style="max-width:100%;border:1px solid #e5e7eb;border-radius:8px;">'


def _molecule_card(smiles: str, img: Optional[str], pred: float, extra_html: str = "") -> str:
    img_html = f'<img src="{img}">' if img else '<div class="pcc-note">could not render</div>'
    return (
        f'<div class="pcc-card">{img_html}'
        f'<div class="pcc-pred">pEC50 = {pred:.3f}</div>'
        f'{extra_html}'
        f'<div class="pcc-smiles">{smiles}</div></div>'
    )


def _section_attention_gallery(model: PhysChemCAL, label_mean: float, label_std: float,
                                device: torch.device, example_smiles: List[str]):
    """Returns (html, all_att_o_values) -- the att_o values are reused by the histogram section."""
    cards = []
    all_att_o: List[float] = []
    for smi in example_smiles:
        pred, att_o = predict_with_attention(model, smi, label_mean, label_std, device)
        if att_o is None:
            cards.append(f'<div class="pcc-card pcc-note">Could not parse/embed:<br>{smi}</div>')
            continue
        all_att_o.extend(att_o.tolist())
        img = render_causal_attention(smi, att_o)
        cards.append(_molecule_card(smi, img, pred))

    html = (
        '<h2>Causal attention -- blue = scaffold, red = causal (att_o)</h2>'
        f'<div class="pcc-grid">{"".join(cards)}</div>'
    )
    return html, all_att_o


def _section_attention_histogram(all_att_o: List[float]) -> str:
    if not all_att_o:
        return ""
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.hist(all_att_o, bins=30, color="#7e22ce", alpha=0.8)
    ax.set_xlabel("att_o (causal weight)")
    ax.set_ylabel("atom count")
    ax.set_xlim(0, 1)
    ax.axvline(statistics.mean(all_att_o), color="#dc2626", linestyle="--", linewidth=1.5,
                label=f"mean={statistics.mean(all_att_o):.3f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    img = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    return (
        '<h2>att_o distribution across the gallery above</h2>'
        '<p class="pcc-note">A tight cluster near 0.5 means the causal/scaffold split hasn\'t '
        'differentiated much yet (expect this early in training); a spread toward 0 and 1 means '
        'CAL is confidently separating atoms.</p>'
        f'<img src="{img}" style="border:1px solid #e5e7eb;border-radius:8px;">'
    )


def _section_phase3(model: PhysChemCAL, label_mean: float, label_std: float, device: torch.device,
                     config: Config, query_smiles_list: List[str]) -> str:
    blocks = []
    for query_smiles in query_smiles_list:
        result = generate_counterfactuals(model, query_smiles, label_mean, label_std, device, config)
        query_pred, att_o = predict_with_attention(model, query_smiles, label_mean, label_std, device)
        query_img = render_causal_attention(query_smiles, att_o) if att_o is not None else None

        cf_cards = []
        for cf in result["counterfactuals"]:
            highlight_img = render_changed_atoms(query_smiles, cf.changed_atoms)
            result_img = render_plain(cf.smiles)
            badge_class = "pcc-badge-up" if cf.direction == "up" else "pcc-badge-down"
            extra = (
                f'<span class="pcc-badge {badge_class}">{cf.direction}</span><br>'
                f'similarity={cf.similarity:.2f} | causal overlap={cf.causal_overlap:.0%}'
            )
            imgs = (
                f'<div style="display:flex;gap:4px;">'
                f'<img src="{highlight_img}" style="width:48%;" title="orange = what changed">'
                f'<img src="{result_img}" style="width:48%;" title="resulting structure"></div>'
            )
            cf_cards.append(
                f'<div class="pcc-card">{imgs}'
                f'<div class="pcc-pred">pEC50 = {cf.pred:.3f}</div>{extra}'
                f'<div class="pcc-smiles">{cf.smiles}</div></div>'
            )

        if not cf_cards:
            cf_html = '<p class="pcc-note">No counterfactuals survived filtering for this query.</p>'
        else:
            cf_html = f'<div class="pcc-grid">{"".join(cf_cards)}</div>'

        alignment = result["alignment_score"]
        alignment_str = f"{alignment:.0%}" if alignment is not None else "n/a"
        blocks.append(
            f'<div class="pcc-query-block">'
            f'<div class="pcc-grid">{_molecule_card(query_smiles, query_img, query_pred)}</div>'
            f'<p><b>{result["causal_atom_count"]}</b> causal atoms | '
            f'CAL-counterfactual alignment: <b>{alignment_str}</b> '
            f'(fraction of edited atoms that were also flagged causal)</p>'
            f'{cf_html}</div>'
        )

    return '<h2>Phase-3: counterfactual explanations</h2>' + "".join(blocks)


def build_dashboard(results_dir: str, checkpoint_path: Optional[str] = None,
                     example_smiles: Optional[List[str]] = None,
                     phase3_query_smiles: Optional[List[str]] = None,
                     config: Optional[Config] = None, device: Optional[torch.device] = None,
                     title: str = "PhysChemCAL Dashboard") -> str:
    """Build the full self-contained dashboard HTML string.

    Args:
        results_dir: e.g. "results_saniya_colab" -- training curves are read from here.
        checkpoint_path: e.g. "checkpoints_saniya_colab/best_model.pt". If
            None or fails to load, attention/Phase-3 sections are skipped
            with a note rather than erroring.
        example_smiles: molecules to show in the causal-attention gallery.
            None/empty skips that section.
        phase3_query_smiles: molecules to run the (expensive) Phase-3
            explainer on fresh. None/empty skips that section entirely --
            this is opt-in per call, never automatic.
        config: hyperparameters (dims must match the checkpoint). Defaults
            to configs.config.DEFAULT_CONFIG.
        device: torch device. Defaults to cuda if available else cpu.
    Returns:
        one HTML string. Pass to IPython.display.HTML(...) to render inline.
    """
    config = config or DEFAULT_CONFIG
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, label_mean, label_std = load_model_for_dashboard(checkpoint_path, config, device)

    parts = [_CSS, f'<div class="pcc-dash"><h1>{title}</h1>',
             f'<div class="pcc-meta">Generated {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} '
             f'| results: {results_dir} | checkpoint: {checkpoint_path or "none loaded"}</div>']

    parts.append(_section_training_curves(results_dir))

    if model is None:
        parts.append('<h2>Causal attention</h2><p class="pcc-note">No checkpoint loaded -- pass checkpoint_path to see this section.</p>')
    elif example_smiles:
        gallery_html, all_att_o = _section_attention_gallery(model, label_mean, label_std, device, example_smiles)
        parts.append(gallery_html)
        parts.append(_section_attention_histogram(all_att_o))
    else:
        parts.append('<h2>Causal attention</h2><p class="pcc-note">No example_smiles given -- pass a few SMILES to see this section.</p>')

    if model is not None and phase3_query_smiles:
        parts.append(_section_phase3(model, label_mean, label_std, device, config, phase3_query_smiles))

    parts.append("</div>")
    return "\n".join(parts)
