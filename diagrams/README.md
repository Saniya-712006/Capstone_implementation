# Architecture diagrams

Hand-authored SVGs (no external tool dependency — plain `rect`/`line`/`text`,
directly embeddable in a report, slide deck, or Word/PowerPoint doc). Each
depicts the actual mechanism in the corresponding source file, not just a
labelled box for each module — verified against the code in `src/` as of the
commit these were added in.

| File | Covers |
|---|---|
| `01_overall_architecture.svg` | End-to-end pipeline: data → PhysChem Encoder → CAL Head → prediction/loss, with the optional Phase-3 branch |
| `02_featurization.svg` | `src/data/mask_matrices.py` — per-molecule featurisation and the block-diagonal batching trick |
| `03_initializer.svg` | `src/models/initializer.py` — the learned-momentum vs. real-geometry asymmetry (p vs. q) |
| `04_physnet.svg` | `src/models/physnet.py` — bond + relational forces, and the O(A²)→O(chunk×A) chunking fix |
| `05_chemnet.svg` | `src/models/chemnet.py` — triplet attention for one hub atom, grouped by source atom |
| `06_cal_head.svg` | `src/models/cal_head.py` — attention split, the three branches, and the shuffle+detach backdoor adjustment |
| `07_phase3_counterfactual.svg` | `src/explain/counterfactual.py` — the CPU-filter-before-GPU-score ordering that keeps Phase-3 cheap |

Regenerate/edit: `python generate.py` (from this folder or anywhere — paths
are resolved relative to the script's own location) rebuilds all 7 SVGs from
`svg_helpers.py`'s grid/box/arrow primitives. No dependencies beyond the
standard library. To preview changes as PNG before committing (optional,
not required to just run `generate.py`): `pip install cairosvg`, then
`cairosvg.svg2png(url="01_overall_architecture.svg", write_to="preview.png")`.
