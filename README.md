# PhysChemCAL

Implementation of the *"Physically Informed Counterfactually-Guided
Disentangled Representation Learning for Molecular Modeling"* capstone
project: a **PhysChem** encoder (learnable Newtonian physics + triplet-
attention chemistry) feeding a **CAL** (Causal Attention Learning) head that
disentangles causal chemistry from spurious scaffold correlations, trained
and evaluated on the **DrugOOD** out-of-distribution benchmark. An optional
**Phase-3** module generates CAL-guided counterfactual explanations.

This is a from-scratch reimplementation following the architecture worked
out in the project's design docs (`hld_physchem.md`, `hld_cal.md`,
`interface_planning.md`, `integration_final.md`) and the Phase-2 report --
not the earlier `capstone-v1.ipynb` proof-of-concept, which re-embedded 3D
conformers on the fly inside the training loop and was too fragile/slow to
build on.

## What's implemented, and one deliberate scope cut

- **PhysChem Stage 1 only** (Initializer -> PhysNet <-> ChemNet loop ->
  atom embeddings `hv`). Stage 2, the `FingerprintGenerator`
  (`GlobalReadout` + `GRUUnion`), is intentionally **not** implemented: CAL
  does its own attention-weighted pooling from atom-level to molecule-level,
  so a Stage-2 fingerprint would be computed but never consumed by any loss
  term. See `src/models/physchem_encoder.py` for the reasoning.
- **CAL head** follows the report's own Section 3.7 / Figure 3 description
  (linear+ReLU projection per branch -> attention-weighted mean pool ->
  LayerNorm -> MLP decoder), which is simpler than the edge-attention +
  branch-GCN version sketched earlier in `interface_planning.md`. See
  `src/models/cal_head.py` for why this version was chosen.
- **Phase-3 counterfactuals** use the "Approach B" design from
  `integration_final.md`: generate SELFIES mutations first (cheap), filter
  by causal-atom overlap with a Maximum Common Substructure check (cheap,
  still no model calls), and only 3D-embed + score the survivors, in
  batches, through the model (the one expensive step, minimised as much as
  the algorithm allows). Off by default -- pass `--phase3` to run it.

## Repository layout

```
capstone_complete/
  configs/config.py          All hyperparameters (single source of truth)
  src/
    data/
      mask_matrices.py       RDKit featurisation + MaskMatrices batching
      dataset.py             .pkl loading, smoke-test truncation, DataLoaders
    models/
      initializer.py         Stage 1a: seed hv/he + initial (p, q)
      physnet.py              PhysNet: learned Newtonian forces + update
      chemnet.py               ChemNet: triplet-attention message passing
      physchem_encoder.py     Wires Initializer + PhysNet/ChemNet loop
      cal_head.py              CAL causal/context/combined branches
      physchem_cal.py         Top-level model (encoder + CAL head)
    training/
      losses.py                c_loss / o_loss / co_loss / conf_loss
      train.py                  Training loop (AMP, grad accum, clipping)
      evaluate.py               RMSE evaluation
    inference/
      predict.py                predict_smiles() / predict_with_attention()
    explain/
      counterfactual.py        Phase-3 CAL-guided counterfactual explainer
    utils/
      seed.py, checkpoint.py, results_logger.py
  main.py                     CLI entrypoint (see "How to run" below)
  requirements.txt
  results/                    One <YYYY-MM-DD>.md log per day (committed)
```

## Data setup

Training/eval data is **never committed to this repo** -- the cached
`.pkl` files are too large for git (some exceed GitHub's 100MB hard file
limit) and Colab's manual upload widget doesn't survive a runtime restart.
Instead:

1. Preprocessing (raw DrugOOD `.json` -> 3D-embedded `.pkl` via RDKit
   ETKDG + MMFF94) is done **locally**, ahead of time, by the team -- this
   pipeline only ever consumes an already-built `.pkl`
   (schema: list of `{mol, smiles, label, split, group_id}` dicts, one
   `mol` per record already carrying a 3D conformer).
2. Upload the finished `.pkl`(s) to a Google Drive folder (not git).
3. In Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Point `--pkl-path` at the mounted file, e.g.
   `/content/drive/MyDrive/capstone_data/drugood_lbap_ec50_scaffold_3d.pkl`.

Only one `.pkl` needs to exist to start -- as more DrugOOD subsets get
preprocessed locally, just upload the new `.pkl` to the same Drive folder
and point `--pkl-path` at it; no code changes needed.

## Installation

```bash
git clone <this-repo-url>
cd capstone_complete
pip install -r requirements.txt
```

**On Colab specifically:** do **not** `pip install torch` -- Colab already
ships a torch build matched to its CUDA driver, and installing a different
one on top is the most common source of "works locally, breaks on Colab"
errors. `requirements.txt` leaves torch commented out for exactly this
reason. Nothing else in this repo depends on `torch-geometric`, which is the
other typical Colab install headache (fragile CUDA/torch wheel matching) --
the PhysChem/CAL implementation here uses the dense `MaskMatrices`
formulation from the design docs instead of PyG `Data` objects.

## How to run

Basic pipeline (PhysChem + CAL train, then evaluate on the OOD test split):

```bash
python main.py --pkl-path /content/drive/MyDrive/capstone_data/drugood_lbap_ec50_scaffold_3d.pkl
```

Fast pipeline sanity check on a truncated slice of the data (20 molecules
per split by default, 2 epochs) -- use this first, especially the first time
you run against a new `.pkl`:

```bash
python main.py --pkl-path <path> --smoke-test
```

Full pipeline plus Phase-3 counterfactual explanations (expensive --
explains 3 randomly sampled OOD-test molecules by default):

```bash
python main.py --pkl-path <path> --phase3
```

Explain specific molecules instead of random samples:

```bash
python main.py --pkl-path <path> --phase3 --query-smiles "CC(=O)Oc1ccccc1C(=O)O" "c1ccccc1"
```

Run Phase-3 against an already-trained checkpoint without retraining:

```bash
python main.py --pkl-path <path> --checkpoint checkpoints/best_model.pt --skip-train --phase3
```

Every run's config and metrics are appended to `results/<today>.md`
(training losses per epoch, best validation RMSE, final OOD test RMSE, and
any Phase-3 counterfactual tables). Commit that file after a run:

```bash
git add results/
git commit -m "Results: <date> smoke test / full run / phase3"
```

Other useful flags: `--epochs`, `--batch-size`, `--accumulation-steps`,
`--seed`, `--device`, `--n-queries` (Phase-3 sample count). Run
`python main.py --help` for the full list.

**If you hit a CUDA out-of-memory error**, reach for `--max-atoms` before
shrinking `--batch-size`: some DrugOOD entries (notably in
`lbap_ec50_scaffold`) are large cyclic peptides with hundreds of heavy atoms,
and the relational-force term's backward-pass memory scales with
`(atoms per batch)^2` -- a handful of these in one batch can exhaust GPU
memory regardless of batch size. `--smoke-test` applies a built-in 150-atom
cap automatically; for a real run, e.g. `--max-atoms 200` excludes anything
larger before batching.

## Team

Saanvi Manjunath - Shaikh Saniya Ali - Sameeksha KC - Sanat Shirwaicar
Capstone Project Phase 2, PES University, under Dr. Bhaskarjyoti Das.
