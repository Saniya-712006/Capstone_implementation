"""
configs/config.py

Single source of truth for every hyperparameter and dimension used across the
PhysChem encoder, the CAL head, training, and the Phase-3 counterfactual
explainer. Every module in src/ imports its constants from here instead of
hardcoding numbers, so changing a dimension (e.g. HV_DIM) only ever requires
editing this one file.

Values below mirror Table 5 ("Training configuration") of the capstone report
and the architecture described in hld_physchem.md / hld_cal.md, not the
paper's original defaults (e.g. HV_DIM is 200 here, matching what was
actually used for the reported results, not the 128 default mentioned in the
architecture chapter).
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- Featurisation dims (fixed by the one-hot schemes in
    # src/data/mask_matrices.py -- do not change without updating those too) ----
    ATOM_FTR_DIM: int = 34
    BOND_FTR_DIM: int = 10

    # ---- Model hidden dims ----
    HV_DIM: int = 200   # atom hidden state size
    HE_DIM: int = 200   # bond hidden state size
    PQ_DIM: int = 3      # position / momentum are 3D

    # ---- PhysChem encoder loop ----
    N_LAYER: int = 2        # outer PhysNet<->ChemNet alternations
    N_ITERATION: int = 2    # Newton steps per PhysNet call
    TAU: float = 0.25       # Newton integration timestep
    RELA_CHUNK: int = 32    # row-chunk size for the O(A^2) repulsion force

    # ---- CAL head ----
    CAL_DROPOUT: float = 0.5

    # ---- Loss weights (Table 2 of the report) ----
    LAMBDA_C: float = 0.5
    LAMBDA_O: float = 1.0
    LAMBDA_CO: float = 0.5
    LAMBDA_CONF: float = 0.1

    # ---- Optimiser / schedule (Table 5 of the report) ----
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    LR_GAMMA: float = 0.99
    GRAD_CLIP_NORM: float = 1.0
    EPOCHS: int = 100
    BATCH_SIZE: int = 8
    ACCUMULATION_STEPS: int = 1

    # ---- Smoke test ----
    SMOKE_N_PER_SPLIT: int = 20

    # ---- Phase-3 counterfactual explainer (interface_planning.md /
    # integration_final.md "Approach B lightweight") ----
    CF_N_MUTATIONS: int = 500        # STONED candidates generated per query
    CF_CAUSAL_THRESHOLD: float = 0.5 # att_o above this = "causal atom"
    CF_MCS_MIN_OVERLAP: int = 1      # min causal atoms touched to survive filtering
    CF_DELTA: float = 1.0            # min |pEC50 shift| to count as a counterfactual
    CF_DBSCAN_EPS: float = 0.15
    CF_TOP_K_EACH_DIRECTION: int = 2 # up + down counterfactuals returned
    CF_PREDICT_BATCH_SIZE: int = 32  # batch size for scoring surviving candidates

    SEED: int = 42


DEFAULT_CONFIG = Config()
