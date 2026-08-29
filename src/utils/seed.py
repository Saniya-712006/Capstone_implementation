"""
src/utils/seed.py

One function, one job: make a run reproducible by seeding every RNG the
pipeline touches (Python's random, numpy, torch CPU, torch CUDA). Called once
at the top of main.py before any data loading or model construction happens.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch (CPU + all CUDA devices) with the same value.

    Args:
        seed: integer seed to apply everywhere.

    Note: this makes RDKit's own embedding calls (ETKDG) reproducible only if
    the caller also passes `seed` into those calls explicitly -- RDKit does
    not read these global RNGs. Since this codebase never re-embeds molecules
    at runtime (data arrives pre-embedded in the .pkl), that's a non-issue
    here, but worth remembering if the preprocessing scripts are touched.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
