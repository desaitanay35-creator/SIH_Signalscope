"""
Deterministic seeding for reproducible SignalScope training runs.

Responsible Team Member: Member 6 (MLOps & Config)
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds Python, NumPy, and PyTorch (CPU and, when present, CUDA) RNGs.

    Also disables cuDNN's nondeterministic autotuner when CUDA is present.
    Call this once, before manifest splitting, dataset construction, model
    initialization, and DataLoader creation - not per-epoch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
