"""
Test-only checkpoint fixture for the RGB+frequency fusion detector.

Builds and saves a randomly-initialized (never ImageNet-pretrained, never
network-fetched) `RGBFrequencyFusionModel` checkpoint with correct
`model_config["architecture"]` metadata, so backend tests can exercise the
real checkpoint-loading/detector code path without committing (or
requiring) the actual production checkpoint. Never imported by production
code - test fixtures only.
"""

from __future__ import annotations

from pathlib import Path

import torch

from model.architectures.fusion_model import build_fusion_model

FIXTURE_SEED = 1234


def build_tiny_fusion_checkpoint(path: Path, seed: int = FIXTURE_SEED) -> Path:
    """Builds a small, deterministic `rgb_frequency_fusion` checkpoint and
    saves it to `path`. Returns `path` for convenience."""
    torch.manual_seed(seed)
    model = build_fusion_model(pretrained=False)
    model.eval()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_spec(),
        },
        path,
    )
    return path
