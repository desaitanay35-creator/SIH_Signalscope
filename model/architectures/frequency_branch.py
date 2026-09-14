"""
Frequency-domain forensic branch for SignalScope (Step 6).

Defines:
  - FrequencyBranch: a small CNN mapping a (N, 1, H, W) log-magnitude FFT
    spectrum to a (N, EMBEDDING_DIM) embedding.
  - FrequencyOnlyModel: FrequencyBranch + a single-logit head - Experiment
    2 (frequency-only) of the Step 6 ablation.

See .claude/specs/06-frequency-fusion.md ("Frequency branch architecture")
for the full design rationale (why this depth/width, why global average
pooling, why no pretrained initialization exists for this branch).

IMPORTANT: a prediction from FrequencyOnlyModel is evidence of a spectral
pattern correlating with the training label - it is NOT, on its own,
evidence of genuine generative-forensic signal. GenImage-family datasets
are known to carry source/compression/resolution biases that a frequency
branch is exactly the kind of feature likely to exploit as a shortcut. See
.claude/specs/06-frequency-fusion.md ("How to interpret the experiment",
Case D) before drawing any conclusion from this model's results.

Responsible Team Member: Member 1 (Core ML & Model Architecture)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from torch import nn

ARCHITECTURE_ID = "frequency_branch"
INPUT_CHANNELS = 1
CONV_CHANNELS = (16, 32, 64)
EMBEDDING_DIM = CONV_CHANNELS[-1]
CLASSIFIER_DROPOUT = 0.4

PROBABILITY_MEANING = (
    "sigmoid(logit) = estimated probability that the input image's "
    "frequency spectrum resembles the training set's AI-generated class. "
    "This is a raw model probability, NOT evidence of genuine "
    "generative-forensic signal on its own - see "
    ".claude/specs/06-frequency-fusion.md (\"How to interpret the "
    "experiment\")."
)


@dataclass(frozen=True)
class FrequencyModelSpec:
    """Plain-data specification of the frequency-only model, for
    logging/reporting (mirrors EfficientNetB4Baseline.get_spec())."""

    architecture: str
    input_channels: int
    conv_channels: List[int]
    embedding_dim: int
    num_output_logits: int
    probability_meaning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FrequencyBranch(nn.Module):
    """Shallow CNN over a single-channel log-magnitude FFT spectrum.

    Deliberately small (three conv blocks, global average pool): spectral
    forensic signal (radial energy falloff, discrete periodic peaks) is a
    comparatively global, low-complexity pattern relative to natural-image
    texture/shape, so a deep hierarchical backbone is not warranted. See
    .claude/specs/06-frequency-fusion.md.
    """

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3 = CONV_CHANNELS
        self.features = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, c1, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the (N, EMBEDDING_DIM) pooled embedding."""
        features = self.features(x)
        pooled = self.pool(features)
        return torch.flatten(pooled, 1)


class FrequencyOnlyModel(nn.Module):
    """Frequency branch + single-logit head - Experiment 2 of the Step 6
    ablation.

    `forward()` accepts either a plain (N, 1, H, W) tensor, or a dict
    produced by model.training.dual_branch_dataset (e.g.
    {"rgb": ..., "frequency": ...}) - only the "frequency" entry is used,
    so this model can be trained/evaluated from the same dual-branch
    dataset used for the fusion model (Experiment 3) without a separate
    frequency-only dataset path. Returns the RAW logit, shape (N, 1) - no
    sigmoid, no thresholding (see PROBABILITY_MEANING).
    """

    def __init__(self, num_output_logits: int = 1) -> None:
        super().__init__()
        self.frequency_branch = FrequencyBranch()
        self.classifier = nn.Sequential(
            nn.Dropout(p=CLASSIFIER_DROPOUT),
            nn.Linear(EMBEDDING_DIM, num_output_logits),
        )
        self.num_output_logits = num_output_logits

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        frequency = x["frequency"] if isinstance(x, dict) else x
        embedding = self.frequency_branch(frequency)
        return self.classifier(embedding)

    def get_spec(self) -> Dict[str, Any]:
        """Returns this model's specification, for checkpoint/logging
        provenance - see model/training/checkpoint.py's architecture
        registry, which dispatches on this dict's "architecture" field."""
        spec = FrequencyModelSpec(
            architecture=ARCHITECTURE_ID,
            input_channels=INPUT_CHANNELS,
            conv_channels=list(CONV_CHANNELS),
            embedding_dim=EMBEDDING_DIM,
            num_output_logits=self.num_output_logits,
            probability_meaning=PROBABILITY_MEANING,
        )
        return spec.to_dict()

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Saves only this model's state_dict (no optimizer/training
        state) - mirrors EfficientNetB4Baseline.save_checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
    ) -> "FrequencyOnlyModel":
        model = cls()
        state_dict = torch.load(Path(path), map_location=map_location)
        model.load_state_dict(state_dict)
        return model


def build_frequency_only_model(num_output_logits: int = 1) -> FrequencyOnlyModel:
    """Builds Experiment 2's frequency-only model.

    No `pretrained` option: there is no ImageNet-equivalent pretraining
    source for FFT spectra (see .claude/specs/06-frequency-fusion.md,
    "Training strategy", "Initialization").
    """
    return FrequencyOnlyModel(num_output_logits=num_output_logits)
