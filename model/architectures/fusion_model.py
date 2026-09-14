"""
RGB + Frequency fusion model for SignalScope (Step 6, Experiment 3).

Reuses the existing EfficientNet-B4 RGB backbone
(model/architectures/efficientnet_b4.py, unmodified by this step) and the
frequency branch (model/architectures/frequency_branch.py, also
unmodified by this file) via simple concatenation fusion - the simplest
method that is experimentally defensible, per
.claude/specs/06-frequency-fusion.md ("Feature fusion"). Dimensions used
here are the real, inspected values (EfficientNet-B4's pooled feature
width is 1792, verified directly against
model/architectures/efficientnet_b4.py's classifier - not invented).

Responsible Team Member: Member 1 (Core ML & Model Architecture)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Union

import torch
from torch import nn

from model.architectures.efficientnet_b4 import EfficientNetB4Baseline
from model.architectures.frequency_branch import EMBEDDING_DIM as FREQUENCY_EMBEDDING_DIM
from model.architectures.frequency_branch import FrequencyBranch

ARCHITECTURE_ID = "rgb_frequency_fusion"

# EfficientNet-B4's actual pooled feature width (backbone.classifier[-1].in_features),
# inspected directly rather than invented - see .claude/specs/06-frequency-fusion.md
# ("Existing components to reuse").
RGB_EMBEDDING_DIM = 1792
FUSION_DIM = RGB_EMBEDDING_DIM + FREQUENCY_EMBEDDING_DIM
CLASSIFIER_DROPOUT = 0.4

PROBABILITY_MEANING = (
    "sigmoid(logit) = P(AI-generated), combining RGB and frequency-domain "
    "evidence via concatenation fusion. This is a raw model probability, "
    "NOT a calibrated one, and any contribution attributable to the "
    "frequency branch should not be read as confirmed forensic evidence "
    "without the robustness/bias checks described in "
    ".claude/specs/06-frequency-fusion.md (\"How to interpret the "
    "experiment\")."
)


@dataclass(frozen=True)
class FusionModelSpec:
    """Plain-data specification of the fusion model, for
    logging/reporting (mirrors EfficientNetB4Baseline.get_spec())."""

    architecture: str
    rgb_embedding_dim: int
    frequency_embedding_dim: int
    fusion_dim: int
    num_output_logits: int
    rgb_pretrained: bool
    probability_meaning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RGBFrequencyFusionModel(nn.Module):
    """RGB (EfficientNet-B4) + frequency branch, fused by concatenation.

    `forward()` accepts a dict {"rgb": (N, 3, H, W), "frequency": (N, 1, H,
    W)} - matching what model.training.dual_branch_dataset's dual-branch
    preprocessors produce, and what torch's default DataLoader collate
    function batches unchanged (a dict of per-sample tensors collates to a
    dict of batched tensors - no custom collate_fn is needed). Returns the
    RAW logit, shape (N, num_output_logits) - no sigmoid, no thresholding.
    """

    def __init__(self, pretrained: bool = True, num_output_logits: int = 1) -> None:
        super().__init__()
        self.rgb_branch = EfficientNetB4Baseline(pretrained=pretrained, num_output_logits=num_output_logits)
        self.frequency_branch = FrequencyBranch()
        self.classifier = nn.Sequential(
            nn.Dropout(p=CLASSIFIER_DROPOUT),
            nn.Linear(FUSION_DIM, num_output_logits),
        )
        self.pretrained = pretrained
        self.num_output_logits = num_output_logits

    def _rgb_embedding(self, rgb: torch.Tensor) -> torch.Tensor:
        """The 1792-dim pooled RGB embedding, bypassing
        self.rgb_branch.backbone.classifier entirely (that Sequential's
        Dropout+Linear(1792,1) play no role in the fusion model - the
        fusion head above replaces the classifier's role at the fused
        embedding's width). See .claude/specs/06-frequency-fusion.md
        ("RGB branch")."""
        backbone = self.rgb_branch.backbone
        features = backbone.features(rgb)
        pooled = backbone.avgpool(features)
        return torch.flatten(pooled, 1)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        rgb_embedding = self._rgb_embedding(x["rgb"])
        frequency_embedding = self.frequency_branch(x["frequency"])
        fused = torch.cat([rgb_embedding, frequency_embedding], dim=1)
        return self.classifier(fused)

    def get_spec(self) -> Dict[str, Any]:
        """Returns this model's specification, for checkpoint/logging
        provenance - see model/training/checkpoint.py's architecture
        registry, which dispatches on this dict's "architecture" field."""
        spec = FusionModelSpec(
            architecture=ARCHITECTURE_ID,
            rgb_embedding_dim=RGB_EMBEDDING_DIM,
            frequency_embedding_dim=FREQUENCY_EMBEDDING_DIM,
            fusion_dim=FUSION_DIM,
            num_output_logits=self.num_output_logits,
            rgb_pretrained=self.pretrained,
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
        pretrained: bool = False,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "RGBFrequencyFusionModel":
        model = cls(pretrained=pretrained)
        state_dict = torch.load(Path(path), map_location=map_location)
        model.load_state_dict(state_dict)
        return model


def build_fusion_model(pretrained: bool = True, num_output_logits: int = 1) -> RGBFrequencyFusionModel:
    """Builds Experiment 3's RGB+frequency fusion model.

    `pretrained=True` by default, matching Step 4's RGB-only baseline
    default, so the fusion model's RGB half starts from the same
    initialization as Experiment 1 - a deliberate fairness choice. See
    .claude/specs/06-frequency-fusion.md ("RGB branch", "Fair comparison").
    """
    return RGBFrequencyFusionModel(pretrained=pretrained, num_output_logits=num_output_logits)
