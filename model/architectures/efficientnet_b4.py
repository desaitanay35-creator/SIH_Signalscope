"""
EfficientNet-B4 Baseline Classifier.

Baseline architecture for SignalScope's real-vs-AI-generated image
classification task: a torchvision EfficientNet-B4 backbone (optionally
ImageNet-pretrained) with its 1000-class ImageNet head replaced by a single
binary logit.

This module defines architecture only - no training loop, no calibration,
no thresholding, and no automatic inference on import. See
data/README.md ("EfficientNet-B4 baseline") for the full rationale and
data/preprocessor.py for the actual image preprocessing pipeline (this
module does not reimplement it).

Responsible Team Member: Member 1 (Core ML & Model Architecture)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

from config.settings import load_data_config, load_model_config

# Fixed architectural facts (not configuration - changing these means a
# different architecture, so they are not YAML-overridable).
CHANNEL_ORDER = "RGB"
PROBABILITY_MEANING = (
    "sigmoid(logit) = estimated probability that the input image is "
    "AI-generated (label 1). This is a raw model probability, NOT a "
    "calibrated one - see data/README.md."
)
BASELINE_NOTES = (
    "Baseline architecture only: ImageNet-pretrained EfficientNet-B4 backbone "
    "with a freshly-initialized single-logit binary head. The head has not "
    "been trained on any real-vs-AI-generated data yet, so its raw output is "
    "not yet meaningful for prediction. No decision threshold is applied "
    "inside the model - thresholding belongs to the evaluation/inference "
    "layer, applied to sigmoid(logit)."
)


@dataclass(frozen=True)
class ModelSpec:
    """Plain-data specification of the baseline model, for logging/reporting."""

    name: str
    architecture: str
    pretrained: bool
    num_output_logits: int
    input_size: List[int]
    channel_order: str
    normalization_mean: List[float]
    normalization_std: List[float]
    label_mapping: Dict[int, str]
    probability_meaning: str
    notes: str

    def to_dict(self) -> Dict:
        return asdict(self)


def get_model_spec(pretrained: Optional[bool] = None, config_path: Optional[str] = None) -> Dict:
    """Returns the baseline model specification as a plain dict.

    Combines `model:` (name/architecture/pretrained/label mapping) and
    `data.preprocessing` (input size/normalization - the single source of
    truth shared with data/preprocessor.py) from config/model_config.yaml.
    """
    model_cfg = load_model_config(config_path)
    data_cfg = load_data_config(config_path)

    resolved_pretrained = model_cfg.pretrained if pretrained is None else pretrained

    spec = ModelSpec(
        name=model_cfg.name,
        architecture=model_cfg.architecture,
        pretrained=resolved_pretrained,
        num_output_logits=model_cfg.num_output_logits,
        input_size=list(data_cfg.preprocessing.image_size),
        channel_order=CHANNEL_ORDER,
        normalization_mean=list(data_cfg.preprocessing.normalization.mean),
        normalization_std=list(data_cfg.preprocessing.normalization.std),
        label_mapping=dict(model_cfg.label_mapping),
        probability_meaning=PROBABILITY_MEANING,
        notes=BASELINE_NOTES,
    )
    return spec.to_dict()


class EfficientNetB4Baseline(nn.Module):
    """EfficientNet-B4 backbone with a single-logit binary classification head.

    Forward pass returns the RAW logit of shape (N, 1) - no sigmoid, no
    thresholding. Apply `torch.sigmoid(logit)` externally to obtain
    P(AI-generated); apply a decision threshold in the evaluation/inference
    layer, not here.
    """

    def __init__(self, pretrained: bool = True, num_output_logits: int = 1) -> None:
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)

        # Replace the 1000-class ImageNet head with a single binary logit.
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_output_logits)

        self.backbone = backbone
        self.pretrained = pretrained
        self.num_output_logits = num_output_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (N, num_output_logits)."""
        return self.backbone(x)

    def get_spec(self) -> Dict:
        """Returns this instance's model specification (see get_model_spec)."""
        spec = get_model_spec(pretrained=self.pretrained)
        spec["num_output_logits"] = self.num_output_logits
        return spec

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Saves only the model's state_dict (no optimizer/training state)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        pretrained: bool = False,
        map_location: Union[str, torch.device] = "cpu",
    ) -> "EfficientNetB4Baseline":
        """Builds a model and loads a previously saved state_dict into it.

        `pretrained` controls only the initial backbone construction before
        the checkpoint is applied - it should normally be False here since
        the checkpoint's weights fully determine the loaded parameters.
        """
        model = cls(pretrained=pretrained)
        state_dict = torch.load(Path(path), map_location=map_location)
        model.load_state_dict(state_dict)
        return model


def build_model(pretrained: bool = True) -> nn.Module:
    """Builds the SignalScope baseline EfficientNet-B4 binary classifier.

    Args:
        pretrained: If True, loads torchvision's ImageNet-pretrained
            EfficientNet-B4 weights (requires network access on first call
            to download them). Set to False in tests/CI to guarantee no
            network access or weight download occurs.

    Returns:
        An `EfficientNetB4Baseline` module. No training, calibration, or
        thresholding is performed - this returns raw architecture only.
    """
    return EfficientNetB4Baseline(pretrained=pretrained)
