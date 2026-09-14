"""
Centralized configuration settings for SignalScope.
Handles environment variables, default paths, and system parameters.
Responsible Team Member: Member 6 (MLOps & Config)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

@dataclass
class Settings:
    PROJECT_NAME: str = "SignalScope"
    VERSION: str = "1.0.0"
    MODEL_WEIGHTS_PATH: str = os.getenv("MODEL_WEIGHTS_PATH", "weights/signalscope_v1.pth")
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    HOST: str = "0.0.0.0"
    PORT: int = 8000

settings = Settings()

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "model_config.yaml"


@dataclass
class NormalizationConfig:
    """Per-channel normalization statistics applied after tensor conversion."""

    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class PreprocessingConfig:
    """Image preprocessing parameters shared by dataset loading and inference."""

    image_size: List[int] = field(default_factory=lambda: [224, 224])
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


@dataclass
class SplitConfig:
    """Controls how a manifest is partitioned into train/val/unseen-test."""

    seed: int = 42
    val_fraction: float = 0.15
    unseen_generators: List[str] = field(default_factory=list)
    held_out_generator_fraction: float = 0.3


@dataclass
class DataConfig:
    """Dataset ingestion configuration, loaded from config/model_config.yaml."""

    manifest_path: str = "data/manifest.csv"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


def load_data_config(config_path: Optional[str] = None) -> DataConfig:
    """Loads the `data:` section of the YAML model configuration.

    Falls back to `DataConfig()` defaults for any missing keys, so a partial
    or absent config file still produces a usable, fully-typed configuration.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        return DataConfig()

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    data_raw = raw.get("data", {}) or {}
    preprocessing_raw = data_raw.get("preprocessing", {}) or {}
    normalization_raw = preprocessing_raw.get("normalization", {}) or {}
    split_raw = data_raw.get("split", {}) or {}

    defaults = DataConfig()

    normalization = NormalizationConfig(
        mean=normalization_raw.get("mean", defaults.preprocessing.normalization.mean),
        std=normalization_raw.get("std", defaults.preprocessing.normalization.std),
    )
    preprocessing = PreprocessingConfig(
        image_size=preprocessing_raw.get("image_size", defaults.preprocessing.image_size),
        normalization=normalization,
    )
    split = SplitConfig(
        seed=split_raw.get("seed", defaults.split.seed),
        val_fraction=split_raw.get("val_fraction", defaults.split.val_fraction),
        unseen_generators=split_raw.get("unseen_generators", list(defaults.split.unseen_generators)),
        held_out_generator_fraction=split_raw.get(
            "held_out_generator_fraction", defaults.split.held_out_generator_fraction
        ),
    )
    return DataConfig(
        manifest_path=data_raw.get("manifest_path", defaults.manifest_path),
        preprocessing=preprocessing,
        split=split,
    )


@dataclass
class ModelConfig:
    """Baseline model architecture configuration, loaded from the `model:`
    section of config/model_config.yaml.

    Input size, channel order, and normalization are intentionally NOT part
    of this dataclass - they are owned by `DataConfig.preprocessing` (single
    source of truth shared with data/preprocessor.py). See
    model/architectures/efficientnet_b4.py::get_model_spec for how the two
    are combined into one reported specification.
    """

    name: str = "EfficientNet-B4"
    architecture: str = "efficientnet_b4"
    pretrained: bool = True
    num_output_logits: int = 1
    label_mapping: Dict[int, str] = field(default_factory=lambda: {0: "real", 1: "ai_generated"})
    threshold: float = 0.5


def load_model_config(config_path: Optional[str] = None) -> ModelConfig:
    """Loads the `model:` section of the YAML model configuration.

    Falls back to `ModelConfig()` defaults for any missing keys.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        return ModelConfig()

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    model_raw = raw.get("model", {}) or {}
    defaults = ModelConfig()

    label_mapping_raw = model_raw.get("label_mapping", defaults.label_mapping)
    label_mapping = {int(k): str(v) for k, v in label_mapping_raw.items()}

    return ModelConfig(
        name=model_raw.get("name", defaults.name),
        architecture=model_raw.get("architecture", defaults.architecture),
        pretrained=bool(model_raw.get("pretrained", defaults.pretrained)),
        num_output_logits=int(model_raw.get("num_output_logits", defaults.num_output_logits)),
        label_mapping=label_mapping,
        threshold=float(model_raw.get("threshold", defaults.threshold)),
    )
