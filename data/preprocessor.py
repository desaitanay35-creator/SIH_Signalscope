"""
Image Preprocessor Module.
Standardizes input images for model inference and feature extraction:
safe loading, RGB conversion, resizing, tensor conversion, and normalization.
Responsible Team Member: Member 3 (Data Pipeline & Preprocessing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from config.settings import PreprocessingConfig, load_data_config


class ImageLoadError(ValueError):
    """Raised when an image file is missing, unreadable, or not a valid image."""


class ImagePreprocessor:
    """Preprocesses raw images into normalized, model-ready arrays/tensors."""

    def __init__(
        self,
        image_size: Sequence[int] = (224, 224),
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ) -> None:
        self.target_size = (int(image_size[0]), int(image_size[1]))
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    @classmethod
    def from_config(cls, config: Optional[PreprocessingConfig] = None, config_path: Optional[str] = None) -> "ImagePreprocessor":
        """Builds a preprocessor from config/model_config.yaml (`data.preprocessing`)."""
        if config is None:
            config = load_data_config(config_path).preprocessing
        return cls(
            image_size=config.image_size,
            mean=config.normalization.mean,
            std=config.normalization.std,
        )

    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Loads an image from disk and converts it to RGB.

        Raises ImageLoadError (never a bare PIL/OS exception) for a missing
        file, an unreadable file, or a file that is not a valid image, so
        callers can treat "invalid image" as one well-defined failure mode.
        """
        path = Path(image_path)
        if not path.is_file():
            raise ImageLoadError(f"Image not found: {path}")
        try:
            with Image.open(path) as img:
                img.load()
                return img.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageLoadError(f"Could not decode image at {path}: {exc}") from exc

    def preprocess(self, image_path: Union[str, Path]):
        """Loads, resizes, tensorizes, and normalizes an image for the model.

        Returns a torch.Tensor of shape (3, H, W) if torch is installed,
        otherwise a numpy.ndarray of the same shape/dtype (float32).
        """
        image = self.load_image(image_path)
        image = image.resize(self.target_size, Image.BILINEAR)

        array = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3)
        array = array.transpose(2, 0, 1)  # (3, H, W)
        array = (array - self.mean) / self.std
        array = np.ascontiguousarray(array, dtype=np.float32)

        return self._to_tensor(array)

    @staticmethod
    def _to_tensor(array: np.ndarray):
        try:
            import torch
        except ImportError:
            return array
        return torch.from_numpy(array)
