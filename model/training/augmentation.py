"""
Training-time image augmentation for the SignalScope training pipeline.

Conservative, forensics-safe augmentations only: horizontal flip, small
rotation, and a high-minimum-scale random crop-resize. Heavy augmentations
(color jitter, blur/noise, JPEG re-compression, cutout/random-erasing,
mixup/cutmix) are deliberately excluded - they either destroy or fabricate
the compression/frequency artifacts that are the actual detection signal.
See .claude/specs/04-training-pipeline.md ("Augmentation strategy").

Responsible Team Member: Member 3 (Data Pipeline & Preprocessing)
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data.preprocessor import ImagePreprocessor

# Conservative bounds, chosen to preserve global frequency/compression
# artifacts rather than to maximize augmentation diversity.
ROTATION_DEGREES = 10
CROP_MIN_SCALE = 0.9


def build_train_transform(preprocessor: ImagePreprocessor) -> transforms.Compose:
    """Returns a PIL-image-to-PIL-image randomized augmentation pipeline.

    Resizes to `preprocessor.target_size` as part of the crop step, so the
    output is ready for `tensorize()` with no further resizing.
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=ROTATION_DEGREES),
            transforms.RandomResizedCrop(
                size=preprocessor.target_size,
                scale=(CROP_MIN_SCALE, 1.0),
            ),
        ]
    )


def tensorize(image: Image.Image, preprocessor: ImagePreprocessor) -> torch.Tensor:
    """Converts an already-resized RGB PIL image into a normalized tensor.

    Mirrors `ImagePreprocessor.preprocess()`'s tensorize/normalize step
    exactly, reading `mean`/`std` from `preprocessor` (the single source of
    truth) so training and inference normalization can never drift apart.
    This duplicates a few lines rather than modifying data/preprocessor.py,
    since augmentation must run on the PIL image before normalization and
    data/ is out of scope for this step.
    """
    array = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3)
    array = array.transpose(2, 0, 1)  # (3, H, W)
    array = (array - preprocessor.mean) / preprocessor.std
    array = np.ascontiguousarray(array, dtype=np.float32)
    return torch.from_numpy(array)
