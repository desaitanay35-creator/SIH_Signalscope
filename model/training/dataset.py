"""
Training/validation dataset construction for the SignalScope training
pipeline.

Thin wrappers around `data.dataset_loader.SignalScopeDataset`: the train
split is given forensics-safe augmentation (via `AugmentedPreprocessor`),
the val split uses the existing deterministic `ImagePreprocessor`
unchanged. Neither wrapper modifies data/dataset_loader.py or
data/preprocessor.py.

Responsible Team Member: Member 3 (Data Pipeline & Preprocessing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

from data.dataset_loader import DatasetRecord, SignalScopeDataset
from data.preprocessor import ImagePreprocessor
from model.training.augmentation import build_train_transform, tensorize


class DuplicateImageError(ValueError):
    """Raised when a manifest contains the same image_path more than once."""


def assert_no_duplicate_images(records: Sequence[DatasetRecord]) -> None:
    """Fails loudly if any `image_path` appears more than once in `records`.

    `data.splitting.split_manifest`'s `check_generator_leakage` only guards
    generator-level leakage; a duplicated row could otherwise place the same
    image in both train and val undetected. Call this once, on the full
    loaded manifest, before splitting.
    """
    seen = set()
    duplicates = set()
    for record in records:
        if record.image_path in seen:
            duplicates.add(record.image_path)
        seen.add(record.image_path)
    if duplicates:
        shown = sorted(duplicates)[:5]
        raise DuplicateImageError(
            f"Manifest contains {len(duplicates)} duplicated image_path value(s): "
            f"{shown}{'...' if len(duplicates) > 5 else ''}"
        )


class AugmentedPreprocessor:
    """Drop-in `.preprocess(image_path)` replacement that applies training
    augmentation before the shared resize/normalize/tensorize step.

    Exposes the same interface `SignalScopeDataset` expects from its
    `preprocessor` argument, so training augmentation can be injected
    without changing data/dataset_loader.py.
    """

    def __init__(self, preprocessor: ImagePreprocessor) -> None:
        self._preprocessor = preprocessor
        self._transform = build_train_transform(preprocessor)

    def preprocess(self, image_path):
        image = self._preprocessor.load_image(image_path)
        image = self._transform(image)
        return tensorize(image, self._preprocessor)


def build_train_dataset(
    records: Sequence[DatasetRecord],
    preprocessor: ImagePreprocessor,
    root_dir: Optional[Union[str, Path]] = None,
) -> SignalScopeDataset:
    """Builds the training dataset: augmentation + shared preprocessing."""
    return SignalScopeDataset(records, preprocessor=AugmentedPreprocessor(preprocessor), root_dir=root_dir)


def build_val_dataset(
    records: Sequence[DatasetRecord],
    preprocessor: ImagePreprocessor,
    root_dir: Optional[Union[str, Path]] = None,
) -> SignalScopeDataset:
    """Builds the validation dataset: deterministic preprocessing only, no
    augmentation - identical to inference-time preprocessing."""
    return SignalScopeDataset(records, preprocessor=preprocessor, root_dir=root_dir)
