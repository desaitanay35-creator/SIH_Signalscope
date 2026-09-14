"""
Generator-Disjoint Dataset Splitting.

Splits a loaded manifest into train/val/test such that the test split can
represent *unseen-generator* evaluation: generators held out of test never
appear in train or val. See data/README.md for the full methodology.

Responsible Team Member: Member 3 (Data Pipeline & Preprocessing)
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Dict, List, Sequence

from config.settings import SplitConfig
from data.dataset_loader import REAL_GENERATOR_ID, DatasetRecord

TRAIN = "train"
VAL = "val"
TEST = "test"  # holds the unseen-generator evaluation set
ALL_SPLITS = (TRAIN, VAL, TEST)


class GeneratorLeakageError(RuntimeError):
    """Raised when a generator/source appears in both a seen split and the
    unseen-generator test split (or is otherwise split-inconsistent)."""


def _non_real_generators(records: Sequence[DatasetRecord]) -> List[str]:
    return sorted({r.generator for r in records if r.generator != REAL_GENERATOR_ID})


def choose_unseen_generators(records: Sequence[DatasetRecord], config: SplitConfig) -> List[str]:
    """Determines which generators are held out entirely for unseen-generator eval.

    If `config.unseen_generators` is non-empty, it is used verbatim (explicit,
    reviewer-controlled choice). Otherwise, a deterministic seeded shuffle of
    all distinct non-"real" generators picks `held_out_generator_fraction` of
    them - same manifest + same config + same seed always yields the same
    held-out set.
    """
    candidates = _non_real_generators(records)
    if config.unseen_generators:
        explicit = sorted(set(config.unseen_generators))
        unknown = set(explicit) - set(candidates)
        if unknown:
            raise ValueError(
                f"config.unseen_generators contains generators not present in "
                f"the manifest: {sorted(unknown)}"
            )
        return explicit

    if not candidates:
        return []

    rng = random.Random(config.seed)
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    n_holdout = max(1, round(len(shuffled) * config.held_out_generator_fraction))
    n_holdout = min(n_holdout, len(shuffled))
    return sorted(shuffled[:n_holdout])


def split_manifest(records: Sequence[DatasetRecord], config: SplitConfig) -> Dict[str, List[DatasetRecord]]:
    """Splits records into train/val/test with generator-disjoint unseen evaluation.

    - `test` (unseen-generator evaluation) contains real images plus every
      record whose generator is in the held-out set chosen by
      `choose_unseen_generators`. Those generators never appear elsewhere.
    - `train`/`val` contain real images plus records from every remaining
      ("seen") generator, split by `config.val_fraction` using a seeded
      shuffle keyed on `config.seed` for reproducibility.

    Deterministic: the same `records` + `config` (in particular the same
    `config.seed`) always produce the same split.
    """
    unseen_generators = set(choose_unseen_generators(records, config))

    test_records = [r for r in records if r.generator in unseen_generators]
    seen_pool = [r for r in records if r.generator not in unseen_generators]

    rng = random.Random(config.seed)
    shuffled_seen = seen_pool[:]
    rng.shuffle(shuffled_seen)

    n_val = round(len(shuffled_seen) * config.val_fraction)
    val_records = shuffled_seen[:n_val]
    train_records = shuffled_seen[n_val:]

    splits = {
        TRAIN: sorted(train_records, key=lambda r: r.image_path),
        VAL: sorted(val_records, key=lambda r: r.image_path),
        TEST: sorted(test_records, key=lambda r: r.image_path),
    }
    check_generator_leakage(splits)
    return splits


def check_generator_leakage(splits: Dict[str, Sequence[DatasetRecord]]) -> None:
    """Fails loudly if any non-"real" generator appears in both a seen split
    (train/val) and the unseen-generator `test` split.

    This is the explicit safeguard required by the SIH unseen-generator
    methodology: it is not enough to split images at random and call it
    unseen-generator evaluation - this check enforces the generator-group
    boundary is actually respected.
    """
    seen_generators = set(_non_real_generators(list(splits.get(TRAIN, [])) + list(splits.get(VAL, []))))
    test_generators = set(_non_real_generators(list(splits.get(TEST, []))))

    overlap = seen_generators & test_generators
    if overlap:
        raise GeneratorLeakageError(
            f"Generator leakage detected: {sorted(overlap)} present in both "
            "train/val and the unseen-generator test split. A generator must "
            "be assigned to exactly one side of the unseen-generator boundary."
        )


def flatten_with_split(splits: Dict[str, Sequence[DatasetRecord]]) -> List[DatasetRecord]:
    """Flattens a {split_name: [records]} mapping into one list of records
    with each record's `split` field set accordingly (e.g. for save_manifest)."""
    flattened: List[DatasetRecord] = []
    for split_name, records in splits.items():
        for record in records:
            flattened.append(replace(record, split=split_name))
    return flattened
