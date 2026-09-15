"""
Dataset Manifest & Loader Module.

Represents the SignalScope dataset as a manifest of (image_path, label,
generator, split) records and exposes a PyTorch-compatible Dataset over it.

See data/README.md for the full manifest format, label/generator semantics,
and the generator-disjoint split methodology.

Responsible Team Member: Member 3 (Data Pipeline & Preprocessing)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # torch is an optional runtime dependency for this module
    _TorchDataset = object

REAL_LABEL = 0
AI_GENERATED_LABEL = 1
VALID_LABELS = (REAL_LABEL, AI_GENERATED_LABEL)

# Convention: every record with label == REAL_LABEL must use this generator id.
REAL_GENERATOR_ID = "real"

REQUIRED_COLUMNS = ("image_path", "label", "generator")


class ManifestError(ValueError):
    """Raised when a dataset manifest is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class DatasetRecord:
    """A single manifest row: one image with its label, source, and split."""

    image_path: str
    label: int
    generator: str
    split: Optional[str] = None

    def __post_init__(self) -> None:
        if self.label not in VALID_LABELS:
            raise ManifestError(
                f"Invalid label {self.label!r} for {self.image_path!r}: "
                f"expected one of {VALID_LABELS} (0=real, 1=AI-generated)."
            )
        if not self.generator:
            raise ManifestError(f"Missing generator/source id for {self.image_path!r}.")
        if self.label == REAL_LABEL and self.generator != REAL_GENERATOR_ID:
            raise ManifestError(
                f"Record {self.image_path!r} has label=0 (real) but generator="
                f"{self.generator!r}; real images must use generator="
                f"{REAL_GENERATOR_ID!r} by convention."
            )
        if self.label == AI_GENERATED_LABEL and self.generator == REAL_GENERATOR_ID:
            raise ManifestError(
                f"Record {self.image_path!r} has label=1 (AI-generated) but "
                f"generator={REAL_GENERATOR_ID!r}; AI-generated images must use "
                "a real generator/source id (e.g. 'stable_diffusion')."
            )


def _row_to_record(row: Dict[str, Any], line_no: int) -> DatasetRecord:
    missing = [col for col in REQUIRED_COLUMNS if row.get(col) in (None, "")]
    if missing:
        raise ManifestError(f"Manifest row {line_no} is missing required column(s): {missing}")
    try:
        label = int(row["label"])
    except (TypeError, ValueError) as exc:
        raise ManifestError(
            f"Manifest row {line_no} has non-integer label {row['label']!r}"
        ) from exc
    split = row.get("split") or None
    return DatasetRecord(
        image_path=str(row["image_path"]),
        label=label,
        generator=str(row["generator"]),
        split=str(split) if split else None,
    )


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("data", []))
    if not isinstance(payload, list):
        raise ManifestError(f"JSON manifest at {path} must be a list of records (or a dict with a 'records' key).")
    return payload


def load_manifest(manifest_path: Union[str, Path]) -> List[DatasetRecord]:
    """Loads a dataset manifest (CSV or JSON) into a list of DatasetRecord.

    The manifest is never generated or fabricated by this function - it must
    point at a real, user-supplied dataset manifest. Raises ManifestError on
    any structural problem (missing columns, bad labels, generator/label
    mismatch) so bad metadata fails loudly at load time rather than silently
    corrupting a later split or training run.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {path}. Supply a real manifest path "
            "via config (data.manifest_path) or the manifest_path argument - "
            "see data/README.md."
        )

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _read_csv(path)
    elif suffix == ".json":
        rows = _read_json(path)
    else:
        raise ManifestError(f"Unsupported manifest format {suffix!r}; use .csv or .json.")

    if not rows:
        raise ManifestError(f"Manifest at {path} contains no records.")

    return [_row_to_record(row, idx + 2 if suffix == ".csv" else idx + 1) for idx, row in enumerate(rows)]


def save_manifest(records: Sequence[DatasetRecord], manifest_path: Union[str, Path]) -> None:
    """Writes records back out as a CSV manifest (e.g. after split assignment)."""
    path = Path(manifest_path)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "label", "generator", "split"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": record.image_path,
                    "label": record.label,
                    "generator": record.generator,
                    "split": record.split or "",
                }
            )


def group_by_existing_split(records: Sequence[DatasetRecord]) -> Dict[str, List[DatasetRecord]]:
    """Groups records by their pre-assigned `split` field.

    Use this only for manifests that already encode a trusted split. To
    derive a generator-disjoint split from scratch, use
    `data.splitting.split_manifest` instead.
    """
    groups: Dict[str, List[DatasetRecord]] = {}
    for record in records:
        if not record.split:
            raise ManifestError(
                f"Record {record.image_path!r} has no split assigned; call "
                "data.splitting.split_manifest() to derive one."
            )
        groups.setdefault(record.split, []).append(record)
    return groups


class SignalScopeDataset(_TorchDataset):
    """Dataset over a fixed list of DatasetRecord, producing (tensor, label, meta).

    Compatible with `torch.utils.data.DataLoader` when torch is installed;
    degrades to a plain indexable/len-able object otherwise so this module
    can be imported and unit-tested without a torch dependency.
    """

    def __init__(
        self,
        records: Sequence[DatasetRecord],
        preprocessor: Optional[Any] = None,
        root_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.records: List[DatasetRecord] = list(records)
        self.root_dir = Path(root_dir) if root_dir else None
        self._preprocessor = preprocessor

    @property
    def preprocessor(self) -> Any:
        if self._preprocessor is None:
            from data.preprocessor import ImagePreprocessor

            self._preprocessor = ImagePreprocessor.from_config()
        return self._preprocessor

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, image_path: str) -> Path:
        candidate = Path(image_path)
        if self.root_dir and not candidate.is_absolute():
            return self.root_dir / candidate
        return candidate

    def __getitem__(self, idx: int):
        record = self.records[idx]
        resolved_path = self._resolve_path(record.image_path)
        tensor = self.preprocessor.preprocess(resolved_path)
        metadata = {"generator": record.generator, "image_path": record.image_path}
        return tensor, record.label, metadata

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        split: Optional[str] = None,
        root_dir: Optional[Union[str, Path]] = None,
        preprocessor: Optional[Any] = None,
    ) -> "SignalScopeDataset":
        """Loads records from a manifest, optionally filtered to one split."""
        records = load_manifest(manifest_path)
        if split is not None:
            records = [r for r in records if r.split == split]
            if not records:
                raise ManifestError(f"No records found for split={split!r} in {manifest_path}")
        return cls(records, preprocessor=preprocessor, root_dir=root_dir)

    def generators(self) -> List[str]:
        """Distinct generator/source ids present in this dataset instance."""
        return sorted({r.generator for r in self.records})


def with_split(record: DatasetRecord, split: str) -> DatasetRecord:
    """Returns a copy of `record` with its split field set to `split`."""
    return replace(record, split=split)
