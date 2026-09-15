"""
Tiny-GenImage Parquet Ingestion.

Converts a GenImage-format Parquet shard (an `image` struct column with
embedded `bytes`/`path`, plus integer `label` and `generator` columns) into
extracted image files on disk and a SignalScope manifest
(image_path, label, generator, split) as defined in data/dataset_loader.py.

This module does not fabricate, download, or reproduce any dataset - it only
converts a real, user-supplied Parquet shard that must already exist on
disk (see data/README.md, "Tiny-GenImage Parquet ingestion").

Reusable across shards: pass a different `parquet_path` /
`image_output_dir` / `manifest_output_path` / `split` to ingest another
GenImage-format shard (e.g. a held-out unseen-generator shard).

Responsible Team Member: Member 3 (Data Pipeline & Preprocessing)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from data.dataset_loader import (
    AI_GENERATED_LABEL,
    REAL_GENERATOR_ID,
    REAL_LABEL,
    DatasetRecord,
    save_manifest,
)

# Verified GenImage generator id -> SignalScope manifest generator string
# (see data/README.md). Unknown ids are never silently inferred - ingestion
# fails loudly instead.
GENIMAGE_GENERATOR_MAP: Dict[int, str] = {
    0: REAL_GENERATOR_ID,  # "real"
    1: "ADM",
    2: "BigGAN",
    3: "GLIDE",
    4: "Midjourney",
    5: "SD14",
    6: "SD15",
    7: "VQDM",
    8: "Wukong",
}

# Verified GenImage label id -> SignalScope manifest label. Numeric values
# are unchanged; GenImage's own "real"/"fake" terminology is not carried
# into the manifest - label 1 means ai_generated in SignalScope convention.
GENIMAGE_LABEL_MAP: Dict[int, int] = {0: REAL_LABEL, 1: AI_GENERATED_LABEL}

DEFAULT_EXTENSION = ".jpg"
_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

MANIFEST_COLUMNS = ("image_path", "label", "generator", "split")


class IngestionError(ValueError):
    """Raised when a Parquet record cannot be safely/validly ingested.

    Never raised silently in place of skipping a record - every corrupted
    or invalid row aborts ingestion with a clear, specific message.
    """


@dataclass(frozen=True)
class IngestionResult:
    """Summary of one `ingest_parquet_shard` run, for reporting/verification."""

    manifest_path: Path
    image_dir: Path
    num_records: int
    num_images_extracted: int
    real_count: int
    ai_count: int
    generator_counts: Dict[str, int]


def _resolve_label(raw_label: Any, row_index: int) -> int:
    try:
        label_id = int(raw_label)
    except (TypeError, ValueError) as exc:
        raise IngestionError(f"Row {row_index}: label {raw_label!r} is not an integer.") from exc
    if label_id not in GENIMAGE_LABEL_MAP:
        raise IngestionError(
            f"Row {row_index}: unrecognized label id {label_id!r}; "
            f"expected one of {sorted(GENIMAGE_LABEL_MAP)} (0=real, 1=fake)."
        )
    return GENIMAGE_LABEL_MAP[label_id]


def _resolve_generator(raw_generator: Any, row_index: int) -> str:
    try:
        generator_id = int(raw_generator)
    except (TypeError, ValueError) as exc:
        raise IngestionError(f"Row {row_index}: generator id {raw_generator!r} is not an integer.") from exc
    if generator_id not in GENIMAGE_GENERATOR_MAP:
        raise IngestionError(
            f"Row {row_index}: unrecognized generator id {generator_id!r}; "
            f"expected one of {sorted(GENIMAGE_GENERATOR_MAP)}. Not silently inferring."
        )
    return GENIMAGE_GENERATOR_MAP[generator_id]


def _base_filename(original_path: Optional[str], row_index: int) -> str:
    """Preserves the original basename/extension when valid; otherwise falls
    back to a deterministic name with the default extension."""
    if original_path:
        name = Path(str(original_path)).name
        if name:
            ext = Path(name).suffix.lower()
            if ext in _VALID_EXTENSIONS:
                return name
            stem = Path(name).stem or f"record_{row_index:06d}"
            return f"{stem}{DEFAULT_EXTENSION}"
    return f"record_{row_index:06d}{DEFAULT_EXTENSION}"


def _dedupe_filename(base_name: str, used_names: Set[str]) -> str:
    """Returns a deterministic, previously-unused filename.

    Collisions are disambiguated with a numeric suffix (`name.jpg`,
    `name__1.jpg`, `name__2.jpg`, ...) so a duplicate source filename can
    never overwrite a previously extracted image.
    """
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    stem = Path(base_name).stem
    ext = Path(base_name).suffix
    counter = 1
    while True:
        candidate = f"{stem}__{counter}{ext}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def _iter_parquet_rows(parquet_path: Path):
    """Yields one dict per row from a GenImage-format Parquet shard."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    required_columns = {"image", "label", "generator"}
    missing_columns = required_columns - set(table.column_names)
    if missing_columns:
        raise IngestionError(
            f"Parquet shard {parquet_path} is missing required column(s): {sorted(missing_columns)}"
        )
    yield from table.to_pylist()


def ingest_parquet_shard(
    parquet_path: Union[str, Path],
    image_output_dir: Union[str, Path],
    manifest_output_path: Union[str, Path],
    split: str = "train",
) -> IngestionResult:
    """Extracts images and writes a SignalScope manifest from a GenImage shard.

    `image_output_dir` and `manifest_output_path` are used exactly as given -
    pass project-relative paths (e.g. "data/raw/genimage/dev/images") to get
    a project-relative manifest; this function never resolves them to
    absolute paths itself.

    Every record in the shard is stamped with the same `split` value - this
    ingestion path treats one shard as one split (e.g. "train" for a
    development shard). Splitting a single shard's records across
    train/val/unseen-generator-test is a separate step
    (see data/splitting.py) applied to the resulting manifest afterwards.

    Raises IngestionError on the first record with missing/empty image
    bytes, an unrecognized label, or an unrecognized generator id - no
    record is silently skipped.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet shard not found: {parquet_path}")

    image_output_dir = Path(image_output_dir)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_output_path = Path(manifest_output_path)
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)

    used_names: Set[str] = set()
    records: List[DatasetRecord] = []
    real_count = 0
    ai_count = 0
    generator_counts: Dict[str, int] = {}

    for row_index, row in enumerate(_iter_parquet_rows(parquet_path)):
        image_field = row.get("image") or {}
        image_bytes = image_field.get("bytes") if isinstance(image_field, dict) else None
        original_path = image_field.get("path") if isinstance(image_field, dict) else None

        if not image_bytes:
            raise IngestionError(
                f"Row {row_index} (source path={original_path!r}) has missing/empty "
                "image bytes; refusing to silently skip a corrupted record."
            )

        label = _resolve_label(row.get("label"), row_index)
        generator = _resolve_generator(row.get("generator"), row_index)

        base_name = _base_filename(original_path, row_index)
        output_name = _dedupe_filename(base_name, used_names)
        output_path = image_output_dir / output_name
        output_path.write_bytes(image_bytes)

        record = DatasetRecord(
            image_path=str((image_output_dir / output_name).as_posix()),
            label=label,
            generator=generator,
            split=split,
        )
        records.append(record)

        if label == REAL_LABEL:
            real_count += 1
        else:
            ai_count += 1
        generator_counts[generator] = generator_counts.get(generator, 0) + 1

    if not records:
        raise IngestionError(f"Parquet shard {parquet_path} contains no records.")

    save_manifest(records, manifest_output_path)
    verify_ingestion(records, manifest_output_path, split)

    return IngestionResult(
        manifest_path=manifest_output_path,
        image_dir=image_output_dir,
        num_records=len(records),
        num_images_extracted=len(used_names),
        real_count=real_count,
        ai_count=ai_count,
        generator_counts=generator_counts,
    )


def verify_ingestion(records: List[DatasetRecord], manifest_output_path: Path, split: str) -> None:
    """Re-verifies the data-integrity requirements after ingestion.

    Raises IngestionError on the first violation found:
      1. manifest row count == distinct extracted image count
      2. every row has a valid label in {0, 1}
      3. every row has a generator from the verified GenImage mapping
      4. every row has split == the requested split
      5. every referenced image file exists on disk
      6. no two rows resolve to the same image file (no silent overwrite)
      7. the manifest file has exactly the required columns
    """
    if not records:
        raise IngestionError("No records to verify - ingestion produced zero rows.")

    resolved_paths: Set[Path] = set()
    for record in records:
        image_path = Path(record.image_path)
        if not image_path.is_file():
            raise IngestionError(f"Manifest references a missing image file: {image_path}")
        resolved_paths.add(image_path.resolve())

    if len(resolved_paths) != len(records):
        raise IngestionError(
            f"Manifest row count ({len(records)}) does not match the number of "
            f"distinct extracted image files ({len(resolved_paths)}); a duplicate "
            "source filename may have overwritten a previously extracted image."
        )

    invalid_labels = [r for r in records if r.label not in (REAL_LABEL, AI_GENERATED_LABEL)]
    if invalid_labels:
        raise IngestionError(f"{len(invalid_labels)} record(s) have an invalid label (expected 0 or 1).")

    known_generators = set(GENIMAGE_GENERATOR_MAP.values())
    invalid_generators = [r for r in records if r.generator not in known_generators]
    if invalid_generators:
        raise IngestionError(
            f"{len(invalid_generators)} record(s) have a generator outside the "
            f"verified GenImage mapping: {sorted({r.generator for r in invalid_generators})}"
        )

    wrong_split = [r for r in records if r.split != split]
    if wrong_split:
        raise IngestionError(f"{len(wrong_split)} record(s) do not have split={split!r}.")

    with open(manifest_output_path, "r", encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    if tuple(header) != MANIFEST_COLUMNS:
        raise IngestionError(f"Manifest {manifest_output_path} has columns {tuple(header)}, expected {MANIFEST_COLUMNS}.")


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest a GenImage-format Parquet shard into extracted images + a SignalScope manifest."
    )
    parser.add_argument("--parquet", required=True, help="Path to the .parquet shard.")
    parser.add_argument("--image-dir", required=True, help="Output directory for extracted images.")
    parser.add_argument("--manifest", required=True, help="Output manifest CSV path.")
    parser.add_argument("--split", default="train", help="Split value stamped onto every row (default: train).")
    return parser


def main() -> None:
    """CLI entrypoint: `python -m data.genimage_ingest --parquet ... --image-dir ... --manifest ...`."""
    args = _build_arg_parser().parse_args()
    result = ingest_parquet_shard(
        parquet_path=args.parquet,
        image_output_dir=args.image_dir,
        manifest_output_path=args.manifest,
        split=args.split,
    )
    print(f"manifest_path        = {result.manifest_path}")
    print(f"image_dir             = {result.image_dir}")
    print(f"num_records            = {result.num_records}")
    print(f"num_images_extracted   = {result.num_images_extracted}")
    print(f"real_count             = {result.real_count}")
    print(f"ai_count               = {result.ai_count}")
    print(f"generator_counts       = {result.generator_counts}")


if __name__ == "__main__":
    main()
