"""
Unit Tests for Tiny-GenImage Parquet Ingestion.

Covers label/generator mapping, image extraction, duplicate-filename
handling, and the data-integrity checks in data/genimage_ingest.py.
All fixtures are tiny synthetic Parquet shards built on the fly (a handful
of small generated PNGs) - never the real 477MB development shard.
Responsible Team Member: Member 6 (MLOps & Testing)
"""

import csv
import io

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("PIL")
from PIL import Image as PILImage

from data.genimage_ingest import (
    GENIMAGE_GENERATOR_MAP,
    GENIMAGE_LABEL_MAP,
    IngestionError,
    MANIFEST_COLUMNS,
    ingest_parquet_shard,
)


def _png_bytes(color) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_shard(tmp_path, rows, name="shard.parquet"):
    """rows: list of dicts with keys bytes, path, label, generator."""
    image_struct = pa.array(
        [{"bytes": r["bytes"], "path": r["path"]} for r in rows],
        type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
    )
    table = pa.table(
        {
            "image": image_struct,
            "label": pa.array([r["label"] for r in rows], type=pa.int64()),
            "generator": pa.array([r["generator"] for r in rows], type=pa.int64()),
        }
    )
    path = tmp_path / name
    pq.write_table(table, path)
    return path


@pytest.fixture
def valid_rows():
    return [
        {"bytes": _png_bytes((10, 10, 10)), "path": "real_001.jpg", "label": 0, "generator": 0},
        {"bytes": _png_bytes((20, 20, 20)), "path": "real_002.jpg", "label": 0, "generator": 0},
        {"bytes": _png_bytes((200, 0, 0)), "path": "adm_001.png", "label": 1, "generator": 1},
        {"bytes": _png_bytes((0, 200, 0)), "path": "mj_001.png", "label": 1, "generator": 4},
        {"bytes": _png_bytes((0, 0, 200)), "path": "wukong_001.png", "label": 1, "generator": 8},
    ]


def _ingest(tmp_path, rows, split="train", name="shard.parquet"):
    shard_path = _write_shard(tmp_path, rows, name=name)
    image_dir = tmp_path / "images"
    manifest_path = tmp_path / "manifest.csv"
    result = ingest_parquet_shard(shard_path, image_dir, manifest_path, split=split)
    return result, image_dir, manifest_path


def _read_manifest_rows(manifest_path):
    with open(manifest_path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Valid ingestion end-to-end
# ---------------------------------------------------------------------------


def test_valid_parquet_ingestion_produces_expected_counts(tmp_path, valid_rows):
    result, image_dir, manifest_path = _ingest(tmp_path, valid_rows)

    assert result.num_records == 5
    assert result.num_images_extracted == 5
    assert result.real_count == 2
    assert result.ai_count == 3
    assert result.generator_counts == {"real": 2, "ADM": 1, "Midjourney": 1, "Wukong": 1}


def test_manifest_has_required_columns(tmp_path, valid_rows):
    _, _, manifest_path = _ingest(tmp_path, valid_rows)
    with open(manifest_path, "r", encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    assert tuple(header) == MANIFEST_COLUMNS


def test_split_value_stamped_on_every_row(tmp_path, valid_rows):
    _, _, manifest_path = _ingest(tmp_path, valid_rows, split="train")
    rows = _read_manifest_rows(manifest_path)
    assert len(rows) == 5
    assert all(row["split"] == "train" for row in rows)


def test_label_mapping_preserves_numeric_values(tmp_path, valid_rows):
    _, _, manifest_path = _ingest(tmp_path, valid_rows)
    rows = _read_manifest_rows(manifest_path)
    by_generator = {row["generator"]: int(row["label"]) for row in rows}
    assert by_generator["real"] == GENIMAGE_LABEL_MAP[0] == 0
    assert by_generator["ADM"] == GENIMAGE_LABEL_MAP[1] == 1
    assert by_generator["Midjourney"] == 1


def test_generator_mapping_matches_verified_table(tmp_path):
    rows = [
        {"bytes": _png_bytes((i, i, i)), "path": f"g{i}.png", "label": 0 if i == 0 else 1, "generator": i}
        for i in range(9)
    ]
    _, _, manifest_path = _ingest(tmp_path, rows)
    manifest_rows = _read_manifest_rows(manifest_path)
    manifest_generators = {row["generator"] for row in manifest_rows}
    assert manifest_generators == set(GENIMAGE_GENERATOR_MAP.values())
    assert manifest_generators == {
        "real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong",
    }


def test_image_extraction_writes_correct_bytes(tmp_path, valid_rows):
    from pathlib import Path

    _, image_dir, manifest_path = _ingest(tmp_path, valid_rows)
    rows = _read_manifest_rows(manifest_path)

    # manifest rows preserve shard order, so they line up with valid_rows.
    for row, source in zip(rows, valid_rows):
        extracted_path = Path(row["image_path"])
        assert extracted_path.is_file()
        assert extracted_path.read_bytes() == source["bytes"]


# ---------------------------------------------------------------------------
# Rejections: missing bytes, invalid label, invalid generator
# ---------------------------------------------------------------------------


def test_missing_image_bytes_raises(tmp_path):
    rows = [{"bytes": None, "path": "broken.jpg", "label": 0, "generator": 0}]
    with pytest.raises(IngestionError):
        _ingest(tmp_path, rows)


def test_empty_image_bytes_raises(tmp_path):
    rows = [{"bytes": b"", "path": "empty.jpg", "label": 0, "generator": 0}]
    with pytest.raises(IngestionError):
        _ingest(tmp_path, rows)


def test_invalid_label_raises(tmp_path):
    rows = [{"bytes": _png_bytes((1, 1, 1)), "path": "bad_label.jpg", "label": 7, "generator": 0}]
    with pytest.raises(IngestionError):
        _ingest(tmp_path, rows)


def test_invalid_generator_id_raises(tmp_path):
    rows = [{"bytes": _png_bytes((1, 1, 1)), "path": "bad_gen.jpg", "label": 1, "generator": 99}]
    with pytest.raises(IngestionError):
        _ingest(tmp_path, rows)


# ---------------------------------------------------------------------------
# Duplicate filenames & manifest/image count consistency
# ---------------------------------------------------------------------------


def test_duplicate_source_filenames_do_not_overwrite(tmp_path):
    rows = [
        {"bytes": _png_bytes((255, 0, 0)), "path": "dup.jpg", "label": 1, "generator": 1},
        {"bytes": _png_bytes((0, 255, 0)), "path": "dup.jpg", "label": 1, "generator": 2},
    ]
    result, image_dir, manifest_path = _ingest(tmp_path, rows)

    manifest_rows = _read_manifest_rows(manifest_path)
    image_paths = {row["image_path"] for row in manifest_rows}

    assert result.num_records == 2
    assert result.num_images_extracted == 2
    assert len(image_paths) == 2  # two distinct files, no overwrite

    from pathlib import Path

    contents = {Path(p).read_bytes() for p in image_paths}
    assert contents == {_png_bytes((255, 0, 0)), _png_bytes((0, 255, 0))}


def test_manifest_row_count_matches_extracted_image_count(tmp_path, valid_rows):
    result, image_dir, _ = _ingest(tmp_path, valid_rows)
    extracted_files = list(image_dir.iterdir())
    assert result.num_records == len(extracted_files) == result.num_images_extracted


# ---------------------------------------------------------------------------
# Determinism across repeated runs
# ---------------------------------------------------------------------------


def test_rerunning_ingestion_is_deterministic(tmp_path, valid_rows):
    shard_path = _write_shard(tmp_path, valid_rows)
    image_dir = tmp_path / "images"
    manifest_path = tmp_path / "manifest.csv"

    result_1 = ingest_parquet_shard(shard_path, image_dir, manifest_path, split="train")
    rows_1 = _read_manifest_rows(manifest_path)

    result_2 = ingest_parquet_shard(shard_path, image_dir, manifest_path, split="train")
    rows_2 = _read_manifest_rows(manifest_path)

    assert result_1.num_records == result_2.num_records == 5
    assert rows_1 == rows_2  # identical manifest content, not duplicated/appended
    assert len(list(image_dir.iterdir())) == 5  # no orphaned/duplicated files
