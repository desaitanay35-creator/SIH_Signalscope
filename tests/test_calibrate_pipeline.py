"""
Integration tests for the SignalScope Step 8A calibration pipeline
(model/calibration/calibrate.py).

Covers end-to-end artifact generation, checkpoint immutability, the
correctness invariants (ROC-AUC rank-invariance, threshold-0.5 decision
invariance), reproducibility, the primary rgb_frequency_fusion pipeline,
and the non-blocking secondary architecture compatibility check. All
against tiny synthetic fixtures (CPU-only, pretrained=False) - never
touches the real GenImage shard or any hidden SIH test set.
"""

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from model.architectures.efficientnet_b4 import EfficientNetB4Baseline
from model.architectures.fusion_model import RGBFrequencyFusionModel
from model.calibration.calibrate import (
    CheckpointModifiedError,
    run_calibration,
    run_secondary_compatibility_check,
)
from model.training.checkpoint import save_checkpoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_image(path, size=(16, 16), color=(120, 40, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def synthetic_manifest(tmp_path):
    """20 real, 10 stable_diffusion (seen), 8 midjourney (unseen). Enough
    real images land in val (val_fraction=0.3, seed=42) to pair against
    the unseen-generator fakes, per the same real-image-pairing protocol
    Step 5/7 use."""
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    rows = []
    for i in range(20):
        path = _write_image(real_dir / f"real_{i:02d}.jpg", color=((i * 5) % 255, (i * 5) % 255, (i * 5) % 255))
        rows.append({"image_path": str(path), "label": 0, "generator": "real"})
    for i in range(10):
        path = _write_image(fake_dir / f"sd_{i:02d}.jpg", color=(200, (i * 10) % 255, 0))
        rows.append({"image_path": str(path), "label": 1, "generator": "stable_diffusion"})
    for i in range(8):
        path = _write_image(fake_dir / f"mj_{i:02d}.jpg", color=(0, 200, (i * 10) % 255))
        rows.append({"image_path": str(path), "label": 1, "generator": "midjourney"})

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "label", "generator", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "split": ""})
    return manifest_path


def _model_config_yaml(tmp_path, manifest_path, unseen_generators, name="model_config.yaml"):
    config_path = tmp_path / name
    unseen_yaml = "[" + ", ".join(f'"{g}"' for g in unseen_generators) + "]"
    config_path.write_text(
        f"""
model:
  name: "EfficientNet-B4"
  architecture: "efficientnet_b4"
  pretrained: false
  num_output_logits: 1
  label_mapping:
    0: "real"
    1: "ai_generated"
  threshold: 0.5
data:
  manifest_path: "{manifest_path.as_posix()}"
  preprocessing:
    image_size: [16, 16]
    normalization:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  split:
    seed: 42
    val_fraction: 0.3
    unseen_generators: {unseen_yaml}
    held_out_generator_fraction: 0.3
""",
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def fusion_checkpoint_path(tmp_path):
    model = RGBFrequencyFusionModel(pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp_path / "fusion_checkpoints" / "best.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        epoch=1,
        best_val_roc_auc=0.6,
        val_metrics_history=[],
        training_config={"epochs": 1},
        seed=42,
    )
    return path


@pytest.fixture
def rgb_only_checkpoint_path(tmp_path):
    model = EfficientNetB4Baseline(pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp_path / "rgb_checkpoints" / "best.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        epoch=1,
        best_val_roc_auc=0.6,
        val_metrics_history=[],
        training_config={"epochs": 1},
        seed=42,
    )
    return path


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Primary experiment: rgb_frequency_fusion checkpoint (Step 7 architecture)
# ---------------------------------------------------------------------------


def test_run_calibration_produces_expected_artifact_bundle_for_fusion_checkpoint(
    tmp_path, synthetic_manifest, fusion_checkpoint_path
):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest, ["midjourney"])

    summary = run_calibration(
        checkpoint_path=str(fusion_checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "calib_output"),
    )

    run_dir = Path(summary["run_dir"])
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "temperature.json").is_file()
    assert (run_dir / "metrics_before.json").is_file()
    assert (run_dir / "metrics_after.json").is_file()
    assert (run_dir / "calibration_report.md").is_file()
    # Nothing is written outside the calibration output directory for this run.
    assert run_dir.parent == (tmp_path / "calib_output")

    assert summary["temperature"] > 0.0
    assert summary["checkpoint_unmodified"] is True

    with open(run_dir / "config.json", encoding="utf-8") as fh:
        config = json.load(fh)
    assert config["checkpoint_architecture"] == "rgb_frequency_fusion"
    assert config["calibration_method"] == "temperature_scaling"
    assert config["checkpoint_unmodified"] is True


def test_run_calibration_does_not_modify_checkpoint_file(tmp_path, synthetic_manifest, fusion_checkpoint_path):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest, ["midjourney"])
    checksum_before = _sha256(fusion_checkpoint_path)

    run_calibration(
        checkpoint_path=str(fusion_checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "calib_output_2"),
    )

    checksum_after = _sha256(fusion_checkpoint_path)
    assert checksum_before == checksum_after


def test_run_calibration_raises_clearly_for_missing_checkpoint(tmp_path, synthetic_manifest):
    with pytest.raises(FileNotFoundError):
        run_calibration(
            checkpoint_path=str(tmp_path / "does_not_exist.pt"),
            manifest_path=str(synthetic_manifest),
        )


def test_run_calibration_fits_temperature_only_on_validation_sample_count(
    tmp_path, synthetic_manifest, fusion_checkpoint_path
):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest, ["midjourney"])

    summary = run_calibration(
        checkpoint_path=str(fusion_checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "calib_output_3"),
    )
    run_dir = Path(summary["run_dir"])

    with open(run_dir / "temperature.json", encoding="utf-8") as fh:
        temperature_info = json.load(fh)
    with open(run_dir / "config.json", encoding="utf-8") as fh:
        config = json.load(fh)

    assert temperature_info["num_fit_samples"] == config["num_val_samples"]
    # Fit sample count must never equal or derive from the unseen-generator
    # (test) split's own sample count in a way that implies test leakage.
    assert temperature_info["num_fit_samples"] != config["num_unseen_generator_eval_samples"]


# ---------------------------------------------------------------------------
# Correctness invariants, verified against persisted artifacts
# ---------------------------------------------------------------------------


def test_run_calibration_roc_auc_is_invariant_within_tolerance(tmp_path, synthetic_manifest, fusion_checkpoint_path):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest, ["midjourney"])

    summary = run_calibration(
        checkpoint_path=str(fusion_checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "calib_output_4"),
    )
    run_dir = Path(summary["run_dir"])

    with open(run_dir / "metrics_before.json", encoding="utf-8") as fh:
        before = json.load(fh)
    with open(run_dir / "metrics_after.json", encoding="utf-8") as fh:
        after = json.load(fh)

    val_roc_before = before["val"]["overall"]["roc_auc"]
    val_roc_after = after["val"]["overall"]["roc_auc"]
    unseen_roc_before = before["unseen_generator"]["overall"]["roc_auc"]
    unseen_roc_after = after["unseen_generator"]["overall"]["roc_auc"]

    if val_roc_before is not None:
        assert val_roc_after is not None
        assert val_roc_before == pytest.approx(val_roc_after, abs=1e-6)
    if unseen_roc_before is not None:
        assert unseen_roc_after is not None
        assert unseen_roc_before == pytest.approx(unseen_roc_after, abs=1e-6)

    # run_calibration itself already asserts this internally (would have
    # raised RankInvarianceViolationError otherwise) - this test confirms
    # the persisted artifacts agree with what was enforced at run time.
    assert summary["val_roc_auc"]["before"] == val_roc_before
    assert summary["val_roc_auc"]["after"] == val_roc_after
    assert summary["unseen_generator_roc_auc"]["before"] == unseen_roc_before
    assert summary["unseen_generator_roc_auc"]["after"] == unseen_roc_after


def test_run_calibration_threshold_0_5_predictions_are_identical_before_and_after(
    tmp_path, synthetic_manifest, fusion_checkpoint_path
):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest, ["midjourney"])

    summary = run_calibration(
        checkpoint_path=str(fusion_checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "calib_output_5"),
    )
    run_dir = Path(summary["run_dir"])

    with open(run_dir / "metrics_before.json", encoding="utf-8") as fh:
        before = json.load(fh)
    with open(run_dir / "metrics_after.json", encoding="utf-8") as fh:
        after = json.load(fh)

    for split in ("val", "unseen_generator"):
        block_before = before[split]["overall"]
        block_after = after[split]["overall"]
        assert block_before["accuracy"] == pytest.approx(block_after["accuracy"])
        assert block_before["macro_f1"] == pytest.approx(block_after["macro_f1"])
        assert block_before["fpr"] == block_after["fpr"]
        assert block_before["confusion_matrix"] == block_after["confusion_matrix"]

    assert summary["threshold_predictions_identical"]["val"] is True
    assert summary["threshold_predictions_identical"]["unseen_generator"] is True


def test_run_calibration_reports_brier_and_ece_for_both_splits(tmp_path, synthetic_manifest, fusion_checkpoint_path):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest, ["midjourney"])

    summary = run_calibration(
        checkpoint_path=str(fusion_checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "calib_output_6"),
    )

    for block in (summary["brier_before"], summary["brier_after"], summary["ece_before"], summary["ece_after"]):
        assert "val" in block and "unseen_generator" in block
        assert 0.0 <= block["val"] <= 1.0
        assert 0.0 <= block["unseen_generator"] <= 1.0


def test_run_calibration_is_reproducible_given_identical_inputs(tmp_path, synthetic_manifest, fusion_checkpoint_path):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest, ["midjourney"])

    summary_a = run_calibration(
        checkpoint_path=str(fusion_checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "calib_output_a"),
    )
    summary_b = run_calibration(
        checkpoint_path=str(fusion_checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "calib_output_b"),
    )

    assert summary_a["temperature"] == pytest.approx(summary_b["temperature"], rel=1e-9)
    assert summary_a["val_roc_auc"] == summary_b["val_roc_auc"]
    assert summary_a["unseen_generator_roc_auc"] == summary_b["unseen_generator_roc_auc"]


# ---------------------------------------------------------------------------
# Report content: dev-shard disclaimer, real-image-pairing note, distinctions
# ---------------------------------------------------------------------------


def test_calibration_report_states_required_disclaimers(tmp_path, synthetic_manifest, fusion_checkpoint_path):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest, ["midjourney"])

    summary = run_calibration(
        checkpoint_path=str(fusion_checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "calib_output_report"),
    )
    report_text = (Path(summary["run_dir"]) / "calibration_report.md").read_text(encoding="utf-8")

    assert "not the final SIH benchmark" in report_text
    assert "does not change the underlying" in report_text or "does not change" in report_text
    assert "rank-invariant" in report_text or "monotonic" in report_text
    assert "not an improvement in AI-generated-image detection" in report_text
    assert "not claimed to be universally reliable" in report_text
    assert "Real images for this unseen-generator result come from the validation split" in report_text


def test_calibration_report_includes_dev_shard_banner_for_genimage_dev_manifest(tmp_path, synthetic_manifest):
    dev_dir = tmp_path / "genimage_dev_copy"
    dev_dir.mkdir()
    dev_manifest_path = dev_dir / "genimage_dev.csv"
    with open(synthetic_manifest, encoding="utf-8", newline="") as src, open(
        dev_manifest_path, "w", encoding="utf-8", newline=""
    ) as dst:
        dst.write(src.read())

    model = RGBFrequencyFusionModel(pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler=None,
        epoch=1,
        best_val_roc_auc=0.6,
        val_metrics_history=[],
        training_config={"epochs": 1},
        seed=42,
    )

    model_config_path = _model_config_yaml(tmp_path, dev_manifest_path, ["midjourney"])
    summary = run_calibration(
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(dev_manifest_path),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "calib_output_dev"),
    )
    report_text = (Path(summary["run_dir"]) / "calibration_report.md").read_text(encoding="utf-8")
    assert "NOT the final SIH benchmark" in report_text


# ---------------------------------------------------------------------------
# Secondary architecture compatibility - non-blocking
# ---------------------------------------------------------------------------


def test_secondary_compatibility_check_succeeds_for_rgb_only_checkpoint(
    tmp_path, synthetic_manifest, rgb_only_checkpoint_path
):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest, ["midjourney"])

    result = run_secondary_compatibility_check(
        checkpoint_path=str(rgb_only_checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "calib_output_secondary"),
    )
    assert result["status"] == "ok"
    assert result["summary"]["temperature"] > 0.0


def test_secondary_compatibility_check_never_raises_on_failure(tmp_path, synthetic_manifest):
    # A missing checkpoint would raise FileNotFoundError from run_calibration
    # directly - the secondary check must catch this and report it instead,
    # so an unavailable/broken secondary checkpoint never blocks Step 8A's
    # primary deliverable.
    result = run_secondary_compatibility_check(
        checkpoint_path=str(tmp_path / "does_not_exist.pt"),
        manifest_path=str(synthetic_manifest),
    )
    assert result["status"] == "failed"
    assert "error" in result
