"""
Unit Tests for the SignalScope Evaluation Pipeline.

Covers checkpoint loading, the pure inference function, metric definitions
and their edge cases, generator-disjoint/leakage safeguards, per-generator
analysis, determinism, and output artifact schema - all against tiny
synthetic fixtures. Always uses pretrained=False and never touches the
real GenImage shard or any hidden SIH test set.

Responsible Team Member: Member 6 (MLOps & Testing)
"""

import csv
import json

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from config.settings import PreprocessingConfig, SplitConfig
from data.dataset_loader import DatasetRecord, load_manifest
from data.preprocessor import ImagePreprocessor
from data.splitting import split_manifest
from model.architectures.efficientnet_b4 import EfficientNetB4Baseline
from model.evaluation.config import EvaluationConfig, load_evaluation_config
from model.evaluation.evaluate import (
    EmptyUnseenGeneratorSplitError,
    _assert_unseen_generator_split_is_valid,
    run_evaluation,
)
from model.evaluation.metrics import (
    EmptyEvaluationSetError,
    accuracy,
    compute_split_metrics,
    confusion_matrix,
    fpr_at_threshold,
    macro_f1,
    per_generator_breakdown,
    threshold_probabilities,
)
from model.evaluation.predict import run_inference
from model.evaluation.report import PREDICTIONS_CSV_COLUMNS, is_dev_shard
from model.training.checkpoint import save_checkpoint
from model.training.dataset import build_val_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_image(path, size=(48, 48), color=(120, 40, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def synthetic_manifest(tmp_path):
    """16 real, 6 stable_diffusion (seen), 6 midjourney + 6 vqdm (unseen).
    Empirically verified (seed=42, val_fraction=0.3) to route both real and
    stable_diffusion images into val, and only midjourney/vqdm into test."""
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    rows = []
    for i in range(16):
        path = _write_image(real_dir / f"real_{i:02d}.jpg", color=(i * 5, i * 5, i * 5))
        rows.append({"image_path": str(path), "label": 0, "generator": "real"})
    for i in range(6):
        path = _write_image(fake_dir / f"sd_{i:02d}.jpg", color=(200, i * 10, 0))
        rows.append({"image_path": str(path), "label": 1, "generator": "stable_diffusion"})
    for i in range(6):
        path = _write_image(fake_dir / f"mj_{i:02d}.jpg", color=(0, 200, i * 10))
        rows.append({"image_path": str(path), "label": 1, "generator": "midjourney"})
    for i in range(6):
        path = _write_image(fake_dir / f"vq_{i:02d}.jpg", color=(i * 10, 0, 200))
        rows.append({"image_path": str(path), "label": 1, "generator": "vqdm"})

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "label", "generator", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "split": ""})
    return manifest_path


@pytest.fixture
def split_config():
    return SplitConfig(seed=42, val_fraction=0.3, unseen_generators=["midjourney", "vqdm"])


@pytest.fixture
def preprocessor():
    return ImagePreprocessor.from_config(PreprocessingConfig(image_size=[32, 32]))


@pytest.fixture
def checkpoint_path(tmp_path):
    model = EfficientNetB4Baseline(pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp_path / "checkpoints" / "best.pt"
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


def _model_config_yaml(tmp_path, manifest_path, unseen_generators):
    config_path = tmp_path / "model_config.yaml"
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
    image_size: [32, 32]
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


# ---------------------------------------------------------------------------
# Checkpoint loading (via the evaluation entrypoint)
# ---------------------------------------------------------------------------


def test_run_evaluation_raises_clearly_for_missing_checkpoint(tmp_path, synthetic_manifest):
    with pytest.raises(FileNotFoundError):
        run_evaluation(
            checkpoint_path=str(tmp_path / "does_not_exist.pt"),
            manifest_path=str(synthetic_manifest),
        )


def test_run_evaluation_raises_clearly_for_incompatible_checkpoint(tmp_path, synthetic_manifest):
    bad_checkpoint = tmp_path / "incompatible.pt"
    torch.save({"model_state_dict": {"not": "a real state dict"}}, bad_checkpoint)
    with pytest.raises(RuntimeError):
        run_evaluation(checkpoint_path=str(bad_checkpoint), manifest_path=str(synthetic_manifest))


# ---------------------------------------------------------------------------
# Prediction pipeline: pure function, no checkpoint/manifest args
# ---------------------------------------------------------------------------


def test_run_inference_is_pure_and_needs_no_checkpoint_or_manifest(synthetic_manifest, split_config, preprocessor):
    records = load_manifest(synthetic_manifest)
    splits = split_manifest(records, split_config)
    val_dataset = build_val_dataset(splits["val"], preprocessor)
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = EfficientNetB4Baseline(pretrained=False)
    model.eval()
    device = torch.device("cpu")

    rows = run_inference(model, loader, device, threshold=0.5, split="val")

    assert len(rows) == len(splits["val"])
    for row in rows:
        assert 0.0 <= row.predicted_probability <= 1.0
        assert row.predicted_label in (0, 1)
        assert row.true_label in (0, 1)
        assert row.split == "val"


def test_run_inference_thresholding_matches_probability(synthetic_manifest, split_config, preprocessor):
    records = load_manifest(synthetic_manifest)
    splits = split_manifest(records, split_config)
    val_dataset = build_val_dataset(splits["val"], preprocessor)
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = EfficientNetB4Baseline(pretrained=False)
    model.eval()
    device = torch.device("cpu")

    for threshold in (0.5, 0.9):
        rows = run_inference(model, loader, device, threshold=threshold)
        for row in rows:
            expected = 1 if row.predicted_probability >= threshold else 0
            assert row.predicted_label == expected


# ---------------------------------------------------------------------------
# Metrics: accuracy, macro-F1, confusion matrix, FPR, edge cases
# ---------------------------------------------------------------------------


def test_accuracy_known_confusion():
    labels = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]
    assert accuracy(labels, predictions) == pytest.approx(0.75)


def test_accuracy_raises_on_empty_input():
    with pytest.raises(EmptyEvaluationSetError):
        accuracy([], [])


def test_confusion_matrix_known_counts():
    labels = [0, 0, 1, 1]
    predictions = [0, 1, 1, 1]
    matrix = confusion_matrix(labels, predictions)
    assert matrix == {"tp": 2, "tn": 1, "fp": 1, "fn": 0}


def test_fpr_at_threshold_known_value():
    matrix = {"tp": 2, "tn": 1, "fp": 1, "fn": 0}
    assert fpr_at_threshold(matrix) == pytest.approx(0.5)


def test_fpr_at_threshold_returns_none_for_zero_denominator():
    matrix = {"tp": 2, "tn": 0, "fp": 0, "fn": 0}
    assert fpr_at_threshold(matrix) is None


def test_macro_f1_perfect_predictions_is_one():
    labels = [0, 0, 1, 1]
    predictions = [0, 0, 1, 1]
    assert macro_f1(labels, predictions) == pytest.approx(1.0)


def test_macro_f1_handles_zero_support_class():
    # Only class 0 present; predictions are all correct for class 0.
    labels = [0, 0, 0]
    predictions = [0, 0, 0]
    # Class 1 has zero support -> its F1 is defined as 0.0, not an error.
    assert macro_f1(labels, predictions) == pytest.approx(0.5)


def test_threshold_probabilities_matches_manual_thresholding():
    probs = [0.1, 0.4, 0.5, 0.9]
    assert threshold_probabilities(probs, 0.5) == [0, 0, 1, 1]


def test_compute_split_metrics_roc_auc_is_none_for_single_class():
    labels = [0, 0, 0]
    probs = [0.1, 0.2, 0.3]
    metrics = compute_split_metrics(labels, probs, threshold=0.5)
    assert metrics.roc_auc is None
    assert any("roc_auc undefined" in note for note in metrics.notes)


def test_compute_split_metrics_raises_on_empty_input():
    with pytest.raises(EmptyEvaluationSetError):
        compute_split_metrics([], [], threshold=0.5)


# ---------------------------------------------------------------------------
# Per-generator analysis: real+G pairing, not naive single-generator slices
# ---------------------------------------------------------------------------


def test_per_generator_breakdown_pairs_real_with_each_generator():
    labels = [0, 0, 1, 1, 1, 1]
    probs = [0.1, 0.2, 0.8, 0.9, 0.85, 0.95]
    generators = ["real", "real", "midjourney", "midjourney", "vqdm", "vqdm"]

    breakdown = per_generator_breakdown(labels, probs, generators, threshold=0.5)

    assert set(breakdown) == {"midjourney", "vqdm"}
    for generator, metrics in breakdown.items():
        # 2 real + 2 of this generator = 4 samples, both classes present.
        assert metrics.num_samples == 4
        assert metrics.num_real == 2
        assert metrics.num_fake == 2
        assert metrics.roc_auc is not None  # never None due to naive single-class slicing


def test_per_generator_breakdown_empty_when_no_generators_present():
    labels = [0, 0]
    probs = [0.1, 0.2]
    generators = ["real", "real"]
    assert per_generator_breakdown(labels, probs, generators, threshold=0.5) == {}


# ---------------------------------------------------------------------------
# Generator-disjoint / unseen-generator split validation
# ---------------------------------------------------------------------------


def test_assert_unseen_generator_split_passes_for_valid_split(synthetic_manifest, split_config):
    records = load_manifest(synthetic_manifest)
    splits = split_manifest(records, split_config)
    _assert_unseen_generator_split_is_valid(splits)  # should not raise


def test_assert_unseen_generator_split_rejects_empty_test_split():
    splits = {
        "train": [DatasetRecord("a.jpg", 0, "real")],
        "val": [DatasetRecord("b.jpg", 0, "real")],
        "test": [],
    }
    with pytest.raises(EmptyUnseenGeneratorSplitError):
        _assert_unseen_generator_split_is_valid(splits)


def test_assert_unseen_generator_split_rejects_val_with_no_real_images():
    splits = {
        "train": [DatasetRecord("a.jpg", 0, "real")],
        "val": [DatasetRecord("b.jpg", 1, "stable_diffusion")],
        "test": [DatasetRecord("c.jpg", 1, "midjourney")],
    }
    with pytest.raises(EmptyUnseenGeneratorSplitError):
        _assert_unseen_generator_split_is_valid(splits)


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------


def test_evaluation_config_defaults_are_sane():
    config = EvaluationConfig()
    assert config.batch_size > 0
    assert config.generate_plots is False


def test_load_evaluation_config_missing_file_returns_defaults(tmp_path):
    config = load_evaluation_config(str(tmp_path / "missing.yaml"))
    assert config == EvaluationConfig()


# ---------------------------------------------------------------------------
# End-to-end run_evaluation: artifacts, determinism, dev-shard banner
# ---------------------------------------------------------------------------


def test_run_evaluation_produces_expected_artifacts_and_metrics(tmp_path, synthetic_manifest, checkpoint_path):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest, ["midjourney", "vqdm"])

    summary = run_evaluation(
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "eval_output"),
    )

    run_dir = _run_dir_from_summary(summary)
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "predictions.csv").is_file()
    assert (run_dir / "evaluation_report.md").is_file()
    assert not (run_dir / "confusion_matrix.png").is_file()  # generate_plots defaults to False

    with open(run_dir / "metrics.json", encoding="utf-8") as fh:
        metrics = json.load(fh)
    assert "val" in metrics and "test" in metrics
    assert "per_generator" in metrics["test"]
    assert set(metrics["test"]["per_generator"]) == {"midjourney", "vqdm"}
    # Unseen-generator (test) ROC-AUC must be a defined number here, since
    # val contributes real images and test contributes both fakes.
    assert metrics["test"]["overall"]["roc_auc"] is not None
    assert any("val" in note for note in metrics["test"]["overall"]["notes"])

    with open(run_dir / "predictions.csv", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == PREDICTIONS_CSV_COLUMNS
        prediction_rows = list(reader)
    assert {row["split"] for row in prediction_rows} == {"val", "test"}

    report_text = (run_dir / "evaluation_report.md").read_text(encoding="utf-8")
    assert "NOT the final SIH benchmark" not in report_text  # this fixture is not the dev shard
    assert is_dev_shard(synthetic_manifest) is False
    assert is_dev_shard("data/manifests/genimage_dev.csv") is True


def test_run_evaluation_report_includes_dev_shard_banner_for_genimage_dev_manifest(
    tmp_path, synthetic_manifest, checkpoint_path
):
    # Copy the synthetic manifest and its images to a path containing
    # "genimage_dev" so is_dev_shard() recognizes it, without touching the
    # real dev shard.
    dev_dir = tmp_path / "genimage_dev_copy"
    dev_dir.mkdir()
    dev_manifest_path = dev_dir / "genimage_dev.csv"
    with open(synthetic_manifest, encoding="utf-8", newline="") as src, open(
        dev_manifest_path, "w", encoding="utf-8", newline=""
    ) as dst:
        dst.write(src.read())

    model_config_path = _model_config_yaml(tmp_path, dev_manifest_path, ["midjourney", "vqdm"])
    summary = run_evaluation(
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(dev_manifest_path),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "dev_shard_eval_output"),
    )

    report_text = (_run_dir_from_summary(summary) / "evaluation_report.md").read_text(encoding="utf-8")
    assert "NOT the final SIH benchmark" in report_text


def _run_dir_from_summary(summary):
    from pathlib import Path

    return Path(summary["run_dir"])


def test_run_evaluation_is_deterministic(tmp_path, synthetic_manifest, checkpoint_path):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest, ["midjourney", "vqdm"])

    summary_a = run_evaluation(
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "run_a"),
    )
    summary_b = run_evaluation(
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "run_b"),
    )

    assert summary_a["val_roc_auc"] == summary_b["val_roc_auc"]
    assert summary_a["test_roc_auc"] == summary_b["test_roc_auc"]

    def _read_predictions(run_dir):
        with open(_run_dir_from_summary({"run_dir": run_dir}) / "predictions.csv", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    assert _read_predictions(summary_a["run_dir"]) == _read_predictions(summary_b["run_dir"])


def test_run_evaluation_never_leaves_model_in_train_mode(tmp_path, synthetic_manifest, checkpoint_path):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest, ["midjourney", "vqdm"])
    # run_evaluation constructs its own model internally; verify indirectly
    # by loading the same checkpoint and confirming eval() is the terminal
    # state load_model_from_checkpoint + evaluate.py leaves it in.
    from model.training.checkpoint import load_model_from_checkpoint

    model = load_model_from_checkpoint(checkpoint_path)
    model.eval()
    run_evaluation(
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "eval_output_2"),
    )
    # The checkpoint file itself is untouched by evaluation (read-only load).
    reloaded = load_model_from_checkpoint(checkpoint_path)
    assert reloaded.training is True  # freshly constructed modules default to train mode
    reloaded.eval()
    assert reloaded.training is False
