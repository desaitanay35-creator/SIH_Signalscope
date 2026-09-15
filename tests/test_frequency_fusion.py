"""
Unit Tests for the SignalScope Frequency-Domain Branch & Fusion Pipeline
(Step 6).

Covers: deterministic FFT transform, shared-augmented-image guarantee,
frequency/fusion architecture forward passes, gradient flow, batch-size
handling, checkpoint save/load roundtrip for all three architectures
(with backward-compatible and unknown-architecture registry behavior),
configuration validation, CPU smoke training for all three model types,
and Step 5 evaluation loading the new architectures. All against tiny
synthetic fixtures. Always uses pretrained=False and never touches the
real GenImage shard or any hidden SIH test set.

Responsible Team Member: Member 6 (MLOps & Testing)
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from config.settings import PreprocessingConfig
from data.dataset_loader import load_manifest
from data.preprocessor import ImagePreprocessor
from model.architectures.efficientnet_b4 import EfficientNetB4Baseline
from model.architectures.frequency_branch import (
    EMBEDDING_DIM as FREQUENCY_EMBEDDING_DIM,
)
from model.architectures.frequency_branch import (
    FrequencyBranch,
    FrequencyOnlyModel,
    build_frequency_only_model,
)
from model.architectures.fusion_model import (
    FUSION_DIM,
    RGB_EMBEDDING_DIM,
    RGBFrequencyFusionModel,
    build_fusion_model,
)
from model.evaluation.evaluate import run_evaluation
from model.training.checkpoint import (
    UnknownArchitectureError,
    load_model_from_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)
from model.training.config import TrainingConfig
from model.training.dual_branch_dataset import (
    AugmentedDualBranchPreprocessor,
    DeterministicDualBranchPreprocessor,
    build_dual_branch_train_dataset,
    build_dual_branch_val_dataset,
)
from model.training.frequency_features import compute_log_magnitude_spectrum
from model.training.train import _load_model_type, run_training


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_image(path, size=(32, 32), color=(120, 40, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _random_rgb_array(height=16, width=16, seed=0):
    rng = np.random.RandomState(seed)
    return rng.rand(height, width, 3).astype(np.float32)


@pytest.fixture
def preprocessor():
    return ImagePreprocessor.from_config(PreprocessingConfig(image_size=[16, 16]))


@pytest.fixture
def synthetic_manifest(tmp_path):
    """A tiny manifest with real + two generators, large enough that a
    smoke-test slice reliably contains both classes."""
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    rows = []
    for i in range(10):
        path = _write_image(real_dir / f"real_{i:02d}.jpg", color=(i * 10, i * 10, i * 10))
        rows.append({"image_path": str(path), "label": 0, "generator": "real"})
    for i in range(10):
        path = _write_image(fake_dir / f"sd_{i:02d}.jpg", color=(200, i * 10, 0))
        rows.append({"image_path": str(path), "label": 1, "generator": "stable_diffusion"})

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "label", "generator", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "split": ""})
    return manifest_path


def _model_config_yaml(tmp_path, manifest_path, name="model_config.yaml"):
    config_path = tmp_path / name
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
    unseen_generators: []
    held_out_generator_fraction: 0.5
""",
        encoding="utf-8",
    )
    return config_path


def _frequency_config_yaml(tmp_path, model_type, name="frequency_config.yaml"):
    config_path = tmp_path / name
    config_path.write_text(f'model_type: "{model_type}"\n', encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Deterministic FFT transform
# ---------------------------------------------------------------------------


def test_compute_log_magnitude_spectrum_is_deterministic():
    array = _random_rgb_array()
    a = compute_log_magnitude_spectrum(array)
    b = compute_log_magnitude_spectrum(array)
    assert torch.equal(a, b)


def test_compute_log_magnitude_spectrum_output_shape():
    for height, width in [(8, 8), (16, 24), (32, 32)]:
        array = _random_rgb_array(height, width)
        output = compute_log_magnitude_spectrum(array)
        assert tuple(output.shape) == (1, height, width)


def test_compute_log_magnitude_spectrum_is_finite_for_random_input():
    array = _random_rgb_array()
    output = compute_log_magnitude_spectrum(array)
    assert torch.isfinite(output).all()


def test_compute_log_magnitude_spectrum_is_finite_for_all_zero_input():
    array = np.zeros((16, 16, 3), dtype=np.float32)
    output = compute_log_magnitude_spectrum(array)
    assert torch.isfinite(output).all()
    # An all-zero image's spectrum is uniformly zero (log1p(0) = 0
    # everywhere); min == max == 0, so normalization must not divide by
    # zero (the _MIN_MAX_EPSILON guard) and must yield exactly zero.
    assert torch.equal(output, torch.zeros_like(output))


def test_compute_log_magnitude_spectrum_is_finite_for_nonzero_constant_input():
    array = np.full((16, 16, 3), 0.5, dtype=np.float32)
    output = compute_log_magnitude_spectrum(array)
    assert torch.isfinite(output).all()


def test_compute_log_magnitude_spectrum_normalized_to_unit_range():
    array = _random_rgb_array()
    output = compute_log_magnitude_spectrum(array)
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_compute_log_magnitude_spectrum_rejects_wrong_shape():
    with pytest.raises(ValueError):
        compute_log_magnitude_spectrum(np.zeros((16, 16), dtype=np.float32))  # missing channel dim


# ---------------------------------------------------------------------------
# Both branches receive the SAME augmented image (review requirement)
# ---------------------------------------------------------------------------


def test_deterministic_dual_branch_preprocessor_derives_both_tensors_from_same_image(tmp_path, preprocessor):
    image_path = _write_image(tmp_path / "sample.jpg", size=(20, 20), color=(50, 100, 150))
    dual = DeterministicDualBranchPreprocessor(preprocessor)

    result = dual.preprocess(image_path)
    assert set(result) == {"rgb", "frequency"}

    # Manually reconstruct what the SAME resized image should produce, and
    # confirm both derived tensors match exactly - proving they came from
    # one shared image object, not independently loaded/resized copies.
    image = preprocessor.load_image(image_path)
    image = image.resize(preprocessor.target_size, Image.BILINEAR)
    from model.training.augmentation import tensorize

    expected_rgb = tensorize(image, preprocessor)
    expected_frequency = compute_log_magnitude_spectrum(np.asarray(image, dtype=np.float32) / 255.0)

    assert torch.equal(result["rgb"], expected_rgb)
    assert torch.equal(result["frequency"], expected_frequency)


def test_augmented_dual_branch_preprocessor_returns_both_tensors_at_correct_shape(tmp_path, preprocessor):
    image_path = _write_image(tmp_path / "sample.jpg", size=(20, 20))
    dual = AugmentedDualBranchPreprocessor(preprocessor)
    result = dual.preprocess(image_path)

    assert tuple(result["rgb"].shape) == (3, 16, 16)
    assert tuple(result["frequency"].shape) == (1, 16, 16)
    assert torch.isfinite(result["rgb"]).all()
    assert torch.isfinite(result["frequency"]).all()


def test_dual_branch_datasets_build_over_a_real_manifest(synthetic_manifest, preprocessor):
    records = load_manifest(synthetic_manifest)
    train_dataset = build_dual_branch_train_dataset(records[:4], preprocessor)
    val_dataset = build_dual_branch_val_dataset(records[4:8], preprocessor)

    tensor, label, meta = train_dataset[0]
    assert set(tensor) == {"rgb", "frequency"}
    assert label in (0, 1)
    assert "generator" in meta

    tensor, label, meta = val_dataset[0]
    assert set(tensor) == {"rgb", "frequency"}


# ---------------------------------------------------------------------------
# Frequency-only architecture (explicitly defined - review requirement)
# ---------------------------------------------------------------------------


def test_frequency_branch_forward_pass_shape():
    branch = FrequencyBranch()
    x = torch.randn(3, 1, 16, 16)
    output = branch(x)
    assert tuple(output.shape) == (3, FREQUENCY_EMBEDDING_DIM)


def test_frequency_only_model_is_a_distinct_explicit_class():
    # The review requirement is that frequency-only is its OWN defined
    # architecture, not a fusion model with one branch zeroed out.
    model = build_frequency_only_model()
    assert isinstance(model, FrequencyOnlyModel)
    assert hasattr(model, "frequency_branch")
    assert not hasattr(model, "rgb_branch")


def test_frequency_only_model_forward_pass_with_plain_tensor():
    model = build_frequency_only_model()
    model.eval()
    x = torch.randn(4, 1, 16, 16)
    with torch.no_grad():
        output = model(x)
    assert tuple(output.shape) == (4, 1)
    assert torch.isfinite(output).all()


def test_frequency_only_model_forward_pass_with_dual_branch_dict():
    model = build_frequency_only_model()
    model.eval()
    batch = {"rgb": torch.randn(4, 3, 16, 16), "frequency": torch.randn(4, 1, 16, 16)}
    with torch.no_grad():
        output = model(batch)
    assert tuple(output.shape) == (4, 1)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_frequency_only_model_batch_size_handling(batch_size):
    model = build_frequency_only_model()
    model.eval()
    x = torch.randn(batch_size, 1, 16, 16)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (batch_size, 1)


def test_frequency_only_model_get_spec_reports_architecture_id():
    model = build_frequency_only_model()
    spec = model.get_spec()
    assert spec["architecture"] == "frequency_branch"
    assert spec["embedding_dim"] == FREQUENCY_EMBEDDING_DIM


def test_frequency_only_model_gradient_flow():
    model = build_frequency_only_model()
    model.train()
    x = torch.randn(4, 1, 16, 16)
    labels = torch.randint(0, 2, (4, 1)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), labels)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"


# ---------------------------------------------------------------------------
# Fusion model
# ---------------------------------------------------------------------------


def test_fusion_dimensions_match_inspected_efficientnet_b4_width():
    rgb_backbone = EfficientNetB4Baseline(pretrained=False)
    assert rgb_backbone.backbone.classifier[-1].in_features == RGB_EMBEDDING_DIM
    assert FUSION_DIM == RGB_EMBEDDING_DIM + FREQUENCY_EMBEDDING_DIM


def test_fusion_model_forward_pass_shape():
    model = build_fusion_model(pretrained=False)
    model.eval()
    batch = {"rgb": torch.randn(2, 3, 32, 32), "frequency": torch.randn(2, 1, 32, 32)}
    with torch.no_grad():
        output = model(batch)
    assert tuple(output.shape) == (2, 1)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("batch_size", [1, 3])
def test_fusion_model_batch_size_handling(batch_size):
    model = build_fusion_model(pretrained=False)
    model.eval()
    batch = {"rgb": torch.randn(batch_size, 3, 32, 32), "frequency": torch.randn(batch_size, 1, 32, 32)}
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (batch_size, 1)


def test_fusion_model_get_spec_reports_architecture_id():
    model = build_fusion_model(pretrained=False)
    spec = model.get_spec()
    assert spec["architecture"] == "rgb_frequency_fusion"
    assert spec["fusion_dim"] == FUSION_DIM


def test_fusion_model_gradient_flow_reaches_both_branches():
    model = build_fusion_model(pretrained=False)
    model.train()
    batch = {"rgb": torch.randn(2, 3, 32, 32), "frequency": torch.randn(2, 1, 32, 32)}
    labels = torch.randint(0, 2, (2, 1)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), labels)
    loss.backward()

    frequency_grad = model.frequency_branch.features[0].weight.grad
    # The RGB branch's original classifier head (backbone.classifier) is
    # intentionally bypassed by the fusion model's forward pass (see
    # RGBFrequencyFusionModel._rgb_embedding) and correctly has NO
    # gradient; check a layer from backbone.features instead, which IS
    # part of the fusion forward pass.
    rgb_grad = next(model.rgb_branch.backbone.features[-1].parameters()).grad
    assert frequency_grad is not None and torch.isfinite(frequency_grad).all()
    assert rgb_grad is not None and torch.isfinite(rgb_grad).all()
    assert model.rgb_branch.backbone.classifier[-1].weight.grad is None  # confirms it's truly unused


# ---------------------------------------------------------------------------
# Checkpoint registry: backward compatibility + clear rejection (review requirement)
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_for_all_three_architectures(tmp_path):
    builders = {
        "efficientnet_b4": lambda: EfficientNetB4Baseline(pretrained=False),
        "frequency_branch": build_frequency_only_model,
        "rgb_frequency_fusion": lambda: build_fusion_model(pretrained=False),
    }
    inputs = {
        "efficientnet_b4": torch.randn(1, 3, 224, 224),
        "frequency_branch": torch.randn(1, 1, 16, 16),
        "rgb_frequency_fusion": {"rgb": torch.randn(1, 3, 32, 32), "frequency": torch.randn(1, 1, 32, 32)},
    }

    for architecture_id, builder in builders.items():
        model = builder()
        model.eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        checkpoint_path = tmp_path / f"{architecture_id}.pt"

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler=None,
            epoch=1,
            best_val_roc_auc=0.6,
            val_metrics_history=[],
            training_config=TrainingConfig(),
            seed=42,
        )

        checkpoint = load_training_checkpoint(checkpoint_path)
        assert checkpoint["model_config"]["architecture"] == architecture_id

        reloaded = load_model_from_checkpoint(checkpoint_path)
        reloaded.eval()

        with torch.no_grad():
            original_output = model(inputs[architecture_id])
            reloaded_output = reloaded(inputs[architecture_id])
        assert torch.allclose(original_output, reloaded_output)


def test_load_model_from_checkpoint_defaults_to_efficientnet_b4_when_architecture_field_absent(tmp_path):
    # Simulates a checkpoint saved before this registry existed (Step 4/5)
    # - no "architecture" key in model_config at all.
    model = EfficientNetB4Baseline(pretrained=False)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"name": "EfficientNet-B4"},  # no "architecture" key
        },
        checkpoint_path,
    )
    reloaded = load_model_from_checkpoint(checkpoint_path)
    assert isinstance(reloaded, EfficientNetB4Baseline)


def test_load_model_from_checkpoint_defaults_to_efficientnet_b4_when_model_config_absent(tmp_path):
    model = EfficientNetB4Baseline(pretrained=False)
    checkpoint_path = tmp_path / "very_legacy.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    reloaded = load_model_from_checkpoint(checkpoint_path)
    assert isinstance(reloaded, EfficientNetB4Baseline)


def test_load_model_from_checkpoint_rejects_unknown_architecture_clearly(tmp_path):
    checkpoint_path = tmp_path / "unknown_arch.pt"
    torch.save(
        {
            "model_state_dict": {},
            "model_config": {"architecture": "some_future_architecture_nobody_registered"},
        },
        checkpoint_path,
    )
    with pytest.raises(UnknownArchitectureError):
        load_model_from_checkpoint(checkpoint_path)


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_load_model_type_defaults_to_rgb_only_when_file_absent(tmp_path):
    assert _load_model_type(str(tmp_path / "does_not_exist.yaml")) == "rgb_only"


def test_load_model_type_reads_frequency_only(tmp_path):
    config_path = _frequency_config_yaml(tmp_path, "frequency_only")
    assert _load_model_type(str(config_path)) == "frequency_only"


def test_load_model_type_reads_rgb_frequency_fusion(tmp_path):
    config_path = _frequency_config_yaml(tmp_path, "rgb_frequency_fusion")
    assert _load_model_type(str(config_path)) == "rgb_frequency_fusion"


def test_load_model_type_rejects_unrecognized_value_clearly(tmp_path):
    config_path = _frequency_config_yaml(tmp_path, "not_a_real_model_type")
    with pytest.raises(ValueError):
        _load_model_type(str(config_path))


# ---------------------------------------------------------------------------
# CPU smoke training: RGB-only (regression), frequency-only, fusion
# ---------------------------------------------------------------------------


def test_cpu_smoke_training_rgb_only_still_works_unchanged(tmp_path, synthetic_manifest):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest)
    training_config = TrainingConfig(
        seed=42, epochs=1, batch_size=2, num_workers=0,
        max_train_samples=6, max_val_samples=4,
        checkpoint_dir=str(tmp_path / "runs_rgb"),
    )
    summary = run_training(training_config, model_config_path=str(model_config_path), smoke_test=True)
    assert summary["epochs_run"] == 1
    checkpoint = load_training_checkpoint(Path(summary["run_dir"]) / "checkpoints" / "last.pt")
    assert checkpoint["model_config"]["architecture"] == "efficientnet_b4"


def test_cpu_smoke_training_frequency_only(tmp_path, synthetic_manifest):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest)
    frequency_config_path = _frequency_config_yaml(tmp_path, "frequency_only")
    training_config = TrainingConfig(
        seed=42, epochs=1, batch_size=2, num_workers=0,
        max_train_samples=6, max_val_samples=4,
        checkpoint_dir=str(tmp_path / "runs_frequency"),
    )
    summary = run_training(
        training_config,
        model_config_path=str(model_config_path),
        smoke_test=True,
        frequency_config_path=str(frequency_config_path),
    )
    assert summary["epochs_run"] == 1
    checkpoint = load_training_checkpoint(Path(summary["run_dir"]) / "checkpoints" / "last.pt")
    assert checkpoint["model_config"]["architecture"] == "frequency_branch"

    reloaded = load_model_from_checkpoint(Path(summary["run_dir"]) / "checkpoints" / "last.pt")
    assert isinstance(reloaded, FrequencyOnlyModel)


def test_cpu_smoke_training_rgb_frequency_fusion(tmp_path, synthetic_manifest):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest)
    frequency_config_path = _frequency_config_yaml(tmp_path, "rgb_frequency_fusion")
    training_config = TrainingConfig(
        seed=42, epochs=1, batch_size=2, num_workers=0,
        max_train_samples=6, max_val_samples=4,
        checkpoint_dir=str(tmp_path / "runs_fusion"),
    )
    summary = run_training(
        training_config,
        model_config_path=str(model_config_path),
        smoke_test=True,
        frequency_config_path=str(frequency_config_path),
    )
    assert summary["epochs_run"] == 1
    checkpoint = load_training_checkpoint(Path(summary["run_dir"]) / "checkpoints" / "last.pt")
    assert checkpoint["model_config"]["architecture"] == "rgb_frequency_fusion"

    reloaded = load_model_from_checkpoint(Path(summary["run_dir"]) / "checkpoints" / "last.pt")
    assert isinstance(reloaded, RGBFrequencyFusionModel)


def Path(value):
    from pathlib import Path

    return Path(value)


# ---------------------------------------------------------------------------
# Step 5 evaluation loading the new architectures
# ---------------------------------------------------------------------------


def test_evaluation_loads_frequency_only_checkpoint(tmp_path, synthetic_manifest):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest)
    frequency_config_path = _frequency_config_yaml(tmp_path, "frequency_only")
    training_config = TrainingConfig(
        seed=42, epochs=1, batch_size=2, num_workers=0,
        max_train_samples=6, max_val_samples=4,
        checkpoint_dir=str(tmp_path / "runs_eval_frequency"),
    )
    train_summary = run_training(
        training_config,
        model_config_path=str(model_config_path),
        smoke_test=True,
        frequency_config_path=str(frequency_config_path),
    )
    checkpoint_path = Path(train_summary["run_dir"]) / "checkpoints" / "last.pt"

    eval_summary = run_evaluation(
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "eval_output_frequency"),
    )
    assert eval_summary["test_roc_auc"] is None or isinstance(eval_summary["test_roc_auc"], float)


def test_evaluation_loads_fusion_checkpoint(tmp_path, synthetic_manifest):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest)
    frequency_config_path = _frequency_config_yaml(tmp_path, "rgb_frequency_fusion")
    training_config = TrainingConfig(
        seed=42, epochs=1, batch_size=2, num_workers=0,
        max_train_samples=6, max_val_samples=4,
        checkpoint_dir=str(tmp_path / "runs_eval_fusion"),
    )
    train_summary = run_training(
        training_config,
        model_config_path=str(model_config_path),
        smoke_test=True,
        frequency_config_path=str(frequency_config_path),
    )
    checkpoint_path = Path(train_summary["run_dir"]) / "checkpoints" / "last.pt"

    eval_summary = run_evaluation(
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "eval_output_fusion"),
    )
    assert eval_summary["test_roc_auc"] is None or isinstance(eval_summary["test_roc_auc"], float)


def test_evaluation_still_loads_rgb_only_checkpoint_unchanged(tmp_path, synthetic_manifest):
    model_config_path = _model_config_yaml(tmp_path, synthetic_manifest)
    training_config = TrainingConfig(
        seed=42, epochs=1, batch_size=2, num_workers=0,
        max_train_samples=6, max_val_samples=4,
        checkpoint_dir=str(tmp_path / "runs_eval_rgb"),
    )
    train_summary = run_training(training_config, model_config_path=str(model_config_path), smoke_test=True)
    checkpoint_path = Path(train_summary["run_dir"]) / "checkpoints" / "last.pt"

    eval_summary = run_evaluation(
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(synthetic_manifest),
        model_config_path=str(model_config_path),
        output_dir_override=str(tmp_path / "eval_output_rgb"),
    )
    assert eval_summary["test_roc_auc"] is None or isinstance(eval_summary["test_roc_auc"], float)
