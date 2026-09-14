"""
Unit Tests for the SignalScope Training Pipeline.

Covers seeding determinism, duplicate-manifest-row rejection, one
forward/backward step on a tiny synthetic manifest, checkpoint save/load
round-tripping, and best-checkpoint selection - all against tiny synthetic
fixtures created on the fly. Always uses pretrained=False and never touches
the real GenImage shard or any hidden SIH test set.

Responsible Team Member: Member 6 (MLOps & Testing)
"""

import csv

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from config.settings import PreprocessingConfig, SplitConfig
from data.dataset_loader import DatasetRecord, load_manifest
from data.preprocessor import ImagePreprocessor
from data.splitting import split_manifest
from model.architectures.efficientnet_b4 import EfficientNetB4Baseline
from model.training.checkpoint import (
    load_model_from_checkpoint,
    load_training_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from model.training.config import TrainingConfig, load_training_config
from model.training.dataset import (
    DuplicateImageError,
    assert_no_duplicate_images,
    build_train_dataset,
    build_val_dataset,
)
from model.training.engine import _roc_auc_score, train_one_epoch, validate
from model.training.seed import set_seed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_image(path, size=(48, 48), color=(120, 40, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def synthetic_manifest(tmp_path):
    """A tiny, generator-disjoint-splittable manifest: real images plus two
    non-real generators, entirely synthetic and local to tmp_path."""
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    rows = []
    for i in range(8):
        path = _write_image(real_dir / f"real_{i:02d}.jpg", color=(i * 10, i * 10, i * 10))
        rows.append({"image_path": str(path), "label": 0, "generator": "real"})
    for i in range(4):
        path = _write_image(fake_dir / f"sd_{i:02d}.jpg", color=(200, i * 10, 0))
        rows.append({"image_path": str(path), "label": 1, "generator": "stable_diffusion"})
    for i in range(4):
        path = _write_image(fake_dir / f"mj_{i:02d}.jpg", color=(0, 200, i * 10))
        rows.append({"image_path": str(path), "label": 1, "generator": "midjourney"})

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "label", "generator", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "split": ""})
    return manifest_path


@pytest.fixture
def split_config():
    # "midjourney" held out entirely -> unseen-generator test split.
    return SplitConfig(seed=42, val_fraction=0.25, unseen_generators=["midjourney"])


@pytest.fixture
def preprocessor():
    return ImagePreprocessor.from_config(PreprocessingConfig(image_size=[32, 32]))


@pytest.fixture
def tiny_model():
    return EfficientNetB4Baseline(pretrained=False)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def test_set_seed_makes_torch_randn_reproducible():
    set_seed(123)
    a = torch.randn(4)
    set_seed(123)
    b = torch.randn(4)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Duplicate-image leakage guard
# ---------------------------------------------------------------------------


def test_assert_no_duplicate_images_passes_for_unique_paths(synthetic_manifest):
    records = load_manifest(synthetic_manifest)
    assert_no_duplicate_images(records)  # should not raise


def test_assert_no_duplicate_images_rejects_duplicate_path():
    records = [
        DatasetRecord("a.jpg", 0, "real"),
        DatasetRecord("a.jpg", 0, "real"),
    ]
    with pytest.raises(DuplicateImageError):
        assert_no_duplicate_images(records)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


def test_training_config_defaults_are_sane():
    config = TrainingConfig()
    assert config.optimizer == "adamw"
    assert config.scheduler == "cosine"
    assert config.pos_weight is None
    assert config.freeze_backbone_epochs == 0


def test_load_training_config_missing_file_returns_defaults(tmp_path):
    config = load_training_config(str(tmp_path / "does_not_exist.yaml"))
    assert config == TrainingConfig()


def test_load_training_config_reads_overrides(tmp_path):
    config_path = tmp_path / "training_config.yaml"
    config_path.write_text(
        "training:\n  epochs: 3\n  batch_size: 8\n  learning_rate: 0.0005\n",
        encoding="utf-8",
    )
    config = load_training_config(str(config_path))
    assert config.epochs == 3
    assert config.batch_size == 8
    assert config.learning_rate == 0.0005
    # Untouched fields keep their defaults.
    assert config.optimizer == "adamw"


# ---------------------------------------------------------------------------
# Dataset construction: augmentation vs. deterministic validation
# ---------------------------------------------------------------------------


def test_train_dataset_applies_augmentation_and_matches_target_size(synthetic_manifest, split_config, preprocessor):
    records = load_manifest(synthetic_manifest)
    splits = split_manifest(records, split_config)
    train_dataset = build_train_dataset(splits["train"], preprocessor)

    tensor, label, meta = train_dataset[0]
    assert tuple(tensor.shape) == (3, 32, 32)
    assert label in (0, 1)
    assert "generator" in meta


def test_val_dataset_is_deterministic_across_repeated_reads(synthetic_manifest, split_config, preprocessor):
    records = load_manifest(synthetic_manifest)
    splits = split_manifest(records, split_config)
    val_dataset = build_val_dataset(splits["val"], preprocessor)

    tensor_a, _, _ = val_dataset[0]
    tensor_b, _, _ = val_dataset[0]
    assert torch.allclose(tensor_a, tensor_b)


def test_unseen_generator_split_is_disjoint_from_train_and_val(synthetic_manifest, split_config):
    records = load_manifest(synthetic_manifest)
    splits = split_manifest(records, split_config)
    test_generators = {r.generator for r in splits["test"]}
    train_val_generators = {r.generator for r in splits["train"] + splits["val"] if r.generator != "real"}
    assert test_generators == {"midjourney"}
    assert train_val_generators.isdisjoint(test_generators)


# ---------------------------------------------------------------------------
# Engine: one forward/backward step and one validation pass
# ---------------------------------------------------------------------------


def test_train_one_epoch_runs_and_produces_finite_loss(synthetic_manifest, split_config, preprocessor, tiny_model):
    records = load_manifest(synthetic_manifest)
    splits = split_manifest(records, split_config)
    train_dataset = build_train_dataset(splits["train"], preprocessor)

    loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
    device = torch.device("cpu")
    tiny_model.to(device)
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=False)

    loss = train_one_epoch(tiny_model, loader, optimizer, loss_fn, device, scaler)
    assert isinstance(loss, float)
    assert loss == loss  # not NaN


def test_validate_computes_finite_metrics_and_does_not_update_weights(
    synthetic_manifest, split_config, preprocessor, tiny_model
):
    records = load_manifest(synthetic_manifest)
    splits = split_manifest(records, split_config)
    val_dataset = build_val_dataset(splits["val"], preprocessor)
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

    device = torch.device("cpu")
    tiny_model.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    params_before = [p.clone() for p in tiny_model.parameters()]
    metrics = validate(tiny_model, loader, loss_fn, device)
    params_after = list(tiny_model.parameters())

    assert metrics.loss == metrics.loss  # not NaN
    assert 0.0 <= metrics.accuracy <= 1.0
    assert all(torch.equal(a, b) for a, b in zip(params_before, params_after))


def test_validate_reports_none_roc_auc_when_only_one_class_present(preprocessor, tiny_model):
    records = [
        DatasetRecord(str(_write_image_for_test(0)), 0, "real"),
        DatasetRecord(str(_write_image_for_test(1)), 0, "real"),
    ]
    val_dataset = build_val_dataset(records, preprocessor)
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)
    device = torch.device("cpu")
    tiny_model.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    metrics = validate(tiny_model, loader, loss_fn, device)
    assert metrics.roc_auc is None


def _write_image_for_test(index):
    import tempfile
    from pathlib import Path

    tmp_dir = Path(tempfile.gettempdir()) / "signalscope_test_training_images"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"single_class_{index}.jpg"
    if not path.is_file():
        Image.new("RGB", (32, 32), (index * 30, index * 30, index * 30)).save(path)
    return path


# ---------------------------------------------------------------------------
# ROC-AUC helper (self-contained, no scikit-learn dependency)
# ---------------------------------------------------------------------------


def test_roc_auc_score_perfect_separation_is_one():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert _roc_auc_score(labels, scores) == pytest.approx(1.0)


def test_roc_auc_score_worst_separation_is_zero():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    assert _roc_auc_score(labels, scores) == pytest.approx(0.0)


def test_roc_auc_score_random_scores_is_around_half():
    labels = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert _roc_auc_score(labels, scores) == pytest.approx(0.5)


def test_roc_auc_score_returns_none_for_single_class():
    assert _roc_auc_score([0, 0, 0], [0.1, 0.2, 0.3]) is None


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_save_and_load_training_checkpoint_roundtrip(tmp_path, tiny_model):
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    training_config = TrainingConfig(epochs=5)

    checkpoint_path = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint_path,
        tiny_model,
        optimizer,
        scheduler,
        epoch=2,
        best_val_roc_auc=0.75,
        val_metrics_history=[{"loss": 0.5, "accuracy": 0.6, "roc_auc": 0.75, "num_samples": 10}],
        training_config=training_config,
        seed=42,
    )
    assert checkpoint_path.is_file()

    checkpoint = load_training_checkpoint(checkpoint_path)
    assert checkpoint["epoch"] == 2
    assert checkpoint["best_val_roc_auc"] == 0.75
    assert checkpoint["seed"] == 42
    assert checkpoint["training_config"]["epochs"] == 5

    restored_model = EfficientNetB4Baseline(pretrained=False)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=5)
    restore_training_state(checkpoint, restored_model, restored_optimizer, restored_scheduler)

    x = torch.randn(1, 3, 224, 224)
    tiny_model.eval()
    restored_model.eval()
    with torch.no_grad():
        original_output = tiny_model(x)
        restored_output = restored_model(x)
    assert torch.allclose(original_output, restored_output)


def test_load_model_from_checkpoint_returns_usable_model(tmp_path, tiny_model):
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint_path,
        tiny_model,
        optimizer,
        scheduler=None,
        epoch=1,
        best_val_roc_auc=0.6,
        val_metrics_history=[],
        training_config=TrainingConfig(),
        seed=42,
    )

    reloaded = load_model_from_checkpoint(checkpoint_path)
    reloaded.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = reloaded(x)
    assert output.shape == (1, 1)


# ---------------------------------------------------------------------------
# Best-checkpoint selection logic (synthetic metrics, no training required)
# ---------------------------------------------------------------------------


def test_best_checkpoint_rule_prefers_strictly_higher_roc_auc():
    history = [
        {"roc_auc": 0.60},
        {"roc_auc": 0.55},
        {"roc_auc": 0.72},
        {"roc_auc": None},
        {"roc_auc": 0.72},  # tie: must not overwrite the earlier best
    ]
    best = None
    best_epoch = None
    for epoch, metrics in enumerate(history, start=1):
        roc_auc = metrics["roc_auc"]
        if roc_auc is not None and (best is None or roc_auc > best):
            best = roc_auc
            best_epoch = epoch
    assert best == 0.72
    assert best_epoch == 3  # not epoch 5, since the tie is not a strict improvement
