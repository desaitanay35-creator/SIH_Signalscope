"""
Unit Tests for the EfficientNet-B4 Baseline Model.

Verifies architecture construction, output contract (single raw logit),
and specification metadata. Always uses pretrained=False so these tests
require no network access and never download ImageNet weights.
Responsible Team Member: Member 6 (MLOps & Testing)
"""

import pytest

torch = pytest.importorskip("torch")

from model.architectures.efficientnet_b4 import (
    EfficientNetB4Baseline,
    build_model,
    get_model_spec,
)


@pytest.fixture(scope="module")
def model():
    # pretrained=False: no network access, no weight download.
    return build_model(pretrained=False)


def test_build_model_succeeds(model):
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_model_contains_efficientnet_b4_backbone(model):
    assert isinstance(model, EfficientNetB4Baseline)
    assert hasattr(model, "backbone")
    # torchvision's EfficientNet-B4 backbone exposes `features` and `classifier`.
    assert hasattr(model.backbone, "features")
    assert hasattr(model.backbone, "classifier")


def test_classifier_produces_exactly_one_output_logit(model):
    final_layer = model.backbone.classifier[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    assert final_layer.out_features == 1


def test_forward_pass_output_shape():
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (2, 1)


def test_forward_pass_output_is_finite():
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert torch.isfinite(output).all()


def test_sigmoid_maps_logits_into_zero_one_range():
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.sigmoid(logits)
    assert probabilities.shape == logits.shape
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_model_constructs_without_pretrained_weights():
    # Must not raise and must not require network access.
    model = build_model(pretrained=False)
    assert model.pretrained is False


def test_forward_pass_works_at_a_non_default_input_size():
    # The architecture is fully-convolutional up to global pooling, so it
    # must not hard-fail on a different (still reasonable) input size.
    model = build_model(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 260, 260)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (2, 1)


def test_get_model_spec_reports_correct_label_semantics():
    spec = get_model_spec(pretrained=False)
    assert spec["label_mapping"] == {0: "real", 1: "ai_generated"}
    assert spec["num_output_logits"] == 1
    assert spec["channel_order"] == "RGB"
    assert spec["pretrained"] is False
    assert "probability_meaning" in spec and "sigmoid" in spec["probability_meaning"].lower()


def test_get_model_spec_reports_input_size_and_normalization():
    spec = get_model_spec(pretrained=False)
    assert spec["input_size"] == [224, 224]
    assert len(spec["normalization_mean"]) == 3
    assert len(spec["normalization_std"]) == 3


def test_model_get_spec_matches_module_level_spec():
    model = build_model(pretrained=False)
    instance_spec = model.get_spec()
    module_spec = get_model_spec(pretrained=False)
    assert instance_spec["architecture"] == module_spec["architecture"]
    assert instance_spec["label_mapping"] == module_spec["label_mapping"]


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model = build_model(pretrained=False)
    model.eval()
    checkpoint_path = tmp_path / "checkpoint.pth"
    model.save_checkpoint(checkpoint_path)
    assert checkpoint_path.is_file()

    reloaded = EfficientNetB4Baseline.load_checkpoint(checkpoint_path, pretrained=False)
    reloaded.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        original_output = model(x)
        reloaded_output = reloaded(x)
    assert torch.allclose(original_output, reloaded_output)


def test_model_does_not_apply_a_threshold(model):
    # The model must expose no threshold/decision attribute - thresholding
    # is an evaluation/inference-layer concern, not a model concern.
    assert not hasattr(model, "threshold")
    assert not hasattr(model, "decision_threshold")
