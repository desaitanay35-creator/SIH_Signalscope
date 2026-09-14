"""
Unit Tests for Model Module.
Verifies model instantiation, weights loading, and inference CLI outputs.
Responsible Team Member: Member 6 (MLOps & Testing)
"""

def test_model_predict_cli():
    from model.predict import predict
    res = predict("dummy.jpg", "weights.pth")
    assert "is_ai_generated" in res
    assert "confidence_score" in res
