"""SignalScope ML model package (architectures, training, inference).

This package is intentionally decoupled from backend/ - it defines the
PyTorch model(s) SignalScope uses to classify real vs. AI-generated images.
The backend's `app/ml/detector.py` contract (raw_ai_probability,
model_version) is implemented by a future adapter, not by this package
directly.
"""
