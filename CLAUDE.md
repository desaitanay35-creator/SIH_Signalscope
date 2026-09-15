# CLAUDE.md

## Project overview

SignalScope is an AI-powered image authenticity detector for SIH 2026. Core task: classify images as real vs AI-generated, with strong generalization to unseen generators.

## Architecture

```text
signalscope/
├── app/                 # FastAPI inference API
├── model/
│   ├── architectures/  # Detector + fusion models
│   ├── training/       # Training/configuration
│   ├── inference/      # Prediction pipeline
│   ├── explainability/ # Grad-CAM/evidence
│   ├── calibration/    # Probability calibration
│   └── robustness/     # Degradation tests
├── experiments/        # Experiment configs/results
├── report/             # SIH report + samples
├── requirements.txt
└── README.md
```

## Code style

* Python: PEP 8, type hints, clear modular functions
* PyTorch for ML; FastAPI for inference
* Never hardcode dataset paths or model parameters
* Log every experiment and metric
* Prefer reproducibility over cleverness

## ML rules

* Optimize unseen-generator ROC-AUC first
* Never train/tune on the hidden SIH test set
* Report AUC, macro-F1, confusion matrix, accuracy and FPR
* Calibrate confidence before user-facing predictions
* Explanations must be grounded in model evidence
* Respect SIH ethics and avoid over-claiming

## Subagent Policy

* Use Explore subagent before unfamiliar code changes
* Use Plan subagent for substantial features
* Use evaluation subagent after model/training changes
* Use XAI/review subagent for explanation work

## Commands

```bash
pip install -r requirements.txt
python -m model.training.train
python -m model.training.evaluate
pytest
uvicorn app.main:app --reload
```
## Development Boundaries

The repository contains backend, frontend, and ML components.

During the ML foundation phase:

- Do not modify backend/
- Do not modify frontend/
- Do not create API routes
- Do not create UI components
- Focus on model/, experiments/, tests/, data/, and ML configuration.

Backend and frontend integration will happen after the ML inference pipeline is stable.