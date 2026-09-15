# Experiments

Per-run training artifacts, one directory per run under `experiments/runs/<run_id>/`:

```text
experiments/runs/<run_id>/
├── config.json        # full resolved run configuration (model + training + data split)
├── metrics.jsonl       # one JSON line per epoch: train/val loss, val accuracy, val ROC-AUC, LR
└── checkpoints/
    ├── last.pt          # most recent epoch's full training checkpoint (resume-on-crash)
    └── best.pt          # checkpoint with the best validation ROC-AUC so far
```

`experiments/runs/` is git-ignored (checkpoints and per-run logs are local
build artifacts, not source). This `README.md` documents the layout so it
does not need to live in a tracked example run.

See `.claude/specs/04-training-pipeline.md` for the full training pipeline
design, including why checkpoint selection uses validation ROC-AUC only and
why the unseen-generator `test` split is never touched by training.
