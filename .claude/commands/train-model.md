---
description: Train a SignalScope model using the current experiment configuration

argument-hint: "Experiment/config name"

allowed-tools: Read, Glob, Grep, Bash(python:\*), Bash(mkdir:\*), Bash(cp:\*)
---

You are a senior ML engineer training a model for SignalScope. Always follow the rules in CLAUDE.md.

User input: $ARGUMENTS

**## Step 1 — Check the experiment**

Read:

- `CLAUDE.md`
- The requested experiment configuration
- Dataset loader
- Model architecture
- Training code
- Previous experiment results

Confirm that the requested experiment is defined and that required files exist.

If the experiment cannot be identified, stop and ask the user to clarify.

**## Step 2 — Verify data separation**

Before training, verify:

- Training data is used only for training
- Validation data is used only for validation
- Hidden SIH test data is not accessed
- No test metrics influence training
- No hidden-test threshold tuning occurs

If leakage is detected, stop immediately.

**## Step 3 — Train the model**

Use the configured:

- Dataset
- Model
- Hyperparameters
- Augmentations
- Random seed
- Number of epochs
- Optimizer
- Scheduler

Do not silently change experiment parameters.

Log:

- Training loss
- Validation loss
- Validation ROC-AUC
- Validation Macro-F1
- Best checkpoint
- Training configuration

**## Step 4 — Save experiment artifacts**

Save results under:

`experiments/<experiment-name>/`

Include where applicable:

- `config.yaml`
- `metrics.json`
- `checkpoint.pt`
- `training.log`
- `experiment.md`

Do not overwrite a previous experiment without explicit instruction.

**## Step 5 — Report**

Print:

```text
Experiment: <experiment-name>
Model:      <model>
Epochs:     <epochs>
Best AUC:   <validation_auc>
Best F1:    <validation_f1>
Checkpoint: <checkpoint_path>
```

Do not claim that the model improved over previous work until `/evaluate-model` verifies the comparison.