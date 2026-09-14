---
description: Design and run a controlled SignalScope ML experiment

argument-hint: "Experiment hypothesis e.g. RGB plus FFT"

allowed-tools: Read, Write, Glob, Grep, Bash(python:\*), Bash(mkdir:\*)
---

You are a senior ML research engineer running a controlled SignalScope experiment. Always follow the rules in CLAUDE.md.

User input: $ARGUMENTS

**## Step 1 — Understand the hypothesis**

From $ARGUMENTS identify:

- Experiment name
- Hypothesis
- Component being changed
- Expected effect

If the hypothesis is unclear, ask the user to clarify.

**## Step 2 — Inspect previous work**

Read:

- `CLAUDE.md`
- Existing model architecture
- Dataset configuration
- Training configuration
- Previous experiment results

Identify the strongest valid baseline.

**## Step 3 — Define the experiment**

Create:

`experiments/<experiment-name>/experiment.md`

Include:

- Hypothesis
- Baseline
- Changed variable
- Dataset
- Data split
- Model
- Hyperparameters
- Expected result
- Success metric
- Risks

Only change the variable required to test the hypothesis.

**## Step 4 — Run the experiment**

Train the model using the controlled configuration.

Do not access the hidden SIH test set.

Record:

- Training configuration
- Validation results
- Runtime
- Checkpoint
- Any failures

**## Step 5 — Evaluate**

Run the required evaluation.

Prioritize:

- Unseen-generator ROC-AUC
- Overall ROC-AUC
- Macro-F1
- Accuracy
- FPR
- Confusion matrix
- Calibration / robustness where applicable

**## Step 6 — Record the conclusion**

Update the experiment record with:

- Final metrics
- Baseline metrics
- Metric deltas
- Whether the hypothesis was supported
- Observed failure cases
- Recommended next experiment

**## Step 7 — Report**

Print:

```text
Hypothesis: <hypothesis>
Result:     <supported / unsupported / inconclusive>

Overall AUC: <value>
Unseen AUC:  <value>
Delta:       <value>

Conclusion:
<short conclusion>

Next:
<recommended experiment>
```