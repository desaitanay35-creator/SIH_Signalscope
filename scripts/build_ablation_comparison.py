"""
Step 7 ablation comparison-assembly script.

Reads three already-completed training + evaluation runs (rgb_only,
frequency_only, rgb_frequency_fusion - produced by the existing,
unmodified `model.training.train` / `model.evaluation.evaluate`
entrypoints) and assembles the comparison artifacts defined in
.claude/specs/07-ablation-experiment.md (Sections 8-10): comparison.md,
comparison.json, run_manifest.json.

This script does NOT train or evaluate anything itself - it only reads
already-written metrics.json/config.json files under experiments/runs/
and never re-runs evaluation with different parameters after inspecting a
result (see the spec's Section 16 acceptance criteria). It is not part of
model/ - it is a one-off experiment-orchestration tool, not reusable
pipeline code.

Usage:
    python -m scripts.build_ablation_comparison \\
        --rgb-only-run <run_id> \\
        --frequency-only-run <run_id> \\
        --fusion-run <run_id> \\
        [--runs-dir experiments/runs] \\
        [--output-dir experiments/ablation/<ablation_run_id>]

Verifies, before writing anything, that the three runs actually share one
resolved split (per the spec's Section 11/12) and that each checkpoint's
recorded architecture matches its intended model_type - raising a clear,
named error and refusing to produce a comparison artifact otherwise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXPECTED_ARCHITECTURE = {
    "rgb_only": "efficientnet_b4",
    "frequency_only": "frequency_branch",
    "rgb_frequency_fusion": "rgb_frequency_fusion",
}

# Verbatim (must match model/evaluation/evaluate.py::REAL_PAIRING_NOTE in
# substance) - reproduced here as a fallback label only; the actual note
# text is always read from each run's own metrics.json, never hardcoded
# into the comparison in place of the real one.
UNSEEN_METRIC_LABEL = "Unseen-generator ROC-AUC (development evaluation)"

DEV_SHARD_BANNER = "> Tiny-GenImage development shard - NOT the final SIH benchmark."


class AblationVerificationError(ValueError):
    """Raised when the three runs do not satisfy the reproducibility
    requirements in .claude/specs/07-ablation-experiment.md (Sections 11,
    12) - the comparison must not be built from mismatched runs."""


def _latest_evaluation_dir(training_run_dir: Path) -> Path:
    evaluation_root = training_run_dir / "evaluation"
    if not evaluation_root.is_dir():
        raise AblationVerificationError(f"No evaluation/ directory found under {training_run_dir}.")
    eval_dirs = sorted(p for p in evaluation_root.iterdir() if p.is_dir())
    if not eval_dirs:
        raise AblationVerificationError(f"No evaluation run found under {evaluation_root}.")
    return eval_dirs[-1]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise AblationVerificationError(f"Expected file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class ExperimentRecord:
    def __init__(self, model_type: str, training_run_dir: Path):
        self.model_type = model_type
        self.training_run_dir = training_run_dir
        self.training_config = _load_json(training_run_dir / "config.json")
        self.eval_dir = _latest_evaluation_dir(training_run_dir)
        self.eval_config = _load_json(self.eval_dir / "config.json")
        self.metrics = _load_json(self.eval_dir / "metrics.json")

    @property
    def run_id(self) -> str:
        return self.training_config.get("run_id", self.training_run_dir.name)

    @property
    def eval_run_id(self) -> str:
        return self.eval_config.get("eval_run_id", self.eval_dir.name)

    @property
    def architecture(self) -> Optional[str]:
        return (self.training_config.get("model_config") or {}).get("architecture")

    @property
    def checkpoint_path(self) -> str:
        return str(self.eval_config.get("checkpoint_path", ""))

    @property
    def manifest_path(self) -> str:
        return str(self.eval_config.get("manifest_path", ""))

    @property
    def split_config(self) -> Dict[str, Any]:
        return self.eval_config.get("split_config", {})

    @property
    def final_val_loss(self) -> Optional[float]:
        """The last epoch's val_loss from this training run's
        metrics.jsonl, if present."""
        metrics_path = self.training_run_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            return None
        last_line = None
        with open(metrics_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last_line = line
        if last_line is None:
            return None
        return json.loads(last_line).get("val_loss")


def _verify(records: Dict[str, ExperimentRecord]) -> None:
    """Raises AblationVerificationError on the first violation found -
    matching .claude/specs/07-ablation-experiment.md, Section 12."""
    for model_type, record in records.items():
        expected = EXPECTED_ARCHITECTURE[model_type]
        if record.architecture != expected:
            raise AblationVerificationError(
                f"{model_type} run {record.run_id!r} declares architecture "
                f"{record.architecture!r}, expected {expected!r}. See Section 12."
            )

    split_configs = {model_type: record.split_config for model_type, record in records.items()}
    reference = next(iter(split_configs.values()))
    for model_type, split_config in split_configs.items():
        if split_config != reference:
            raise AblationVerificationError(
                "Resolved split_config differs between experiments - the three runs did not "
                f"share one split (see Section 11/12). {model_type} has {split_config!r}, "
                f"expected {reference!r}."
            )

    sample_counts = {
        model_type: (record.eval_config.get("num_val_samples"), record.eval_config.get("num_test_samples"))
        for model_type, record in records.items()
    }
    reference_counts = next(iter(sample_counts.values()))
    for model_type, counts in sample_counts.items():
        if counts != reference_counts:
            raise AblationVerificationError(
                f"num_val_samples/num_test_samples differ for {model_type}: {counts!r} vs. "
                f"expected {reference_counts!r} (Section 12)."
            )

    manifest_paths = {record.manifest_path for record in records.values()}
    if len(manifest_paths) != 1:
        raise AblationVerificationError(f"The three runs used different manifests: {manifest_paths!r}.")


def _is_dev_shard(manifest_path: str) -> bool:
    return "genimage_dev" in manifest_path


def _overall(metrics: Dict[str, Any], split: str) -> Dict[str, Any]:
    return metrics[split]["overall"]


def _per_generator(metrics: Dict[str, Any], split: str) -> Dict[str, Any]:
    return metrics[split]["per_generator"]


def _real_pairing_note(overall: Dict[str, Any]) -> Optional[str]:
    for note in overall.get("notes", []):
        if "real" in note.lower() and "val" in note.lower():
            return note
    return None


def build_comparison(records: Dict[str, ExperimentRecord]) -> Dict[str, Any]:
    """Builds the comparison.json payload (Sections 9-10). Does not
    write anything - pure data assembly."""
    experiments: List[Dict[str, Any]] = []
    for model_type in ("rgb_only", "frequency_only", "rgb_frequency_fusion"):
        record = records[model_type]
        test_overall = _overall(record.metrics, "test")
        val_overall = _overall(record.metrics, "val")

        experiments.append(
            {
                "experiment": model_type,
                "run_id": record.run_id,
                "eval_run_id": record.eval_run_id,
                "checkpoint_path": record.checkpoint_path,
                "architecture": record.architecture,
                "val_roc_auc": val_overall.get("roc_auc"),
                "val_loss": record.final_val_loss,
                "unseen_generator_roc_auc_dev_eval": test_overall.get("roc_auc"),
                "accuracy": test_overall.get("accuracy"),
                "macro_f1": test_overall.get("macro_f1"),
                "fpr": test_overall.get("fpr"),
                "confusion_matrix": test_overall.get("confusion_matrix"),
                "num_val_samples": record.eval_config.get("num_val_samples"),
                "num_test_samples": record.eval_config.get("num_test_samples"),
                "num_unseen_generator_eval_samples": record.eval_config.get("num_unseen_generator_eval_samples"),
                "real_image_pairing_note": _real_pairing_note(test_overall),
                "per_generator_unseen": _per_generator(record.metrics, "test"),
                "per_generator_seen_validation": _per_generator(record.metrics, "val"),
                "notes": test_overall.get("notes", []),
            }
        )

    is_dev = any(_is_dev_shard(record.manifest_path) for record in records.values())
    return {
        "metric_label": UNSEEN_METRIC_LABEL,
        "dataset_is_dev_shard": is_dev,
        "manifest_path": next(iter(records.values())).manifest_path,
        "split_config": next(iter(records.values())).split_config,
        "experiments": experiments,
        "generated_at": time.time(),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = ["# Step 7 Ablation Comparison", ""]
    if payload["dataset_is_dev_shard"]:
        lines.append(DEV_SHARD_BANNER)
        lines.append("")

    lines.append(
        f"| Experiment | Val ROC-AUC | {payload['metric_label']} | Accuracy | Macro-F1 | FPR |"
    )
    lines.append("|---|---|---|---|---|---|")
    for exp in payload["experiments"]:
        lines.append(
            f"| {exp['experiment']} | {_fmt(exp['val_roc_auc'])} | "
            f"{_fmt(exp['unseen_generator_roc_auc_dev_eval'])} | {_fmt(exp['accuracy'])} | "
            f"{_fmt(exp['macro_f1'])} | {_fmt(exp['fpr'])} |"
        )
    lines.append("")

    real_pairing_note = next(
        (exp["real_image_pairing_note"] for exp in payload["experiments"] if exp["real_image_pairing_note"]),
        None,
    )
    if real_pairing_note:
        lines.append(f"> *{real_pairing_note}*")
        lines.append("")

    lines.append("## Extended metrics")
    lines.append("")
    lines.append("| Experiment | Val Loss | Confusion Matrix | Num Val | Num Test | Num Unseen-Eval |")
    lines.append("|---|---|---|---|---|---|")
    for exp in payload["experiments"]:
        cm = exp["confusion_matrix"] or {}
        cm_str = f"TP={cm.get('tp')} TN={cm.get('tn')} FP={cm.get('fp')} FN={cm.get('fn')}"
        lines.append(
            f"| {exp['experiment']} | {_fmt(exp['val_loss'])} | {cm_str} | "
            f"{exp['num_val_samples']} | {exp['num_test_samples']} | {exp['num_unseen_generator_eval_samples']} |"
        )
    lines.append("")

    lines.append(f"## Per-generator {payload['metric_label']}")
    lines.append("")
    unseen_generators = sorted(payload["experiments"][0]["per_generator_unseen"])
    lines.append("| Generator (unseen) | " + " | ".join(exp["experiment"] for exp in payload["experiments"]) + " |")
    lines.append("|---|" + "---|" * len(payload["experiments"]))
    for generator in unseen_generators:
        row = [generator]
        for exp in payload["experiments"]:
            entry = exp["per_generator_unseen"].get(generator, {})
            row.append(_fmt(entry.get("roc_auc")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Per-generator seen-generator (validation) ROC-AUC")
    lines.append("")
    seen_generators = sorted(payload["experiments"][0]["per_generator_seen_validation"])
    lines.append("| Generator (seen) | " + " | ".join(exp["experiment"] for exp in payload["experiments"]) + " |")
    lines.append("|---|" + "---|" * len(payload["experiments"]))
    for generator in seen_generators:
        row = [generator]
        for exp in payload["experiments"]:
            entry = exp["per_generator_seen_validation"].get(generator, {})
            row.append(_fmt(entry.get("roc_auc")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines) + "\n"


def run(
    runs_dir: str,
    rgb_only_run: str,
    frequency_only_run: str,
    fusion_run: str,
    output_dir: str,
) -> Dict[str, Any]:
    runs_root = Path(runs_dir)
    records = {
        "rgb_only": ExperimentRecord("rgb_only", runs_root / rgb_only_run),
        "frequency_only": ExperimentRecord("frequency_only", runs_root / frequency_only_run),
        "rgb_frequency_fusion": ExperimentRecord("rgb_frequency_fusion", runs_root / fusion_run),
    }
    _verify(records)

    payload = build_comparison(records)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "comparison.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    with open(output_path / "comparison.md", "w", encoding="utf-8") as fh:
        fh.write(render_markdown(payload))

    run_manifest = {
        model_type: {"run_id": record.run_id, "eval_run_id": record.eval_run_id, "checkpoint_path": record.checkpoint_path}
        for model_type, record in records.items()
    }
    run_manifest["verification"] = "passed"
    with open(output_path / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(run_manifest, fh, indent=2, default=str)

    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble the Step 7 ablation comparison artifacts.")
    parser.add_argument("--runs-dir", default="experiments/runs")
    parser.add_argument("--rgb-only-run", required=True)
    parser.add_argument("--frequency-only-run", required=True)
    parser.add_argument("--fusion-run", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(
        runs_dir=args.runs_dir,
        rgb_only_run=args.rgb_only_run,
        frequency_only_run=args.frequency_only_run,
        fusion_run=args.fusion_run,
        output_dir=args.output_dir,
    )
    print(f"Comparison written to {args.output_dir}")


if __name__ == "__main__":
    main()
