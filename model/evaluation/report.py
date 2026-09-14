"""
Evaluation run artifacts for the SignalScope evaluation pipeline.

Writes config.json (via the existing model.training.logging_utils.RunLogger,
reused rather than a second logger class), metrics.json, predictions.csv,
evaluation_report.md, and (only when explicitly enabled and available) an
optional confusion_matrix.png. See
.claude/specs/05-evaluation-pipeline.md ("Output artifacts").

Responsible Team Member: Member 6 (MLOps & Config)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from model.evaluation.metrics import SplitMetrics
from model.evaluation.predict import PredictionRow
from model.training.logging_utils import RunLogger

DEV_SHARD_MARKER = "genimage_dev"
DEV_SHARD_BANNER = "> Tiny-GenImage development shard - NOT the final SIH benchmark."

PREDICTIONS_CSV_COLUMNS = (
    "image_path",
    "true_label",
    "predicted_probability",
    "predicted_label",
    "generator",
    "split",
)


def is_dev_shard(manifest_path: Union[str, Path]) -> bool:
    return DEV_SHARD_MARKER in str(manifest_path)


def write_config(run_dir: Union[str, Path], config: Dict[str, Any]) -> None:
    """Writes the full reproducibility record via the existing RunLogger."""
    RunLogger(run_dir).write_config(config)


def write_predictions_csv(rows: List[PredictionRow], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PREDICTIONS_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _split_block(overall: SplitMetrics, per_generator: Dict[str, SplitMetrics]) -> Dict[str, Any]:
    return {
        "overall": overall.to_dict(),
        "per_generator": {generator: metrics.to_dict() for generator, metrics in per_generator.items()},
    }


def build_metrics_payload(
    val_overall: SplitMetrics,
    val_per_generator: Dict[str, SplitMetrics],
    test_overall: SplitMetrics,
    test_per_generator: Dict[str, SplitMetrics],
) -> Dict[str, Any]:
    """Assembles the full metrics.json payload: seen-generator (val) and
    unseen-generator (test) blocks, each with an overall result and a
    per-generator breakdown."""
    return {
        "val": _split_block(val_overall, val_per_generator),
        "test": _split_block(test_overall, test_per_generator),
    }


def write_metrics_json(payload: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _format_metric(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_split_section(title: str, block: Dict[str, Any]) -> List[str]:
    overall = block["overall"]
    lines = [f"## {title}", ""]
    lines.append(f"- num_samples: {overall['num_samples']} (real={overall['num_real']}, fake={overall['num_fake']})")
    lines.append(f"- ROC-AUC: {_format_metric(overall['roc_auc'])}")
    lines.append(f"- Accuracy: {_format_metric(overall['accuracy'])}")
    lines.append(f"- Macro-F1: {_format_metric(overall['macro_f1'])}")
    lines.append(f"- FPR @ threshold={overall['threshold']}: {_format_metric(overall['fpr'])}")
    cm = overall["confusion_matrix"]
    lines.append(f"- Confusion matrix: TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")
    for note in overall.get("notes", []):
        lines.append(f"- Note: {note}")
    lines.append("")

    per_generator = block["per_generator"]
    if per_generator:
        lines.append(f"### Per-generator ({title})")
        lines.append("")
        lines.append("| generator | num_samples | num_fake | ROC-AUC | Accuracy | Macro-F1 |")
        lines.append("|---|---|---|---|---|---|")
        for generator in sorted(per_generator):
            metrics = per_generator[generator]
            lines.append(
                f"| {generator} | {metrics['num_samples']} | {metrics['num_fake']} | "
                f"{_format_metric(metrics['roc_auc'])} | {_format_metric(metrics['accuracy'])} | "
                f"{_format_metric(metrics['macro_f1'])} |"
            )
        lines.append("")
    return lines


def write_evaluation_report_md(
    payload: Dict[str, Any],
    manifest_path: Union[str, Path],
    path: Union[str, Path],
) -> None:
    """Writes the human-readable summary. Headline is the unseen-generator
    (test) ROC-AUC; a dev-shard disclaimer banner is prepended whenever the
    manifest is the Tiny-GenImage development shard - not optional."""
    lines: List[str] = ["# SignalScope Evaluation Report", ""]
    if is_dev_shard(manifest_path):
        lines.append(DEV_SHARD_BANNER)
        lines.append("")

    test_roc_auc = payload["test"]["overall"]["roc_auc"]
    lines.append(f"**Headline (unseen-generator ROC-AUC): {_format_metric(test_roc_auc)}**")
    lines.append("")

    lines.extend(_render_split_section("Unseen-generator test performance", payload["test"]))
    lines.extend(_render_split_section("Seen-generator validation performance", payload["val"]))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def maybe_write_confusion_matrix_plot(
    confusion: Dict[str, int],
    path: Union[str, Path],
    enabled: bool,
) -> bool:
    """Renders confusion_matrix.png via matplotlib, only when `enabled` and
    matplotlib is importable. Returns True if the plot was written, False
    otherwise (never raises just because matplotlib is absent - it is an
    optional dependency, see .claude/specs/05-evaluation-pipeline.md
    "Dependencies")."""
    if not enabled:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    matrix = [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]]
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["real", "ai_generated"])
    ax.set_yticks([0, 1], labels=["real", "ai_generated"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True
