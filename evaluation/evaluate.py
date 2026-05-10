"""
SiteSafe — End-to-End Evaluation
=================================

Runs SiteSafe over a labelled test set and produces:

* ``results/metrics.json``        — full numerical results
* ``results/results.md``          — human-readable summary table
* ``results/calibration_pairs.csv`` — input for ``plot_calibration.py``
* ``results/confusion_matrix.json`` — long-form per-standard confusion

Test-set format
---------------

A JSONL file where each line is::

    {
      "image_path": "data/eval/site_001.jpg",
      "violations": [
        {"standard_id": "1926.501(b)(1)"},
        {"standard_id": "1926.100(a)"}
      ]
    }

A row with an empty ``violations`` list represents a compliant site.

Run::

    python evaluation/evaluate.py \\
        --test-jsonl evaluation/data/test.jsonl \\
        --output-dir evaluation/results \\
        --model sitesafe
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterable

# Make sibling imports work regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference import (  # noqa: E402
    InferenceResult,
    ModelNotFoundError,
    OllamaUnavailableError,
    run_inference,
)
from evaluation.metrics import (  # noqa: E402
    citation_accuracy,
    confusion_matrix_long,
    detection_recall_per_standard,
    expected_calibration_error,
    false_positive_rate_per_standard,
    mean_inference_time,
    overall_detection_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sitesafe.evaluate")


# ---------------------------------------------------------------------------
# Report parsing — we extract (standard_id, confidence) pairs from the model's
# Markdown output. Robust to small format drift.
# ---------------------------------------------------------------------------

_STD_RE = re.compile(
    r"\*\*Regulation:\*\*\s*(?:29\s*CFR\s*)?(?P<std>1926[\.\(\)\w-]+)",
    re.IGNORECASE,
)
_CONF_RE = re.compile(
    r"\*\*Confidence:\*\*\s*(?P<val>[0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
_VIOL_BLOCK_RE = re.compile(
    r"### Violation\s+\d+:.*?(?=^### |\Z)",
    re.DOTALL | re.MULTILINE,
)


def parse_predictions(report_md: str) -> list[dict]:
    """Extract a list of {standard_id, confidence} dicts from a SiteSafe report."""
    blocks = _VIOL_BLOCK_RE.findall(report_md)
    out: list[dict] = []
    for blk in blocks:
        std_match = _STD_RE.search(blk)
        conf_match = _CONF_RE.search(blk)
        if not std_match:
            continue
        std = std_match.group("std").strip().rstrip(",.")
        try:
            conf = float(conf_match.group("val")) if conf_match else 0.5
        except ValueError:
            conf = 0.5
        out.append({"standard_id": std, "confidence": min(max(conf, 0.0), 1.0)})
    return out


# ---------------------------------------------------------------------------
# Test set loading
# ---------------------------------------------------------------------------

def load_test_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Test JSONL not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d: %s", line_no, exc)
    return rows


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    test_rows: list[dict],
    *,
    model: str,
    repo_root: Path,
    timeout_seconds: float = 180.0,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], list[dict], list[float]]:
    predictions: dict[str, list[dict]] = {}
    ground_truth: dict[str, list[dict]] = {}
    cal_pairs: list[dict] = []
    latencies: list[float] = []

    for i, row in enumerate(test_rows, start=1):
        image_path = (repo_root / row["image_path"]).resolve()
        image_id = row.get("image_id", str(image_path))
        gt_violations = row.get("violations", [])
        ground_truth[image_id] = gt_violations
        gt_std_set = {v["standard_id"] for v in gt_violations}

        log.info("[%d/%d] %s", i, len(test_rows), image_path)
        try:
            result: InferenceResult = run_inference(
                image_path,
                model=model,
                site_name=row.get("site_name", ""),
                location=row.get("location", ""),
                date=row.get("date", ""),
                timeout_seconds=timeout_seconds,
            )
        except (OllamaUnavailableError, ModelNotFoundError) as exc:
            raise SystemExit(str(exc))
        except Exception as exc:  # noqa: BLE001
            log.warning("Inference failed for %s: %s", image_path, exc)
            predictions[image_id] = []
            continue

        latencies.append(result.latency_seconds)
        pred_violations = parse_predictions(result.text)
        predictions[image_id] = pred_violations

        for pv in pred_violations:
            cal_pairs.append({
                "image_id":       image_id,
                "raw_confidence": pv["confidence"],
                "label":          int(pv["standard_id"] in gt_std_set),
            })

    return predictions, ground_truth, cal_pairs, latencies


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def render_markdown(metrics: dict) -> str:
    lines: list[str] = []
    overall = metrics["overall"]
    lat = metrics.get("latency") or {}

    lines.append("# SiteSafe — Evaluation Results\n")
    lines.append("## Overall\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Macro recall    | {overall['macro_recall']:.3f} |")
    lines.append(f"| Macro precision | {overall['macro_precision']:.3f} |")
    lines.append(f"| Macro F1        | {overall['macro_f1']:.3f} |")
    lines.append(f"| Micro recall    | {overall['micro_recall']:.3f} |")
    lines.append(f"| Micro precision | {overall['micro_precision']:.3f} |")
    lines.append(f"| Micro F1        | {overall['micro_f1']:.3f} |")
    lines.append(f"| Citation accuracy | {metrics['citation_accuracy']:.3f} |")
    lines.append(f"| Raw ECE         | {metrics['raw_ece']:.3f} |")
    if lat:
        lines.append(
            f"| Mean latency    | {lat['mean_seconds']:.2f} s "
            f"(median {lat['median_seconds']:.2f} s, p95 {lat['p95_seconds']:.2f} s, n={lat['n']}) |"
        )
    lines.append("")

    lines.append("## Per-Standard\n")
    lines.append("| CFR | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for std, m in sorted(metrics["per_standard"].items()):
        lines.append(
            f"| {std} | {m['true_positives']} | {m['false_positives']} | "
            f"{m['false_negatives']} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} | {m['f1']:.3f} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SiteSafe on a labelled test set.")
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--model", default=None,
                        help="Ollama model name (default: $SITESAFE_MODEL or 'sitesafe').")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_rows = load_test_jsonl(args.test_jsonl)
    log.info("Loaded %d test rows", len(test_rows))

    predictions, ground_truth, cal_pairs, latencies = run_eval(
        test_rows, model=args.model or "sitesafe",
        repo_root=repo_root, timeout_seconds=args.timeout,
    )

    per_standard = detection_recall_per_standard(predictions, ground_truth)
    overall = overall_detection_metrics(per_standard)
    fpr_per_std = false_positive_rate_per_standard(per_standard, n_images=len(test_rows))
    confusion = confusion_matrix_long(predictions, ground_truth)

    citation_acc = float("nan")
    pred_cfrs: list[str] = []
    true_cfrs: list[str] = []
    for image_id, preds in predictions.items():
        for p in preds:
            pred_cfrs.append(p["standard_id"])
        for g in ground_truth.get(image_id, []):
            true_cfrs.append(g["standard_id"])
    if pred_cfrs:
        citation_acc = citation_accuracy(pred_cfrs, true_cfrs)

    raw_confidences = [p["raw_confidence"] for p in cal_pairs]
    raw_labels = [p["label"] for p in cal_pairs]
    raw_ece = expected_calibration_error(raw_confidences, raw_labels) if raw_confidences else float("nan")

    latency_stats = mean_inference_time(latencies)

    metrics = {
        "model":             args.model or "sitesafe",
        "n_images":          len(test_rows),
        "n_predictions":     len(pred_cfrs),
        "n_ground_truth":    len(true_cfrs),
        "overall":           overall,
        "per_standard":      {std: m.as_dict() for std, m in per_standard.items()},
        "false_positive_rate": fpr_per_std,
        "citation_accuracy": citation_acc,
        "raw_ece":           raw_ece,
        "latency":           latency_stats.as_dict() if latency_stats else None,
        "ran_at":            time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (args.output_dir / "results.md").write_text(render_markdown(metrics), encoding="utf-8")
    (args.output_dir / "confusion_matrix.json").write_text(
        json.dumps(confusion, indent=2), encoding="utf-8"
    )

    cal_csv = args.output_dir / "calibration_pairs.csv"
    with cal_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "raw_confidence", "label"])
        for p in cal_pairs:
            writer.writerow([p["image_id"], f"{p['raw_confidence']:.4f}", p["label"]])

    log.info("Wrote %s", args.output_dir / "metrics.json")
    log.info("Wrote %s", args.output_dir / "results.md")
    log.info("Wrote %s", args.output_dir / "confusion_matrix.json")
    log.info("Wrote %s", cal_csv)
    log.info("Next: python evaluation/plot_calibration.py --input %s", cal_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
