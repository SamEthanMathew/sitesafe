"""
SiteSafe — Evaluation Metrics
==============================

Metric computation for model predictions vs. ground truth annotations.

A "prediction" or "ground-truth" entry is a list of (image_id, [violations])
pairs, where each violation is a dict::

    {"standard_id": "1926.501(b)(1)", "confidence": 0.92}

Ground-truth confidences default to 1.0 if missing.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

log = logging.getLogger("sitesafe.metrics")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_cfr(s: str) -> str:
    return s.strip().lower().replace(" ", "")


def _violation_set(items: Iterable[dict]) -> set[str]:
    return {_normalize_cfr(i["standard_id"]) for i in items if "standard_id" in i}


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

@dataclass
class CategoryMetrics:
    standard_id: str
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return float(self.true_positives / denom) if denom else float("nan")

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return float(self.true_positives / denom) if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not p or not r or np.isnan(p) or np.isnan(r):
            return float("nan")
        return 2.0 * p * r / (p + r)

    def as_dict(self) -> dict:
        return {
            "standard_id": self.standard_id,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
        }


def detection_recall_per_standard(
    predictions: dict[str, list[dict]],
    ground_truth: dict[str, list[dict]],
) -> dict[str, CategoryMetrics]:
    """Compute per-standard precision / recall / F1.

    Both args map ``image_id -> list of violation dicts``.
    """
    standards: set[str] = set()
    for v in ground_truth.values():
        standards.update(_violation_set(v))
    for v in predictions.values():
        standards.update(_violation_set(v))

    out: dict[str, CategoryMetrics] = {}
    for std in sorted(standards):
        tp = fp = fn = 0
        for image_id, gt_list in ground_truth.items():
            gt_set = _violation_set(gt_list)
            pred_set = _violation_set(predictions.get(image_id, []))
            std_norm = _normalize_cfr(std)
            in_gt = std_norm in gt_set
            in_pred = std_norm in pred_set
            tp += int(in_gt and in_pred)
            fp += int((not in_gt) and in_pred)
            fn += int(in_gt and (not in_pred))
        out[std] = CategoryMetrics(std, tp, fp, fn)
    return out


def overall_detection_metrics(per_std: dict[str, CategoryMetrics]) -> dict[str, float]:
    """Micro- and macro-averaged precision / recall / F1."""
    if not per_std:
        return {
            "macro_precision": float("nan"),
            "macro_recall":    float("nan"),
            "macro_f1":        float("nan"),
            "micro_precision": float("nan"),
            "micro_recall":    float("nan"),
            "micro_f1":        float("nan"),
        }

    tps = sum(m.true_positives for m in per_std.values())
    fps = sum(m.false_positives for m in per_std.values())
    fns = sum(m.false_negatives for m in per_std.values())
    micro_precision = tps / (tps + fps) if (tps + fps) else float("nan")
    micro_recall    = tps / (tps + fns) if (tps + fns) else float("nan")
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision and micro_recall and not np.isnan(micro_precision) and not np.isnan(micro_recall))
        else float("nan")
    )

    macro_precision = float(np.nanmean([m.precision for m in per_std.values()]))
    macro_recall    = float(np.nanmean([m.recall    for m in per_std.values()]))
    macro_f1        = float(np.nanmean([m.f1        for m in per_std.values()]))

    return {
        "macro_precision": macro_precision,
        "macro_recall":    macro_recall,
        "macro_f1":        macro_f1,
        "micro_precision": micro_precision,
        "micro_recall":    micro_recall,
        "micro_f1":        micro_f1,
    }


def false_positive_rate_per_standard(
    per_std: dict[str, CategoryMetrics],
    n_images: int,
) -> dict[str, float]:
    """Per-standard FPR using number of images as the negative-pool denominator."""
    if n_images == 0:
        return {std: float("nan") for std in per_std}
    return {std: m.false_positives / n_images for std, m in per_std.items()}


# ---------------------------------------------------------------------------
# Citation accuracy
# ---------------------------------------------------------------------------

def citation_accuracy(predicted_cfrs: list[str], true_cfrs: list[str]) -> float:
    """Fraction of predicted citations that exactly match the ground truth."""
    if not predicted_cfrs:
        return float("nan")
    correct = sum(
        _normalize_cfr(p) in {_normalize_cfr(t) for t in true_cfrs}
        for p in predicted_cfrs
    )
    return correct / len(predicted_cfrs)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Weighted |accuracy - confidence| across bins."""
    confidences = np.asarray(confidences, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if confidences.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = confidences.size
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(labels[mask].mean() - confidences[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    n: int
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    stdev_seconds: float

    def as_dict(self) -> dict:
        return {
            "n":               self.n,
            "mean_seconds":    self.mean_seconds,
            "median_seconds":  self.median_seconds,
            "p95_seconds":     self.p95_seconds,
            "stdev_seconds":   self.stdev_seconds,
        }


def mean_inference_time(latencies_seconds: list[float]) -> Optional[LatencyStats]:
    if not latencies_seconds:
        return None
    arr = np.asarray(latencies_seconds, dtype=float)
    return LatencyStats(
        n=arr.size,
        mean_seconds=float(arr.mean()),
        median_seconds=float(np.median(arr)),
        p95_seconds=float(np.percentile(arr, 95)) if arr.size >= 2 else float(arr[0]),
        stdev_seconds=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    )


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_long(
    predictions: dict[str, list[dict]],
    ground_truth: dict[str, list[dict]],
) -> dict[str, dict[str, int]]:
    """A simplistic long-form CM: ``{actual: {predicted: count}}``.

    For multi-label data we collapse to "any of these standards" by image,
    which is appropriate for SiteSafe because each violation is independently
    detectable.
    """
    standards: set[str] = set()
    for v in ground_truth.values():
        standards.update(_violation_set(v))
    for v in predictions.values():
        standards.update(_violation_set(v))

    matrix = {actual: {pred: 0 for pred in standards | {"none"}} for actual in standards | {"none"}}

    for image_id, gt_list in ground_truth.items():
        pred_set = _violation_set(predictions.get(image_id, []))
        gt_set = _violation_set(gt_list)
        if not gt_set and not pred_set:
            matrix["none"]["none"] += 1
            continue
        if not gt_set:
            for p in pred_set:
                matrix["none"][p] += 1
            continue
        if not pred_set:
            for g in gt_set:
                matrix[g]["none"] += 1
            continue
        for g in gt_set:
            for p in pred_set:
                matrix[g][p] += 1

    return matrix
