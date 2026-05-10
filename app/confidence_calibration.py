"""
SiteSafe — Confidence Calibration
==================================

Raw LLM-emitted confidences are notoriously over-confident. For a
safety-critical application that's a problem: a 0.95 from the model and a
0.95 from a calibrated probability should mean the same thing — that 95 %
of similarly-labelled detections are true positives.

We use **Platt scaling** (logistic regression on a single feature: the raw
confidence). It's the simplest reliable calibrator for binary-style problems,
which is what each violation detection effectively is (true positive vs. not).

Pipeline::

    raw_confidences, true_labels = collect_validation_pairs()
    cal = ConfidenceCalibrator()
    cal.fit(raw_confidences, true_labels)
    cal.save("app/calibrator.joblib")

    # at inference time
    cal = ConfidenceCalibrator.load("app/calibrator.joblib")
    calibrated = cal.calibrate(raw_value)

We also expose:
    * ``compute_ece`` — Expected Calibration Error
    * ``plot_reliability_diagram`` — before/after diagram for the writeup

Tip: target ECE < 0.10 for a safety-critical readout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("sitesafe.calibration")


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

@dataclass
class _FitState:
    n_samples: int
    base_rate: float


class ConfidenceCalibrator:
    """Platt-scaling calibrator for SiteSafe violation confidences.

    The fit is logistic-regression with a single feature (the raw confidence)
    and an intercept. ``calibrate(x)`` returns ``sigmoid(a*x + b)``.
    """

    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression

        self._lr = LogisticRegression(C=1.0, solver="lbfgs")
        self.is_fitted: bool = False
        self.fit_state: Optional[_FitState] = None

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, raw_confidences: np.ndarray, true_labels: np.ndarray) -> None:
        raw = np.asarray(raw_confidences, dtype=float).reshape(-1, 1)
        labels = np.asarray(true_labels, dtype=int)
        if raw.shape[0] != labels.shape[0]:
            raise ValueError("raw_confidences and true_labels must have the same length.")
        if raw.shape[0] < 8:
            raise ValueError("At least 8 calibration samples required for stable Platt scaling.")
        if len(np.unique(labels)) < 2:
            raise ValueError(
                "Calibration set must contain both positives and negatives "
                "(true and false detections)."
            )

        self._lr.fit(raw, labels)
        self.is_fitted = True
        self.fit_state = _FitState(
            n_samples=int(raw.shape[0]),
            base_rate=float(labels.mean()),
        )
        log.info(
            "Fit Platt scaler on %d samples (base rate=%.3f).",
            self.fit_state.n_samples, self.fit_state.base_rate,
        )

    def calibrate(self, raw_confidence: float) -> float:
        if not self.is_fitted:
            return float(np.clip(raw_confidence, 0.0, 1.0))
        x = np.array([[raw_confidence]], dtype=float)
        return float(self._lr.predict_proba(x)[0, 1])

    def calibrate_batch(self, raw_confidences: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.clip(np.asarray(raw_confidences, dtype=float), 0.0, 1.0)
        x = np.asarray(raw_confidences, dtype=float).reshape(-1, 1)
        return self._lr.predict_proba(x)[:, 1]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ece(
        confidences: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error — weighted |accuracy - confidence| per bin."""
        confidences = np.asarray(confidences, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if confidences.size == 0:
            return float("nan")
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = confidences.size
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences <= hi)
            if not mask.any():
                continue
            bin_conf = confidences[mask].mean()
            bin_acc = labels[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        return float(ece)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "lr": self._lr,
                "is_fitted": self.is_fitted,
                "fit_state": self.fit_state,
            },
            path,
        )
        log.info("Saved calibrator → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceCalibrator":
        import joblib

        payload = joblib.load(Path(path))
        instance = cls()
        instance._lr = payload["lr"]
        instance.is_fitted = bool(payload.get("is_fitted", True))
        instance.fit_state = payload.get("fit_state")
        return instance


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    raw_confidences: np.ndarray,
    calibrated_confidences: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
    output_path: str | Path = "evaluation/results/reliability_diagram.png",
    title: str = "SiteSafe — Reliability Diagram",
) -> Path:
    """Render a 2-panel matplotlib figure: raw vs. calibrated reliability."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    raw = np.asarray(raw_confidences, dtype=float)
    cal = np.asarray(calibrated_confidences, dtype=float)
    y = np.asarray(labels, dtype=int)

    raw_frac, raw_mean = calibration_curve(y, raw, n_bins=n_bins, strategy="uniform")
    cal_frac, cal_mean = calibration_curve(y, cal, n_bins=n_bins, strategy="uniform")

    raw_ece = ConfidenceCalibrator.compute_ece(raw, y, n_bins=n_bins)
    cal_ece = ConfidenceCalibrator.compute_ece(cal, y, n_bins=n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    for ax, mean, frac, ece, label in (
        (axes[0], raw_mean, raw_frac, raw_ece, "Raw model confidence"),
        (axes[1], cal_mean, cal_frac, cal_ece, "Platt-calibrated"),
    ):
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
        ax.plot(mean, frac, marker="o", linewidth=2, label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction of true positives")
        ax.set_title(f"{label} — ECE = {ece:.3f}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved reliability diagram → %s", out)
    return out


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Tiny synthetic demo: confirm the calibrator improves ECE on synthetic data."""
    rng = np.random.default_rng(0)
    n = 400
    # Synthetic over-confident model: scores are "true_score + 0.15" clipped.
    true_scores = rng.beta(2, 2, n)
    raw = np.clip(true_scores + 0.15, 0.0, 0.99)
    labels = (rng.uniform(size=n) < true_scores).astype(int)

    cal = ConfidenceCalibrator()
    cal.fit(raw, labels)

    raw_ece = ConfidenceCalibrator.compute_ece(raw, labels)
    cal_ece = ConfidenceCalibrator.compute_ece(cal.calibrate_batch(raw), labels)
    print(f"Synthetic ECE: raw={raw_ece:.3f}  →  calibrated={cal_ece:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _demo()
