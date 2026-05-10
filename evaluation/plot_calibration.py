"""
SiteSafe — Calibration Plot Renderer
=====================================

Standalone CLI to fit a calibrator on collected (raw_confidence, label)
pairs and render the SiteSafe reliability diagram.

Input CSV format (first row is header)::

    raw_confidence,label
    0.92,1
    0.55,0
    0.83,1
    ...

Run::

    python evaluation/plot_calibration.py \\
        --input evaluation/results/calibration_pairs.csv \\
        --calibrator app/calibrator.joblib \\
        --plot evaluation/results/reliability_diagram.png
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

# Path massaging so this module works whether you run it from repo root or
# inside the evaluation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.confidence_calibration import ConfidenceCalibrator, plot_reliability_diagram  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sitesafe.calibration_plot")


def load_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raws: list[float] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                raws.append(float(row["raw_confidence"]))
                labels.append(int(row["label"]))
            except (KeyError, ValueError) as exc:
                log.warning("Skipping bad row %r: %s", row, exc)
    return np.array(raws, dtype=float), np.array(labels, dtype=int)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV with columns raw_confidence,label.")
    parser.add_argument("--calibrator", type=Path, default=Path("app/calibrator.joblib"))
    parser.add_argument("--plot", type=Path, default=Path("evaluation/results/reliability_diagram.png"))
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args(argv)

    raws, labels = load_pairs(args.input)
    if raws.size == 0:
        raise SystemExit("No valid (raw_confidence,label) rows in input CSV.")
    log.info("Loaded %d (raw, label) pairs from %s", raws.size, args.input)

    calibrator = ConfidenceCalibrator()
    calibrator.fit(raws, labels)
    calibrator.save(args.calibrator)

    calibrated = calibrator.calibrate_batch(raws)
    raw_ece = ConfidenceCalibrator.compute_ece(raws, labels, n_bins=args.n_bins)
    cal_ece = ConfidenceCalibrator.compute_ece(calibrated, labels, n_bins=args.n_bins)
    log.info("Raw ECE = %.4f  →  Calibrated ECE = %.4f", raw_ece, cal_ece)

    plot_reliability_diagram(
        raws, calibrated, labels,
        n_bins=args.n_bins, output_path=args.plot,
    )
    log.info("Done. Calibrator → %s; diagram → %s", args.calibrator, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
