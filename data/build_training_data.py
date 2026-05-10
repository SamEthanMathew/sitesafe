"""
SiteSafe — Training Data Pipeline
==================================

Constructs the SFT (supervised fine-tuning) dataset that is fed to Unsloth's
``SFTTrainer`` (see ``training/finetune_gemma4_e4b.py``).

Each training example is a single conversation in Unsloth's "messages"
format::

    {
      "messages": [
        {"role": "user",      "content": [{"type": "image", "image": <path>},
                                          {"type": "text",  "text":  <prompt>}]},
        {"role": "assistant", "content":  <structured violation report>}
      ]
    }

Three operating modes
---------------------

``--mode roboflow``
    Walk a Roboflow-style YOLO export (``images/`` + ``labels/`` with class IDs).
    Map detected classes (``NO-Hardhat``, ``NO-Safety Vest``, etc.) to OSHA
    standards using ``CLASS_TO_VIOLATION``.

``--mode csv``
    Read manual annotations from a CSV (``image_path,violations``) where
    ``violations`` is a ``;``-separated list of CFR standard IDs (e.g.
    ``1926.501(b)(1);1926.100(a)``). Use this for the highest-quality slice
    of the dataset.

``--mode template``
    Emit one example per image with a placeholder report that a human can
    edit before training. Useful when you want to rapidly bootstrap labels.

Output is JSONL at the path passed to ``--output`` (default
``data/training/train.jsonl``).

Run::

    python data/build_training_data.py --mode roboflow \\
        --images-dir data/datasets/roboflow/train/images \\
        --labels-dir data/datasets/roboflow/train/labels \\
        --classes-file data/datasets/roboflow/data.yaml \\
        --output data/training/train.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sitesafe.train_data")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "osha_regulations.db"

# Map Roboflow Construction Site Safety (and similar) class names to OSHA
# regulation IDs. Names are lower-cased and stripped during lookup so casing
# doesn't matter at the call site.
CLASS_TO_VIOLATION: dict[str, str] = {
    "no-hardhat": "1926.100(a)",
    "no_hardhat": "1926.100(a)",
    "nohardhat":  "1926.100(a)",
    "no-helmet":  "1926.100(a)",

    "no-safety vest": "1926.95(a)",
    "no-safety-vest": "1926.95(a)",
    "no_vest":        "1926.95(a)",

    "no-mask":   "1926.95(a)",   # general PPE — masking enforced under the same provision

    "person-no-hardhat-no-vest": "1926.100(a)",  # combined detection — choose the most severe

    # Edge / scaffold detections (when class taxonomy supports them)
    "person-at-edge":     "1926.501(b)(1)",
    "scaffold-no-rail":   "1926.451(g)(4)",
    "ladder-misuse":      "1926.1053(b)(1)",
    "open-hole":          "1926.501(b)(4)",
    "exposed-wire":       "1926.405(g)(2)(iv)",
    "open-trench":        "1926.652(a)(1)",
}

# These classes (from PPE-detection datasets) indicate compliant PPE — used to
# count "looks compliant" frames so we can balance the dataset with negatives.
COMPLIANT_CLASSES = {
    "hardhat", "helmet", "safety vest", "safety-vest", "vest", "mask", "person",
}

DEFAULT_USER_PROMPT = (
    "Analyze this construction site photo for OSHA safety violations. "
    "Identify all hazards visible in the image, cite the specific 29 CFR 1926 "
    "regulation, assess severity (Other/Serious/Willful), provide a confidence "
    "score from 0.0 to 1.0, and recommend specific corrective actions. "
    "If the site appears compliant, say so explicitly and note the limitations of "
    "photo-based assessment."
)


# ---------------------------------------------------------------------------
# OSHA DB helper (kept local so this script doesn't require ``app/`` on path)
# ---------------------------------------------------------------------------

@dataclass
class RegulationRow:
    standard_id: str
    title: str
    requirement_text: str
    violation_type: str
    min_penalty: int
    max_penalty: int
    corrective_action: str
    fatal_four_category: Optional[str]
    visual_indicators: str


def load_regulations(db_path: Path = DB_PATH) -> dict[str, RegulationRow]:
    if not db_path.exists():
        raise SystemExit(
            f"OSHA database not found at {db_path}. "
            "Run `python data/build_osha_db.py` first."
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT standard_id, title, requirement_text, violation_type, "
            "min_penalty, max_penalty, corrective_action, fatal_four_category, "
            "visual_indicators FROM regulations"
        ).fetchall()
    finally:
        conn.close()
    return {r["standard_id"]: RegulationRow(**dict(r)) for r in rows}


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _format_violation_block(reg: RegulationRow, confidence: float, idx: int) -> str:
    severity = reg.violation_type.capitalize()
    penalty_range = f"${reg.min_penalty:,} – ${reg.max_penalty:,}"
    return (
        f"### Violation {idx}: {reg.title}\n"
        f"- **Regulation:** 29 CFR {reg.standard_id} — {reg.title}\n"
        f"- **Observation:** {reg.visual_indicators}\n"
        f"- **Severity:** {severity}\n"
        f"- **Confidence:** {confidence:.2f}\n"
        f"- **Penalty Range:** {penalty_range} per instance\n"
        f"- **Corrective Action:** {reg.corrective_action}\n"
    )


def render_violation_report(
    regs: list[RegulationRow],
    confidences: list[float],
) -> str:
    if not regs:
        return (
            "## SiteSafe Violation Report\n\n"
            "**Violations Detected: 0**\n\n"
            "Based on visual analysis, this site appears to be in compliance with "
            "applicable OSHA construction standards. Workers visible in the frame "
            "are wearing required PPE (hard hats and high-visibility apparel). "
            "Fall protection appears to be in place where required.\n\n"
            "**Note:** This assessment is based on a single photograph and cannot "
            "evaluate all safety conditions. A comprehensive safety inspection should "
            "include physical examination of equipment, review of safety documentation, "
            "and assessment of conditions not visible in this image (e.g., electrical "
            "grounding, excavation shoring, chemical exposure).\n\n"
            "---\n"
            "**Site Compliance Status:** APPEARS COMPLIANT — No visible violations "
            "detected. Standard monitoring recommended."
        )

    blocks = [
        _format_violation_block(reg, conf, i + 1)
        for i, (reg, conf) in enumerate(zip(regs, confidences))
    ]
    has_serious_or_willful = any(r.violation_type in {"serious", "willful"} for r in regs)
    status = (
        "NON-COMPLIANT — immediate corrective action required before work continues."
        if has_serious_or_willful
        else "DEFICIENCIES IDENTIFIED — corrective action required."
    )
    n = len(regs)
    return (
        "## SiteSafe Violation Report\n\n"
        f"**Violations Detected: {n}**\n\n"
        + "\n".join(blocks)
        + "\n---\n"
        + f"**Site Compliance Status:** {status}"
    )


# ---------------------------------------------------------------------------
# Roboflow / YOLO ingest
# ---------------------------------------------------------------------------

def _read_yolo_classes(classes_file: Path) -> list[str]:
    """Read a YOLO ``data.yaml`` or plain ``classes.txt`` file into a list."""
    text = classes_file.read_text(encoding="utf-8")
    if classes_file.suffix.lower() in {".yaml", ".yml"}:
        # tiny inline YAML parse — Roboflow's data.yaml has a ``names: [...]`` line
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("names:"):
                rest = line.split(":", 1)[1].strip()
                if rest.startswith("["):
                    return [n.strip().strip("'\"") for n in rest.strip("[]").split(",") if n.strip()]
        # block-style YAML
        names: list[str] = []
        in_names = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "names:":
                in_names = True
                continue
            if in_names:
                if not stripped.startswith("-"):
                    break
                names.append(stripped.lstrip("- ").strip().strip("'\""))
        if names:
            return names
    # plain text — one class per line
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _yolo_label_classes(label_file: Path, classes: list[str]) -> set[str]:
    """Return the set of class names present in a YOLO label file."""
    if not label_file.exists():
        return set()
    out: set[str] = set()
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            continue
        if 0 <= cid < len(classes):
            out.add(classes[cid].strip())
    return out


def iter_roboflow_examples(
    images_dir: Path,
    labels_dir: Path,
    classes: list[str],
) -> Iterable[tuple[Path, set[str]]]:
    images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    log.info("Found %d images in %s", len(images), images_dir)
    for img in images:
        label = labels_dir / (img.stem + ".txt")
        yield img, _yolo_label_classes(label, classes)


def violations_from_classes(present: set[str]) -> list[str]:
    """Map a frame's detected class names to OSHA standard IDs (deduped)."""
    out: list[str] = []
    for c in present:
        std = CLASS_TO_VIOLATION.get(c.strip().lower())
        if std and std not in out:
            out.append(std)
    return out


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    examples: list[dict] = field(default_factory=list)
    n_violation: int = 0
    n_compliant: int = 0
    n_skipped: int = 0


def build_examples_from_roboflow(
    images_dir: Path,
    labels_dir: Path,
    classes_file: Path,
    regulations: dict[str, RegulationRow],
    user_prompt: str,
    *,
    target_compliant_ratio: float = 0.20,
    rng: random.Random,
) -> BuildResult:
    classes = _read_yolo_classes(classes_file)
    log.info("Loaded %d YOLO classes from %s", len(classes), classes_file)

    result = BuildResult()
    compliant_pool: list[Path] = []

    for img_path, present_classes in iter_roboflow_examples(images_dir, labels_dir, classes):
        std_ids = violations_from_classes(present_classes)
        if std_ids:
            regs = [regulations[s] for s in std_ids if s in regulations]
            if not regs:
                result.n_skipped += 1
                continue
            confidences = [round(rng.uniform(0.78, 0.95), 2) for _ in regs]
            result.examples.append(
                _make_example(img_path, user_prompt, render_violation_report(regs, confidences))
            )
            result.n_violation += 1
        elif any(c.strip().lower() in COMPLIANT_CLASSES for c in present_classes):
            compliant_pool.append(img_path)
        else:
            result.n_skipped += 1

    # Add ~20% compliant examples
    target_compliant = max(1, int(result.n_violation * target_compliant_ratio))
    rng.shuffle(compliant_pool)
    for img_path in compliant_pool[:target_compliant]:
        result.examples.append(
            _make_example(img_path, user_prompt, render_violation_report([], []))
        )
        result.n_compliant += 1

    return result


def build_examples_from_csv(
    csv_path: Path,
    regulations: dict[str, RegulationRow],
    user_prompt: str,
    *,
    rng: random.Random,
) -> BuildResult:
    result = BuildResult()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            img_path = Path(row["image_path"])
            std_ids = [s.strip() for s in row.get("violations", "").split(";") if s.strip()]
            regs = [regulations[s] for s in std_ids if s in regulations]
            if std_ids and not regs:
                log.warning("CSV row references unknown standards: %s", std_ids)
                result.n_skipped += 1
                continue
            confidences = [round(rng.uniform(0.85, 0.98), 2) for _ in regs]
            report = render_violation_report(regs, confidences)
            result.examples.append(_make_example(img_path, user_prompt, report))
            if regs:
                result.n_violation += 1
            else:
                result.n_compliant += 1
    return result


def build_examples_from_templates(
    images_dir: Path,
    user_prompt: str,
) -> BuildResult:
    result = BuildResult()
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        result.examples.append(
            _make_example(
                img,
                user_prompt,
                "## SiteSafe Violation Report\n\n[TODO — fill in violations or replace with compliant report]",
            )
        )
        result.n_violation += 1
    return result


def _make_example(image_path: Path, user_text: str, assistant_text: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": assistant_text},
        ]
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build SiteSafe SFT training data.")
    parser.add_argument("--mode", choices=["roboflow", "csv", "template"], required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "training" / "train.jsonl")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)

    # Roboflow-mode args
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--classes-file", type=Path)
    parser.add_argument("--compliant-ratio", type=float, default=0.20)

    # CSV-mode args
    parser.add_argument("--csv", type=Path)

    args = parser.parse_args(argv)
    rng = random.Random(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    regulations = load_regulations(args.db)
    log.info("Loaded %d regulations from DB", len(regulations))

    if args.mode == "roboflow":
        if not (args.images_dir and args.labels_dir and args.classes_file):
            parser.error("--images-dir, --labels-dir, --classes-file required for roboflow mode")
        result = build_examples_from_roboflow(
            args.images_dir, args.labels_dir, args.classes_file,
            regulations, args.user_prompt,
            target_compliant_ratio=args.compliant_ratio, rng=rng,
        )
    elif args.mode == "csv":
        if not args.csv:
            parser.error("--csv required for csv mode")
        result = build_examples_from_csv(args.csv, regulations, args.user_prompt, rng=rng)
    else:  # template
        if not args.images_dir:
            parser.error("--images-dir required for template mode")
        result = build_examples_from_templates(args.images_dir, args.user_prompt)

    rng.shuffle(result.examples)
    with args.output.open("w", encoding="utf-8") as fh:
        for ex in result.examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    log.info(
        "Wrote %d examples to %s (violations=%d, compliant=%d, skipped=%d)",
        len(result.examples), args.output,
        result.n_violation, result.n_compliant, result.n_skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
