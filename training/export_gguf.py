"""
SiteSafe — Standalone GGUF Exporter
====================================

Convert a fine-tuned (or merged) HuggingFace-format Gemma 4 checkpoint to a
GGUF that Ollama can load. Use this when you trained without ``--export-gguf``
or want to re-quantize an existing checkpoint at a different precision.

Run::

    python training/export_gguf.py \\
        --model-dir training/runs/sitesafe/merged \\
        --output sitesafe-gemma4-e4b \\
        --quantization q4_k_m

Quantization presets (size vs. quality):
    * ``q4_k_m`` — recommended for edge inference (≈3 GB for E4B)
    * ``q5_k_m`` — slightly higher quality, ≈3.7 GB
    * ``q8_0``  — near-lossless, ≈4.2 GB
    * ``f16``   — full precision, ≈8 GB
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sitesafe.export")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a Gemma 4 checkpoint to GGUF.")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Directory containing a merged HF-format model.")
    parser.add_argument("--output", default="sitesafe-gemma4-e4b",
                        help="GGUF output directory / filename stem.")
    parser.add_argument("--quantization", default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"])
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args(argv)

    if not args.model_dir.exists():
        raise SystemExit(f"Model dir not found: {args.model_dir}")

    try:
        from unsloth import FastVisionModel
    except ImportError as exc:
        raise SystemExit(
            "Unsloth is not installed in this environment. Run `pip install unsloth`. "
            f"Original error: {exc}"
        )

    log.info("Loading merged model from %s", args.model_dir)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(args.model_dir),
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
    )

    log.info("Saving GGUF (%s) → %s", args.quantization, args.output)
    model.save_pretrained_gguf(
        args.output,
        tokenizer,
        quantization_method=args.quantization,
    )
    log.info("Done. Register with Ollama:")
    log.info("    ollama create sitesafe -f training/Modelfile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
