"""
SiteSafe — Gemma 4 E4B Fine-Tuning (Unsloth + LoRA + GGUF Export)
==================================================================

End-to-end training entry-point. Uses Unsloth's ``FastVisionModel`` for
multimodal fine-tuning, falls back to text-adapter-only fine-tuning if the
vision pass OOMs on a free Kaggle T4 (16 GB VRAM).

Run::

    python training/finetune_gemma4_e4b.py \\
        --train-jsonl data/training/train.jsonl \\
        --output-dir training/runs/sitesafe \\
        --max-steps 200 \\
        --learning-rate 2e-4 \\
        --r 16

Hyperparameters here are tuned for the free Kaggle T4 (16 GB) — they should
also work on a single 24 GB consumer card with room to spare.

The script is robust to two common failure modes:
  * **Vision OOM** — automatically retried with ``finetune_vision_layers=False``.
  * **Missing optional dependencies** — checked up-front with a clear error
    message instead of mid-training tracebacks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sitesafe.train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Gemma 4 E4B for SiteSafe.")
    p.add_argument("--train-jsonl", type=Path, required=True,
                   help="Path to JSONL produced by data/build_training_data.py.")
    p.add_argument("--output-dir", type=Path, default=Path("training/runs/sitesafe"),
                   help="Where Unsloth writes checkpoints and the merged model.")

    # Model / LoRA
    p.add_argument("--model-name", default="unsloth/gemma-4-E4B-it",
                   help="HF Hub repo or local path for the base model.")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--load-in-4bit", action="store_true", default=True)
    p.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--r", type=int, default=16, help="LoRA rank.")
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--text-only", action="store_true",
                   help="Skip vision LoRA layers (use this if vision pass OOMs).")

    # Training schedule
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # Export
    p.add_argument("--export-gguf", action="store_true", default=True,
                   help="Export the merged model to a 4-bit GGUF after training.")
    p.add_argument("--no-export-gguf", dest="export_gguf", action="store_false")
    p.add_argument("--gguf-quantization", default="q4_k_m",
                   choices=["q4_k_m", "q5_k_m", "q8_0", "f16"])
    p.add_argument("--gguf-output-name", default="sitesafe-gemma4-e4b")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_jsonl_dataset(path: Path):
    """Load a JSONL file into an HF Dataset of {"messages": [...]} rows."""
    from datasets import Dataset

    if not path.exists():
        raise SystemExit(f"Training JSONL not found: {path}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed JSONL line %d: %s", line_no, exc)
    if not rows:
        raise SystemExit(f"No valid examples found in {path}")
    log.info("Loaded %d training examples from %s", len(rows), path)
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, finetune_vision_layers: bool) -> Path:
    """Run one training pass. Returns the directory where the merged model lives."""
    # Imported here so `--help` works without a CUDA-enabled env.
    try:
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator
    except ImportError as exc:
        raise SystemExit(
            "Unsloth is not installed. Activate the training env or run "
            "`pip install unsloth`. Original error: %s" % exc
        )
    try:
        from trl import SFTTrainer, SFTConfig
    except ImportError as exc:
        raise SystemExit(f"trl is not installed (`pip install trl`): {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading model %s (4bit=%s)", args.model_name, args.load_in_4bit)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    log.info("Applying LoRA (r=%d, alpha=%d, vision_layers=%s)",
             args.r, args.lora_alpha, finetune_vision_layers)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision_layers,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
    )

    dataset = load_jsonl_dataset(args.train_jsonl)

    sft_config = SFTConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=args.logging_steps,
        output_dir=str(args.output_dir),
        seed=args.seed,
        save_strategy="steps",
        save_steps=max(args.max_steps // 4, 25),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=sft_config,
    )

    log.info("Starting training for %d steps", args.max_steps)
    trainer.train()
    log.info("Training complete")

    merged_dir = args.output_dir / "merged"
    log.info("Saving merged model to %s", merged_dir)
    model.save_pretrained_merged(str(merged_dir), tokenizer)
    return merged_dir


# ---------------------------------------------------------------------------
# OOM-safe entry point
# ---------------------------------------------------------------------------

def is_oom(exc: BaseException) -> bool:
    txt = str(exc).lower()
    return "out of memory" in txt or "cuda oom" in txt or "cublas_status_alloc_failed" in txt


def maybe_export_gguf(args: argparse.Namespace, merged_dir: Path) -> Optional[Path]:
    if not args.export_gguf:
        return None
    try:
        from unsloth import FastVisionModel
    except ImportError:
        log.warning("Unsloth not available — skipping GGUF export.")
        return None

    log.info("Loading merged model for GGUF export")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(merged_dir),
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
    )
    gguf_path = args.output_dir / args.gguf_output_name
    log.info("Saving GGUF (%s) → %s", args.gguf_quantization, gguf_path)
    model.save_pretrained_gguf(
        str(gguf_path),
        tokenizer,
        quantization_method=args.gguf_quantization,
    )
    return gguf_path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # First attempt: vision LoRA enabled (unless --text-only).
    finetune_vision = not args.text_only

    try:
        merged_dir = train(args, finetune_vision_layers=finetune_vision)
    except (RuntimeError, MemoryError) as exc:
        if finetune_vision and is_oom(exc):
            log.warning("Vision LoRA OOMed (%s). Retrying with text-only LoRA.", exc)
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
            merged_dir = train(args, finetune_vision_layers=False)
        else:
            raise

    gguf_path = maybe_export_gguf(args, merged_dir)
    log.info("Done. Merged model at %s", merged_dir)
    if gguf_path:
        log.info("GGUF written to %s", gguf_path)
        log.info("Now register with Ollama:")
        log.info("    ollama create sitesafe -f training/Modelfile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
