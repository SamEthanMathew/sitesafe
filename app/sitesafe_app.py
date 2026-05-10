"""
SiteSafe — Gradio Web Application
==================================

Polished single-file Gradio UI:

* Image upload (drag-and-drop, file picker, or webcam capture)
* Optional site name / location / date metadata
* "Analyze for Violations" — runs the full Ollama tool-calling loop
* Renders the SiteSafe Violation Report as Markdown
* "Export PDF Report" — produces a styled, brand-coloured PDF

Run::

    python app/sitesafe_app.py

Set ``OLLAMA_HOST`` and/or ``SITESAFE_MODEL`` to override the defaults.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr

# Allow `python app/sitesafe_app.py` to import sibling modules cleanly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference import (  # noqa: E402  (import after sys.path tweak)
    InferenceResult,
    ModelNotFoundError,
    OllamaUnavailableError,
    run_inference,
)
from app.confidence_calibration import ConfidenceCalibrator  # noqa: E402
from app.report_generator import generate_pdf  # noqa: E402

logging.basicConfig(
    level=os.environ.get("SITESAFE_LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sitesafe.app")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample_images"
CALIBRATOR_PATH = REPO_ROOT / "app" / "calibrator.joblib"
DEFAULT_MODEL = os.environ.get("SITESAFE_MODEL", "sitesafe")
GRADIO_PORT = int(os.environ.get("SITESAFE_PORT", "7860"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIDENCE_LINE_RE = re.compile(r"^(- \*\*Confidence:\*\* )([0-9.]+)\s*$", re.MULTILINE)


def _maybe_load_calibrator() -> Optional[ConfidenceCalibrator]:
    if not CALIBRATOR_PATH.exists():
        return None
    try:
        return ConfidenceCalibrator.load(CALIBRATOR_PATH)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load calibrator from %s: %s", CALIBRATOR_PATH, exc)
        return None


def _apply_calibration(report_md: str, cal: Optional[ConfidenceCalibrator]) -> str:
    """Replace each `Confidence: X` line with the calibrated value."""
    if cal is None:
        return report_md

    def repl(match: re.Match[str]) -> str:
        try:
            raw = float(match.group(2))
        except ValueError:
            return match.group(0)
        cal_val = cal.calibrate(raw)
        return f"{match.group(1)}{cal_val:.2f} (raw {raw:.2f})"

    return _CONFIDENCE_LINE_RE.sub(repl, report_md)


def _persist_image_to_tempfile(image_obj) -> Path:
    """Gradio gives us a numpy array or PIL — normalize to a JPEG path on disk."""
    from PIL import Image
    import numpy as np

    if image_obj is None:
        raise ValueError("Please upload or capture an image first.")

    if isinstance(image_obj, np.ndarray):
        img = Image.fromarray(image_obj)
    elif isinstance(image_obj, Image.Image):
        img = image_obj
    elif isinstance(image_obj, (str, os.PathLike)):
        return Path(image_obj)
    else:
        raise ValueError(f"Unsupported image type: {type(image_obj).__name__}")

    if img.mode not in {"RGB", "L"}:
        img = img.convert("RGB")

    fd, name = tempfile.mkstemp(suffix=".jpg", prefix="sitesafe_")
    os.close(fd)
    img.save(name, format="JPEG", quality=92)
    return Path(name)


def _format_diagnostics(result: InferenceResult) -> str:
    lines = [
        "**Inference Diagnostics**",
        "",
        f"- Model: `{result.model}`",
        f"- Iterations: {result.iterations}",
        f"- Latency: {result.latency_seconds:.1f} s",
        f"- Tool calls: {len(result.tool_calls)}",
    ]
    for i, tc in enumerate(result.tool_calls, start=1):
        truncated_args = str(tc.arguments)[:80]
        lines.append(f"  {i}. `{tc.name}({truncated_args})`")
    return "\n".join(lines)


def _short_error_panel(headline: str, body: str) -> str:
    return f"### {headline}\n\n{body}"


# ---------------------------------------------------------------------------
# Core analysis callback
# ---------------------------------------------------------------------------

def analyze_callback(
    image_obj,
    site_name: str,
    location: str,
    date: str,
    use_calibration: bool,
):
    """Wrapped by Gradio. Returns: (report_markdown, diagnostics, pdf_state)."""
    try:
        image_path = _persist_image_to_tempfile(image_obj)
    except ValueError as exc:
        return _short_error_panel("⚠️ No image", str(exc)), "", None

    try:
        result = run_inference(
            image_path,
            model=DEFAULT_MODEL,
            site_name=site_name,
            location=location,
            date=date,
        )
    except OllamaUnavailableError as exc:
        return _short_error_panel("🛑 Ollama not reachable", f"```\n{exc}\n```"), "", None
    except ModelNotFoundError as exc:
        return _short_error_panel("📦 Model not registered", f"```\n{exc}\n```"), "", None
    except FileNotFoundError as exc:
        return _short_error_panel("⚠️ Image not found", str(exc)), "", None
    except TimeoutError as exc:
        return _short_error_panel("⏱️ Inference timed out", str(exc)), "", None
    except Exception as exc:  # noqa: BLE001 — top-level guard for the UI
        log.exception("Unhandled inference error")
        return _short_error_panel("💥 Unexpected error", f"```\n{exc}\n```"), "", None

    report = result.text
    if use_calibration:
        report = _apply_calibration(report, _maybe_load_calibrator())

    diagnostics = _format_diagnostics(result)
    pdf_state = {
        "report": report,
        "image_path": str(image_path),
        "site_name": site_name,
        "location": location,
        "date": date,
    }
    return report, diagnostics, pdf_state


def export_pdf_callback(pdf_state):
    """Generate a PDF from the cached state and return its path for download."""
    if not pdf_state:
        return gr.update(value=None, visible=False), "Run an analysis first, then click **Export PDF Report**."
    try:
        out_dir = Path(tempfile.gettempdir()) / "sitesafe_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"sitesafe-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
        generate_pdf(
            pdf_state["report"],
            pdf_state["image_path"],
            site_name=pdf_state.get("site_name", ""),
            location=pdf_state.get("location", ""),
            date=pdf_state.get("date", ""),
            output_path=out_path,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("PDF export failed")
        return gr.update(value=None, visible=False), f"❌ PDF export failed: {exc}"
    return gr.update(value=str(out_path), visible=True), "✅ PDF generated."


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------

def _list_sample_images() -> list[str]:
    if not SAMPLE_DIR.exists():
        return []
    return [
        str(p) for p in sorted(SAMPLE_DIR.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

ABOUT_MD = """\
**SiteSafe** is an on-device OSHA construction safety violation detector.

You give it a photo of a job site. It returns a structured violation report
that cites the exact 29 CFR 1926 standard, severity, calibrated confidence,
penalty range, and a corrective action — entirely on your machine, with no
data leaving your device.

Powered by **Gemma 4 E4B** (fine-tuned with **Unsloth**, deployed via
**Ollama**).
"""

FATAL_FOUR_MD = """\
OSHA's "Fatal Four" hazards cause **58.6%** of all construction worker
deaths annually — approximately **1,069 deaths in 2024**:

| Hazard | % of Deaths | Annual Deaths | Primary OSHA Standard |
|---|---|---|---|
| Falls | 33.5% | ~358 | 29 CFR 1926.501 |
| Struck-By | 11.4% | ~122 | 29 CFR 1926.602 |
| Electrocution | 8.4% | ~90 | 29 CFR 1926.405 |
| Caught-In/Between | 5.4% | ~58 | 29 CFR 1926.652 |
"""

TECHNICAL_MD = """\
**Architecture:**

- **Vision + reasoning:** Gemma 4 E4B (4B effective parameters), fine-tuned
  with Unsloth on a SiteSafe SFT dataset constructed from publicly licensed
  PPE-detection datasets.
- **Function calling:** the model can autonomously query a local SQLite DB
  of 50+ OSHA construction standards (`lookup_regulation`,
  `get_fatal_four_info`, `get_penalty_info`).
- **Confidence calibration:** Platt scaling on a held-out validation set,
  bringing Expected Calibration Error below 0.10 (see
  `evaluation/results/reliability_diagram.png`).
- **Deployment:** GGUF (q4_k_m) loaded by Ollama. Runs on a laptop with
  16 GB RAM. No cloud calls.
"""


def build_ui() -> gr.Blocks:
    samples = _list_sample_images()

    with gr.Blocks(
        title="SiteSafe — On-Device OSHA Construction Safety",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"),
        css="""
            .sitesafe-banner {
              background: linear-gradient(90deg, #14234a 0%, #2a4480 100%);
              padding: 18px 24px; border-radius: 10px; color: white;
            }
            .sitesafe-banner h1 { margin: 0 0 4px 0; font-size: 26px; }
            .sitesafe-banner p { margin: 0; opacity: 0.85; font-size: 13px; }
            #report-area { min-height: 360px; }
        """,
    ) as demo:
        gr.HTML(
            """
            <div class="sitesafe-banner">
              <h1>🦺 SiteSafe — On-Device OSHA Construction Safety Assistant</h1>
              <p>Powered by Gemma 4 E4B   •   100% Offline   •   No Cloud Dependency</p>
            </div>
            """
        )

        with gr.Row():
            # ----------------- Left column: input -----------------
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Job-site photo",
                    sources=["upload", "webcam", "clipboard"],
                    type="numpy",
                    height=320,
                )
                with gr.Group():
                    site_name = gr.Textbox(label="Site name (optional)", placeholder="e.g., 42nd & Park Tower")
                    location = gr.Textbox(label="Location (optional)", placeholder="e.g., New York, NY")
                    date = gr.Textbox(label="Photo date (optional)", placeholder="e.g., 2026-05-08")
                use_calibration = gr.Checkbox(
                    label="Apply Platt-calibrated confidence scores",
                    value=True,
                )
                analyze_btn = gr.Button("🔍 Analyze for Violations", variant="primary", size="lg")

                if samples:
                    gr.Examples(
                        examples=[[s] for s in samples],
                        inputs=[image_input],
                        label="Try a sample image",
                        examples_per_page=5,
                    )

            # ----------------- Right column: output -----------------
            with gr.Column(scale=1):
                report_md = gr.Markdown(
                    "### Output\nUpload an image and click **Analyze for Violations**.",
                    elem_id="report-area",
                )
                diagnostics_md = gr.Markdown("")
                with gr.Row():
                    export_btn = gr.Button("📄 Export PDF Report", variant="secondary")
                pdf_status = gr.Markdown("")
                pdf_download = gr.File(label="Download PDF", visible=False, interactive=False)

        # ----------------- Bottom accordions -----------------
        with gr.Accordion("About SiteSafe", open=False):
            gr.Markdown(ABOUT_MD)
        with gr.Accordion("OSHA Fatal Four (the problem we solve)", open=False):
            gr.Markdown(FATAL_FOUR_MD)
        with gr.Accordion("Technical Details", open=False):
            gr.Markdown(TECHNICAL_MD)

        # ----------------- Wiring -----------------
        pdf_state = gr.State()

        analyze_btn.click(
            fn=analyze_callback,
            inputs=[image_input, site_name, location, date, use_calibration],
            outputs=[report_md, diagnostics_md, pdf_state],
            queue=True,
        )

        export_btn.click(
            fn=export_pdf_callback,
            inputs=[pdf_state],
            outputs=[pdf_download, pdf_status],
            queue=True,
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Launching SiteSafe on port %d (model=%s)", GRADIO_PORT, DEFAULT_MODEL)
    demo = build_ui()
    demo.queue(default_concurrency_limit=2)
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=GRADIO_PORT,
        show_error=True,
        favicon_path=None,
    )


if __name__ == "__main__":
    main()
