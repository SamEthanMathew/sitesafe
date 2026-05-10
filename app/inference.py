"""
SiteSafe — Ollama Inference Wrapper
====================================

Talks to a local Ollama server, drives the multi-turn tool-calling loop, and
returns a single ``InferenceResult`` containing the model's final structured
violation report along with raw timings and any tool calls that were made
along the way.

The wrapper is designed to **fail loudly and helpfully**:

* If Ollama isn't reachable, it raises ``OllamaUnavailableError`` with the
  exact remediation steps a contractor (i.e. someone who has never used
  Ollama before) needs.
* If the model is missing, it raises ``ModelNotFoundError`` with the
  ``ollama create`` command they should run.
* If the loop runs away (model keeps requesting tool calls forever), it
  short-circuits at ``max_iterations`` and returns whatever it has.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .osha_tools import TOOLS, dispatch
from .prompts import SYSTEM_PROMPT, render_user_prompt

log = logging.getLogger("sitesafe.inference")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OllamaUnavailableError(RuntimeError):
    """Raised when we can't reach the Ollama server."""


class ModelNotFoundError(RuntimeError):
    """Raised when the configured Ollama model is not registered."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ToolCallTrace:
    name: str
    arguments: dict
    result: dict


@dataclass
class InferenceResult:
    text: str
    iterations: int
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    latency_seconds: float = 0.0
    model: str = ""
    raw_messages: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image(image_path: Path) -> str:
    """Ollama accepts base64-encoded images in the ``images`` array."""
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _ensure_ollama_module():
    try:
        import ollama  # noqa: WPS433 — runtime import is intentional
        return ollama
    except ImportError as exc:
        raise OllamaUnavailableError(
            "The `ollama` Python package is not installed.\n"
            "Install with:  pip install ollama\n"
            f"(original error: {exc})"
        ) from exc


def _client(host: str | None = None):
    ollama = _ensure_ollama_module()
    host = host or os.environ.get("OLLAMA_HOST")
    if host:
        return ollama.Client(host=host)
    return ollama  # module-level client uses default http://localhost:11434


def _check_model_available(client_or_module, model: str) -> None:
    try:
        listing = client_or_module.list()
    except Exception as exc:  # noqa: BLE001
        raise OllamaUnavailableError(
            "Cannot reach Ollama at "
            f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}.\n"
            "Start it with:  ollama serve   (or run the docker-compose stack)\n"
            f"(underlying error: {exc})"
        ) from exc

    # Normalize across ollama-python versions (dict vs object)
    available: list[str] = []
    raw_models: Iterable[Any] = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
    for m in raw_models:
        if isinstance(m, dict):
            name = m.get("model") or m.get("name")
        else:
            name = getattr(m, "model", None) or getattr(m, "name", None)
        if name:
            available.append(str(name))

    if not any(model == m or m.startswith(f"{model}:") for m in available):
        raise ModelNotFoundError(
            f"Model {model!r} is not registered with Ollama.\n"
            f"Available: {available or 'none'}\n"
            "Build it from the SiteSafe Modelfile:\n"
            "    ollama create sitesafe -f training/Modelfile\n"
            "or pull a generic Gemma 3 base for prototyping:\n"
            "    ollama pull gemma3:4b"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    image_path: str | Path,
    *,
    model: str | None = None,
    site_name: str = "",
    location: str = "",
    date: str = "",
    max_iterations: int = 4,
    temperature: float = 0.3,
    timeout_seconds: float = 120.0,
    host: str | None = None,
) -> InferenceResult:
    """Run the full SiteSafe inference loop and return the structured report.

    The loop:
        1. Send {system, user(image+prompt)} to the model.
        2. If the model returns ``tool_calls``, execute each via
           ``osha_tools.dispatch`` and append a ``tool`` message back in.
        3. Repeat until the model responds with no tool calls or
           ``max_iterations`` is reached.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = model or os.environ.get("SITESAFE_MODEL", "sitesafe")
    client = _client(host)
    _check_model_available(client, model)

    user_prompt = render_user_prompt(site_name=site_name, location=location, date=date)
    image_b64 = _encode_image(image_path)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_prompt,
            "images": [image_b64],
        },
    ]

    tool_traces: list[ToolCallTrace] = []
    started = time.perf_counter()
    final_text = ""
    iteration = 0
    tools_supported = True  # toggled to False on first 'does not support tools' 400

    for iteration in range(1, max_iterations + 1):
        log.info("Inference iteration %d/%d (model=%s, tools=%s)",
                 iteration, max_iterations, model, tools_supported)
        chat_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature, "num_ctx": 4096},
            "stream": False,
        }
        if tools_supported:
            chat_kwargs["tools"] = TOOLS
        try:
            response = client.chat(**chat_kwargs)
        except Exception as exc:  # noqa: BLE001 — surfacing the real cause matters here
            # Some Ollama base models (e.g. gemma3:4b) don't advertise tool support.
            # Retry once without `tools` so the prototype path still works.
            if tools_supported and "does not support tools" in str(exc).lower():
                log.warning(
                    "Model %s does not support tool calling — retrying without tools. "
                    "(Fine-tune via training/finetune_kaggle_notebook.ipynb to unlock function calling.)",
                    model,
                )
                tools_supported = False
                continue
            elapsed = time.perf_counter() - started
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Ollama inference exceeded {timeout_seconds}s. "
                    "Re-run on a smaller model (gemma3:4b) or upgrade hardware."
                ) from exc
            raise

        message = (
            response["message"] if isinstance(response, dict) else getattr(response, "message", {})
        )
        content = (message.get("content") if isinstance(message, dict) else getattr(message, "content", "")) or ""
        tool_calls = (
            message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
        ) or []

        # Echo the assistant turn so the next call has the context.
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

        if not tool_calls:
            final_text = content.strip()
            break

        for call in tool_calls:
            fn = call["function"] if isinstance(call, dict) else getattr(call, "function", {})
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
            raw_args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", {})
            arguments = raw_args if isinstance(raw_args, dict) else _safe_json_loads(raw_args)

            log.info("Model invoked tool %s with %s", name, arguments)
            result = dispatch(name, arguments)
            tool_traces.append(ToolCallTrace(name=name, arguments=arguments, result=result))

            messages.append({
                "role": "tool",
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # Hard ceiling on runaway loops.
        if iteration == max_iterations:
            log.warning("Reached max_iterations=%d without a final answer; returning best-effort.", max_iterations)
            final_text = content.strip() if content else (
                "## SiteSafe Violation Report\n\n"
                "**Analysis incomplete** — the model exceeded its tool-call budget. "
                "Re-run with a smaller image or increase `max_iterations`."
            )

    elapsed = time.perf_counter() - started
    return InferenceResult(
        text=final_text or "(no content returned)",
        iterations=iteration,
        tool_calls=tool_traces,
        latency_seconds=elapsed,
        model=model,
        raw_messages=messages,
    )


def _safe_json_loads(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, str)):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return {}
    return {}
