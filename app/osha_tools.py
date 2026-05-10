"""
SiteSafe — OSHA Function-Calling Tools
=======================================

These functions are exposed to Gemma 4 as tools (via Ollama's
function-calling API). The model can choose to invoke any of them while
analyzing a photo to ground its response in the local OSHA SQLite knowledge
base.

Each tool is a thin, type-checked wrapper over ``data/build_osha_db.py``'s
query helpers, plus a JSON-Schema definition that goes into the request's
``tools`` argument.

Public surface:

* ``lookup_regulation(standard_id=None, keyword=None)``
* ``get_fatal_four_info(category)``
* ``get_penalty_info(violation_type)``
* ``TOOLS`` — list of dicts to pass straight through to Ollama
* ``dispatch(name, arguments)`` — execute a tool call by name
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import the build_osha_db module without polluting sys.path long-term.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILD_OSHA_DB_PATH = _REPO_ROOT / "data" / "build_osha_db.py"


def _load_db_module():
    """Lazy-load the OSHA DB helpers so import order is robust."""
    spec = importlib.util.spec_from_file_location("sitesafe_osha_db", _BUILD_OSHA_DB_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {_BUILD_OSHA_DB_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sitesafe_osha_db", module)
    spec.loader.exec_module(module)
    return module


_db = _load_db_module()
log = logging.getLogger("sitesafe.tools")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def lookup_regulation(
    standard_id: str | None = None,
    keyword: str | None = None,
) -> dict[str, Any]:
    """Look up an OSHA regulation by standard ID or keyword search."""
    if not standard_id and not keyword:
        return {
            "error": "Provide either standard_id (e.g. '1926.501(b)(1)') or keyword (e.g. 'fall protection').",
            "results": [],
        }

    rows = _db.query_regulation(standard_id=standard_id, keyword=keyword)
    if not rows:
        return {
            "error": f"No regulation matched (standard_id={standard_id!r}, keyword={keyword!r}).",
            "results": [],
        }

    results = [
        {
            "standard_id":       r["standard_id"],
            "title":             r["title"],
            "subpart":           f"{r['subpart']} — {r['subpart_name']}",
            "requirement_text":  r["requirement_text"],
            "violation_type":    r["violation_type"],
            "penalty_range":     f"${r['min_penalty']:,} – ${r['max_penalty']:,}",
            "corrective_action": r["corrective_action"],
            "fatal_four_category": r["fatal_four_category"],
            "visual_indicators": r["visual_indicators"],
        }
        for r in rows
    ]
    return {"results": results}


def get_fatal_four_info(category: str) -> dict[str, Any]:
    """Return Fatal Four statistics + applicable regulations for a category."""
    info = _db.get_fatal_four_info(category)
    if info is None:
        return {
            "error": f"Unknown Fatal Four category: {category!r}. "
                     "Use one of Falls, Struck-By, Electrocution, Caught-In/Between.",
        }
    return info


def get_penalty_info(violation_type: str) -> dict[str, Any]:
    """Return the penalty schedule for a violation severity class."""
    info = _db.get_penalty_info(violation_type)
    if info is None:
        return {
            "error": f"Unknown violation type: {violation_type!r}. "
                     "Use one of: serious, willful, other, repeat.",
        }
    return info


# ---------------------------------------------------------------------------
# Tool schemas (passed to Ollama)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_regulation",
            "description": (
                "Look up an OSHA construction safety regulation by its CFR standard ID "
                "(e.g., '1926.501(b)(1)') or by keyword search (e.g., 'fall protection', "
                "'hard hat', 'scaffold guardrail'). Returns the regulation text, "
                "violation severity, penalty range, and recommended corrective action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "standard_id": {
                        "type": "string",
                        "description": "The OSHA CFR standard number, e.g., '1926.501(b)(1)'.",
                    },
                    "keyword": {
                        "type": "string",
                        "description": (
                            "Keyword to search for in regulation titles, requirement text, "
                            "and visual indicators (e.g., 'fall protection', 'scaffold', "
                            "'hard hat', 'GFCI', 'trench')."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fatal_four_info",
            "description": (
                "Get information about one of OSHA's Fatal Four construction hazards: "
                "the percentage of construction deaths it causes annually, an estimated "
                "annual death count, and the primary regulations that apply."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["Falls", "Struck-By", "Electrocution", "Caught-In/Between"],
                        "description": "The Fatal Four category to retrieve.",
                    },
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_penalty_info",
            "description": (
                "Get the OSHA penalty schedule (min/max in USD) for a violation severity class."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "violation_type": {
                        "type": "string",
                        "enum": ["serious", "willful", "other", "repeat"],
                        "description": "The severity class to retrieve.",
                    },
                },
                "required": ["violation_type"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatcher (used by inference.py to actually run a tool call)
# ---------------------------------------------------------------------------

_DISPATCH = {
    "lookup_regulation": lookup_regulation,
    "get_fatal_four_info": get_fatal_four_info,
    "get_penalty_info": get_penalty_info,
}


def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool call. Returns a JSON-serializable dict."""
    fn = _DISPATCH.get(name)
    if fn is None:
        log.warning("Unknown tool name: %s", name)
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**arguments)
    except TypeError as exc:
        log.warning("Bad arguments for %s: %s", name, exc)
        return {"error": f"Invalid arguments for {name}: {exc}"}
    except Exception as exc:  # last-resort safety net
        log.exception("Tool %s raised", name)
        return {"error": f"Tool {name} raised: {exc}"}
