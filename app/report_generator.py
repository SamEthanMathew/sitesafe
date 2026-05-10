"""
SiteSafe — PDF Report Generator
================================

Turns a Markdown SiteSafe Violation Report (the model's output) into a
polished PDF that a site supervisor can hand to a foreman or attach to an
email. Uses ``fpdf2`` so we don't need a working LaTeX install on edge
deployments.

Public surface:

    pdf_path = generate_pdf(
        report_markdown,
        image_path,
        site_name="42nd & Park Tower",
        location="New York, NY",
        date="2026-05-08",
        output_path="reports/sitesafe-report.pdf",
    )
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from fpdf import FPDF
from fpdf.enums import XPos, YPos

log = logging.getLogger("sitesafe.report")

# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedViolation:
    title: str
    regulation: str = ""
    observation: str = ""
    severity: str = ""
    confidence: str = ""
    penalty_range: str = ""
    corrective_action: str = ""
    extras: dict[str, str] = field(default_factory=dict)


@dataclass
class ParsedReport:
    n_violations: int
    violations: list[ParsedViolation]
    site_status: str
    compliant_summary: Optional[str] = None  # only populated when n_violations == 0


_VIOLATION_HEADER_RE = re.compile(
    r"^\s*#{1,6}\s*Violation\s+\d+\s*(?::\s*(?P<title>.+?))?\s*$",
    re.IGNORECASE,
)
# Accept '-', '*', '•' bullets (the model varies); allow leading whitespace.
_BULLET_RE = re.compile(
    r"^\s*[-*•]\s+\*\*(?P<key>[^:*]+?):\*\*\s*(?P<val>.*)$",
)
_STATUS_RE = re.compile(r"^\*\*Site Compliance Status:\*\*\s*(?P<status>.+)$", re.IGNORECASE)
_COUNT_RE = re.compile(r"^\*\*Violations Detected:\s*(?P<n>\d+)\*\*\s*$", re.IGNORECASE)

# Tolerant key normalization — different prompts and base models name fields
# differently. Map every reasonable spelling onto our canonical attribute name.
_KEY_ALIASES: dict[str, str] = {
    "regulation":             "regulation",
    "cfrstandard":            "regulation",
    "cfr":                    "regulation",
    "standard":               "regulation",
    "oshastandard":           "regulation",
    "observation":            "observation",
    "description":            "observation",
    "whatisee":               "observation",
    "whatyousee":             "observation",
    "severity":               "severity",
    "severityclassification": "severity",
    "classification":         "severity",
    "confidence":             "confidence",
    "confidencescore":        "confidence",
    "penaltyrange":           "penalty_range",
    "oshapenaltyrange":       "penalty_range",
    "penalty":                "penalty_range",
    "fine":                   "penalty_range",
    "correctiveaction":       "corrective_action",
    "correctiveactions":      "corrective_action",
    "remediation":            "corrective_action",
    "fix":                    "corrective_action",
}


def _normalize_key(raw: str) -> str:
    """Lowercase + strip non-alphanumerics so 'CFR Standard' → 'cfrstandard'."""
    return re.sub(r"[^a-z0-9]+", "", raw.lower())


def parse_report(markdown: str) -> ParsedReport:
    """Best-effort parse of a SiteSafe Violation Report Markdown blob.

    The parser is intentionally tolerant — different base models (the
    prototype gemma3:4b, the fine-tuned SiteSafe model, future LoRAs) emit
    slight format drift around bullet characters, key wording, and whether
    the violation title is on the heading line.
    """
    declared_n: int | None = None
    violations: list[ParsedViolation] = []
    current: ParsedViolation | None = None
    site_status = "(unknown)"
    compliant_summary_lines: list[str] = []
    in_compliant_summary_zone = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        m = _COUNT_RE.match(line)
        if m:
            declared_n = int(m.group("n"))
            in_compliant_summary_zone = (declared_n == 0)
            continue

        m = _VIOLATION_HEADER_RE.match(line)
        if m:
            in_compliant_summary_zone = False
            title = (m.group("title") or "").strip() or "(unspecified violation)"
            current = ParsedViolation(title=title)
            violations.append(current)
            continue

        m = _STATUS_RE.match(line)
        if m:
            site_status = m.group("status").strip()
            in_compliant_summary_zone = False
            continue

        if current is not None:
            mb = _BULLET_RE.match(line)
            if mb:
                key_norm = _normalize_key(mb.group("key"))
                val = mb.group("val").strip()
                attr = _KEY_ALIASES.get(key_norm)
                if attr:
                    # Don't clobber an already-populated field if a model emits
                    # both "Regulation" and "CFR Standard" — first-wins keeps
                    # the canonical citation.
                    existing = getattr(current, attr) or ""
                    if not existing:
                        setattr(current, attr, val)
                else:
                    current.extras[key_norm] = val
                # If the violation header didn't carry a title, reuse the
                # regulation title once we know it.
                if attr == "regulation" and current.title == "(unspecified violation)":
                    current.title = val.split(" — ", 1)[-1].strip() or current.title
                continue

        if in_compliant_summary_zone and line.strip():
            # Strip the heading and the closing horizontal rule.
            if line.startswith("##") or line.startswith("---"):
                continue
            compliant_summary_lines.append(line.strip())

    # Trust the declared count if the model gave one; otherwise infer from
    # what we parsed.
    n_violations = declared_n if declared_n is not None else len(violations)

    summary = " ".join(compliant_summary_lines).strip() or None
    return ParsedReport(
        n_violations=n_violations,
        violations=violations,
        site_status=site_status,
        compliant_summary=summary,
    )


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

# fpdf core fonts are Latin-1 only — strip anything that won't encode.
def _ascii(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


SEVERITY_COLORS = {
    "serious":   (200, 50, 50),
    "willful":   (170, 0, 0),
    "other":     (215, 140, 0),
    "compliant": (40, 130, 60),
}


class _SiteSafePDF(FPDF):
    """Custom FPDF subclass with a SiteSafe-branded header/footer."""

    def header(self) -> None:
        self.set_fill_color(20, 35, 70)
        self.rect(0, 0, self.w, 18, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.set_xy(10, 5)
        self.cell(0, 10, _ascii("SiteSafe — OSHA Construction Safety Report"), align="L",
                  new_x=XPos.LMARGIN, new_y=YPos.TOP)
        self.set_font("Helvetica", "", 9)
        self.set_xy(10, 11)
        self.cell(0, 6, _ascii("Generated on-device by Gemma 4 E4B   |   100% offline"),
                  align="L", new_x=XPos.LMARGIN, new_y=YPos.TOP)
        self.set_xy(self.l_margin, 22)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(
            0, 5,
            _ascii(
                "SiteSafe is an assistive tool. It does not replace certified OSHA "
                "inspections or qualified competent persons. "
                f"Page {self.page_no()}/{{nb}}"
            ),
            align="C",
            new_x=XPos.LMARGIN, new_y=YPos.TOP,
        )


def _draw_severity_chip(pdf: FPDF, severity: str) -> None:
    sev_lower = severity.lower()
    color = SEVERITY_COLORS.get(
        next((k for k in SEVERITY_COLORS if k in sev_lower), "other"),
        (90, 90, 90),
    )
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    label = _ascii(severity.upper() or "UNKNOWN")
    width = pdf.get_string_width(label) + 6
    # Chip ends a line — cursor jumps to next-line left margin.
    pdf.cell(width, 6, label, fill=True, align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)


def _draw_kv(pdf: FPDF, key: str, value: str) -> None:
    """One bold-key + value row. Always starts at the left margin and ends on
    a fresh line, regardless of how many lines the value wraps to."""
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 10)
    key_width = 38
    pdf.cell(key_width, 6, _ascii(f"{key}:"),
             new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 10)
    val_width = pdf.w - pdf.r_margin - pdf.get_x()
    if val_width < 30:
        # Defensive — never let multi_cell get a width below a single character.
        pdf.ln(6)
        pdf.set_x(pdf.l_margin + key_width)
        val_width = pdf.w - pdf.r_margin - pdf.get_x()
    pdf.multi_cell(
        val_width, 6, _ascii(value or "(not provided)"),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )


def _draw_violation(pdf: FPDF, idx: int, viol: ParsedViolation) -> None:
    pdf.set_fill_color(245, 247, 252)
    pdf.set_draw_color(20, 35, 70)
    pdf.set_line_width(0.4)

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, _ascii(f"Violation {idx}: {viol.title}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if viol.severity:
        _draw_severity_chip(pdf, viol.severity)
    else:
        pdf.ln(2)

    _draw_kv(pdf, "Regulation", viol.regulation)
    _draw_kv(pdf, "Observation", viol.observation)
    _draw_kv(pdf, "Confidence", viol.confidence)
    _draw_kv(pdf, "Penalty range", viol.penalty_range)
    _draw_kv(pdf, "Corrective action", viol.corrective_action)

    end_y = pdf.get_y()
    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, end_y + 1, pdf.w - pdf.r_margin, end_y + 1)
    pdf.ln(5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pdf(
    report_markdown: str,
    image_path: str | Path,
    *,
    site_name: str = "",
    location: str = "",
    date: str = "",
    output_path: str | Path | None = None,
) -> Path:
    """Render the report (Markdown + photo) to a PDF and return the output path."""
    parsed = parse_report(report_markdown)

    pdf = _SiteSafePDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_y(28)

    # Site information block ------------------------------------------------
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _ascii("Site Information"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_fill_color(238, 240, 246)
    pdf.cell(0, 6, _ascii(f"Site name : {site_name or '(unspecified)'}"),
             fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, _ascii(f"Location  : {location or '(unspecified)'}"),
             fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, _ascii(f"Photo date: {date or '(unspecified)'}"),
             fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, _ascii(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M %Z').strip()}"),
             fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # Image preview ---------------------------------------------------------
    img_path = Path(image_path)
    if img_path.exists():
        try:
            available_w = pdf.w - pdf.l_margin - pdf.r_margin
            target_w = min(available_w, 130)
            pdf.image(str(img_path), x=pdf.l_margin, w=target_w)
            pdf.ln(4)
        except Exception as exc:  # noqa: BLE001 — never let an image crash the report
            log.warning("Could not embed image %s: %s", img_path, exc)

    # Summary banner --------------------------------------------------------
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _ascii(f"Violations detected: {parsed.n_violations}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if parsed.n_violations == 0:
        pdf.set_fill_color(*SEVERITY_COLORS["compliant"])
    elif "non-compliant" in parsed.site_status.lower():
        pdf.set_fill_color(*SEVERITY_COLORS["serious"])
    else:
        pdf.set_fill_color(*SEVERITY_COLORS["other"])
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, _ascii(f"Status: {parsed.site_status}"),
             fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Per-violation blocks --------------------------------------------------
    if parsed.violations:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 7, _ascii("Findings"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        for i, v in enumerate(parsed.violations, start=1):
            _draw_violation(pdf, i, v)
    elif parsed.compliant_summary:
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _ascii(parsed.compliant_summary),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Disclaimer ------------------------------------------------------------
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(95, 95, 95)
    pdf.multi_cell(
        0, 5,
        _ascii(
            "This report is generated by SiteSafe, an AI-powered assistive tool. "
            "It is not a determination of fact or a substitute for an OSHA inspection. "
            "Photo-based assessment cannot evaluate every hazard (e.g., electrical "
            "grounding, excavation shoring, chemical exposure, training records). "
            "Always pair this output with on-the-ground inspection by a qualified "
            "competent person."
        ),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    # Output ---------------------------------------------------------------
    if output_path is None:
        output_path = Path("reports") / f"sitesafe-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    log.info("Wrote PDF report → %s", output_path)
    return output_path
