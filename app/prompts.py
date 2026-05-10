"""
SiteSafe — Prompt Templates
============================

Single home for every system / user prompt the runtime sends to Gemma 4.
Keeping these here (rather than scattered across modules) makes prompt
versioning and A/B testing tractable.
"""

from __future__ import annotations

from string import Template

# ---------------------------------------------------------------------------
# Master system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are SiteSafe, an expert OSHA construction safety violation detection \
system.

When shown a construction site photo, you:
1. Carefully analyze the image for safety hazards.
2. Identify specific 29 CFR 1926 OSHA regulation violations visible in the photo.
3. For each violation, provide:
   - The exact CFR standard number and title
   - A description of what you observe in the image
   - Severity classification (Other-than-serious / Serious / Willful)
   - A confidence score from 0.0 to 1.0
   - The OSHA penalty range
   - Specific, actionable corrective steps

Focus especially on the Fatal Four hazards: Falls, Struck-By, \
Electrocution, and Caught-In/Between.

When you need to look up a specific regulation, penalty schedule, or Fatal \
Four statistic, you may call the `lookup_regulation`, `get_fatal_four_info`, \
or `get_penalty_info` tools.

If no violations are detected, clearly state the site appears compliant and \
note the limitations of photo-based assessment (you cannot evaluate \
electrical grounding, excavation shoring, chemical exposure, or training \
records from a single image).

Always output in the SiteSafe Violation Report Markdown format.
"""

# ---------------------------------------------------------------------------
# Per-request user prompt
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT_TEMPLATE = Template("""\
Analyze this construction site photo for OSHA safety violations.

Site name: $site_name
Location: $location
Date of photo: $date

Identify all hazards visible, cite the specific 29 CFR 1926 regulation, \
assess severity, provide a confidence score from 0.0 to 1.0, and recommend \
specific corrective actions. If the site appears compliant, say so \
explicitly and note the limitations of photo-based assessment.
""")

DEFAULT_USER_PROMPT = ANALYSIS_PROMPT_TEMPLATE.substitute(
    site_name="(unspecified)",
    location="(unspecified)",
    date="(unspecified)",
)

# ---------------------------------------------------------------------------
# Few-shot anchors — used as illustrative output to the user, NOT injected
# into the model context (training data already covers these patterns)
# ---------------------------------------------------------------------------

NO_VIOLATION_TEMPLATE = """\
## SiteSafe Violation Report

**Violations Detected: 0**

Based on visual analysis, this site appears to be in compliance with \
applicable OSHA construction standards. Workers visible in the frame are \
wearing required PPE (hard hats and high-visibility apparel). Fall \
protection appears to be in place where required.

**Note:** This assessment is based on a single photograph and cannot \
evaluate all safety conditions. A comprehensive safety inspection should \
include physical examination of equipment, review of safety documentation, \
and assessment of conditions not visible in this image (e.g., electrical \
grounding, excavation shoring, chemical exposure).

---
**Site Compliance Status:** APPEARS COMPLIANT — No visible violations \
detected. Standard monitoring recommended.
"""

VIOLATION_TEMPLATE = """\
## SiteSafe Violation Report

**Violations Detected: 2**

### Violation 1: Missing Fall Protection
- **Regulation:** 29 CFR 1926.501(b)(1) — Unprotected Sides and Edges
- **Observation:** Worker visible at approximately 12 feet elevation on \
scaffold platform without guardrail system or personal fall arrest system.
- **Severity:** Serious
- **Confidence:** 0.92
- **Penalty Range:** $1,190 – $16,131 per instance
- **Corrective Action:** Immediately cease work at elevation. Install \
guardrail system with 42-inch (+/- 3 in.) top rail, midrail, and toeboard \
per 29 CFR 1926.502(b). Alternative: provide personal fall arrest system \
with full body harness anchored to capable anchorage per 29 CFR 1926.502(d).

### Violation 2: No Hard Hat in Active Zone
- **Regulation:** 29 CFR 1926.100(a) — Head Protection
- **Observation:** One worker in active work zone not wearing protective \
helmet while overhead work and material movement is in progress.
- **Severity:** Serious
- **Confidence:** 0.88
- **Penalty Range:** $1,190 – $16,131
- **Corrective Action:** Provide ANSI Z89.1-2014 compliant protective \
helmet. Enforce mandatory hard hat policy in all areas with danger of head \
injury from impact, falling or flying objects, or electrical shock.

---
**Site Compliance Status:** NON-COMPLIANT — 2 serious violations identified. \
Immediate corrective action required before work continues.
"""


def render_user_prompt(site_name: str = "", location: str = "", date: str = "") -> str:
    return ANALYSIS_PROMPT_TEMPLATE.substitute(
        site_name=site_name or "(unspecified)",
        location=location or "(unspecified)",
        date=date or "(unspecified)",
    )
