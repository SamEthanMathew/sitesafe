"""
SiteSafe — OSHA Regulation Database Builder
============================================

Builds a SQLite database (``data/osha_regulations.db``) containing the
OSHA construction-industry regulations (29 CFR Part 1926) that SiteSafe
cites when it detects a violation.

The database is the **single source of truth** for citations, penalty
ranges, and corrective-action language. Both the Gemma 4 function-calling
tool (``app/osha_tools.py``) and the training-data pipeline
(``data/build_training_data.py``) read from it.

Run::

    python data/build_osha_db.py

The script is idempotent — it drops and recreates the tables on every
invocation so you can edit the regulation list at the top of this file
and rebuild without leftover state.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sitesafe.osha_db")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).resolve().parent / "osha_regulations.db"

# Penalty figures track the OSHA 2024 inflation-adjusted civil-penalty
# schedule. ``min_penalty`` is the minimum **per violation** for that severity
# class, ``max_penalty`` is the maximum.
PENALTY_TABLE = [
    ("other", 0, 16131,
     "Other-than-serious — relates to job safety/health but unlikely to cause death or serious harm."),
    ("serious", 1190, 16131,
     "Serious — substantial probability of death or serious physical harm; employer knew or should have known."),
    ("willful", 11524, 161323,
     "Willful — employer intentionally and knowingly committed; or committed with plain indifference."),
    ("repeat", 1190, 161323,
     "Repeat — same or substantially similar violation cited within 5 years."),
]

# Fatal Four annual statistics (BLS 2024 + OSHA reporting).
FATAL_FOUR = [
    ("Falls", 33.5, 358,
     "Falls from elevation are the #1 killer in construction. Most are preventable with guardrails, "
     "personal fall arrest, or safety nets at 6+ feet."),
    ("Struck-By", 11.4, 122,
     "Struck-by hazards include vehicles backing up, swinging crane loads, falling materials, and "
     "flying objects. Often caused by lack of high-visibility apparel and poor traffic control plans."),
    ("Electrocution", 8.4, 90,
     "Electrocution typically results from contact with overhead power lines, energized circuits, "
     "or improperly grounded equipment. GFCI and lockout/tagout programs prevent the majority."),
    ("Caught-In/Between", 5.4, 58,
     "Caught-in/between hazards include trench collapses, equipment rollovers, and entanglement in "
     "rotating machinery. Excavation protective systems are required at 5+ feet."),
]


# ---------------------------------------------------------------------------
# Regulation records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Regulation:
    standard_id: str
    title: str
    subpart: str
    subpart_name: str
    requirement_text: str
    violation_type: str  # 'serious' | 'willful' | 'other' | 'repeat'
    min_penalty: int
    max_penalty: int
    corrective_action: str
    fatal_four_category: Optional[str]  # 'Falls' | 'Struck-By' | ... | None
    visual_indicators: str
    keywords: str  # space-separated, lowercased; used for keyword search


# Big regulation table. Edit here, rerun this script, the rest of the
# pipeline picks up changes automatically.
REGULATIONS: list[Regulation] = [
    # ------------------- FALL PROTECTION (Subpart M) -----------------------
    Regulation(
        standard_id="1926.501(b)(1)",
        title="Unprotected Sides and Edges",
        subpart="M",
        subpart_name="Fall Protection",
        requirement_text=(
            "Each employee on a walking/working surface (horizontal and vertical surface) "
            "with an unprotected side or edge which is 6 feet (1.8 m) or more above a lower "
            "level shall be protected from falling by the use of guardrail systems, safety net "
            "systems, or personal fall arrest systems."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Install a guardrail system with 42-inch (+/- 3 in.) top rail, midrail at midpoint, "
            "and toeboard per 29 CFR 1926.502(b); OR provide a personal fall arrest system with "
            "full body harness anchored to a capable anchorage per 29 CFR 1926.502(d); OR install "
            "a safety net within 30 feet of the working surface per 29 CFR 1926.502(c). Stop work "
            "at elevation until protection is in place."
        ),
        fatal_four_category="Falls",
        visual_indicators=(
            "Workers at elevated positions (roofs, scaffolds, platforms, leading edges) without "
            "visible guardrails, safety nets, or harness/lanyard systems. Look for lanyards "
            "trailing freely, missing top/mid rails, and unguarded roof edges."
        ),
        keywords="fall protection unprotected edge guardrail harness 6 feet elevation",
    ),
    Regulation(
        standard_id="1926.501(b)(2)",
        title="Leading Edges",
        subpart="M",
        subpart_name="Fall Protection",
        requirement_text=(
            "Each employee who is constructing a leading edge 6 feet or more above lower levels "
            "shall be protected from falling by guardrail systems, safety net systems, or personal "
            "fall arrest systems."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Where guardrails are infeasible, develop a written fall protection plan per "
            "29 CFR 1926.502(k) and use safety nets or PFAS. Train designated 'leading-edge' workers "
            "on the plan before assignment."
        ),
        fatal_four_category="Falls",
        visual_indicators=(
            "Workers placing decking, framing, or roofing at the open edge of a structure with no "
            "rope grabs, controlled-access zone tape, or anchored harnesses."
        ),
        keywords="leading edge fall protection decking roofing framing",
    ),
    Regulation(
        standard_id="1926.501(b)(3)",
        title="Hoist Areas",
        subpart="M",
        subpart_name="Fall Protection",
        requirement_text=(
            "Each employee in a hoist area shall be protected from falling 6 feet or more to lower "
            "levels by guardrail systems or personal fall arrest systems."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Where guardrails must be removed for material handling, employees leaning through "
            "the access opening must be tied off. Reinstall the guardrail immediately after the lift."
        ),
        fatal_four_category="Falls",
        visual_indicators="Open hoist openings with no chain gate, removable rail, or tied-off worker.",
        keywords="hoist area material lift opening guardrail",
    ),
    Regulation(
        standard_id="1926.501(b)(4)",
        title="Holes",
        subpart="M",
        subpart_name="Fall Protection",
        requirement_text=(
            "Each employee on walking/working surfaces shall be protected from falling through "
            "holes (including skylights) more than 6 feet above lower levels by personal fall "
            "arrest systems, covers, or guardrail systems erected around such holes."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Cover all floor holes with material capable of supporting twice the maximum load "
            "and label 'HOLE' or 'COVER' per 29 CFR 1926.502(i). Secure covers against displacement."
        ),
        fatal_four_category="Falls",
        visual_indicators=(
            "Uncovered floor holes, missing skylight protection, plywood covers without labelling "
            "or fasteners, openings that workers could step into."
        ),
        keywords="hole cover skylight floor opening fall through",
    ),
    Regulation(
        standard_id="1926.501(b)(10)",
        title="Roofing — Low-Slope Roofs",
        subpart="M",
        subpart_name="Fall Protection",
        requirement_text=(
            "Each employee engaged in roofing activities on low-slope roofs (slope <= 4:12) with "
            "unprotected sides and edges 6 feet or more above lower levels shall be protected from "
            "falling by guardrails, safety nets, PFAS, or a combination including warning line "
            "system and safety monitoring system."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Erect warning lines no closer than 6 feet from the edge per 29 CFR 1926.502(f) and "
            "designate a competent safety monitor per 29 CFR 1926.502(h) for work between the line "
            "and edge."
        ),
        fatal_four_category="Falls",
        visual_indicators="Workers on flat or low-slope roofs without parapet, warning line, or harness.",
        keywords="roofing low slope warning line monitor",
    ),
    Regulation(
        standard_id="1926.501(b)(11)",
        title="Steep Roofs",
        subpart="M",
        subpart_name="Fall Protection",
        requirement_text=(
            "Each employee on a steep roof (slope > 4:12) with unprotected sides and edges 6 feet "
            "or more above lower levels shall be protected from falling by guardrail systems with "
            "toeboards, safety net systems, or personal fall arrest systems."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Use a personal fall arrest system on steep roofs; warning lines and safety monitors "
            "are not permitted on slopes greater than 4:12."
        ),
        fatal_four_category="Falls",
        visual_indicators="Workers on pitched/steep roofs without harness lines back to a ridge anchor.",
        keywords="steep roof pitched harness ridge anchor",
    ),
    Regulation(
        standard_id="1926.501(b)(13)",
        title="Residential Construction",
        subpart="M",
        subpart_name="Fall Protection",
        requirement_text=(
            "Each employee engaged in residential construction activities 6 feet or more above lower "
            "levels shall be protected by guardrail systems, safety net systems, or personal fall "
            "arrest systems unless the employer can demonstrate it is infeasible or creates a greater "
            "hazard."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Default to PFAS for residential framing; if infeasible, develop a written, site-specific "
            "fall protection plan per OSHA Compliance Directive STD 03-11-002."
        ),
        fatal_four_category="Falls",
        visual_indicators=(
            "Framers walking on top plates, joists, or trusses without fall protection on a "
            "residential build."
        ),
        keywords="residential construction framing housing 6 feet",
    ),
    Regulation(
        standard_id="1926.501(b)(15)",
        title="Walking/Working Surfaces — Falling Objects",
        subpart="M",
        subpart_name="Fall Protection",
        requirement_text=(
            "When an employee is exposed to falling objects, the employer shall have each employee "
            "wear a hard hat and shall implement protection from falling objects."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Erect toeboards, screens, or guardrails per 29 CFR 1926.502(j); barricade the area "
            "below; or use canopies. Always require hard hats."
        ),
        fatal_four_category="Struck-By",
        visual_indicators=(
            "Tools, materials, or debris loose at edge with workers below; no toeboards or canopy."
        ),
        keywords="falling objects toeboard canopy barricade hard hat",
    ),
    Regulation(
        standard_id="1926.502(b)",
        title="Guardrail Systems — Construction Criteria",
        subpart="M",
        subpart_name="Fall Protection",
        requirement_text=(
            "Top edge height of top rails shall be 42 inches (+/- 3 in.) above the walking/working "
            "level. Midrails, screens, mesh, intermediate vertical members, or equivalent intermediate "
            "structural members shall be installed between the top edge and the walking/working surface "
            "when there is no wall at least 21 inches high."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Verify rails meet 200-lb top-rail load and 150-lb midrail load. Add toeboards 3.5 inches "
            "minimum where workers are below."
        ),
        fatal_four_category="Falls",
        visual_indicators=(
            "Guardrails too low (knee height), missing midrails, sagging cable rails without proper "
            "tension, or rails terminating short of corners."
        ),
        keywords="guardrail top rail midrail toeboard 42 inches",
    ),
    Regulation(
        standard_id="1926.502(c)",
        title="Safety Net Systems",
        subpart="M",
        subpart_name="Fall Protection",
        requirement_text=(
            "Safety nets shall be installed as close as practicable under the walking/working surface "
            "but never more than 30 feet below such level. They shall be inspected at least once a week "
            "and after any occurrence which could affect their integrity."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Confirm a successful drop test (400 lb sandbag) prior to use, and replace nets after a "
            "fall arrest event."
        ),
        fatal_four_category="Falls",
        visual_indicators="Safety nets sagging to ground, debris-filled, or installed > 30 ft below the work surface.",
        keywords="safety net drop test 30 feet inspection",
    ),
    Regulation(
        standard_id="1926.502(d)",
        title="Personal Fall Arrest Systems",
        subpart="M",
        subpart_name="Fall Protection",
        requirement_text=(
            "Personal fall arrest systems shall, when stopping a fall: (a) limit maximum arresting "
            "force on an employee to 1,800 lb when used with a body harness; (b) be rigged such that "
            "an employee can neither free fall more than 6 feet nor contact any lower level; and "
            "(c) be inspected prior to each use for wear, damage, and other deterioration."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Use a full-body harness (not body belts), shock-absorbing lanyard, and an anchorage "
            "capable of supporting 5,000 lb per attached worker. Train workers on inspection."
        ),
        fatal_four_category="Falls",
        visual_indicators=(
            "Workers wearing body belts (prohibited since 1998), missing harness chest straps, "
            "lanyards anchored to inadequate points (e.g., pipe), or worn webbing."
        ),
        keywords="personal fall arrest harness lanyard anchorage 5000 lb",
    ),
    Regulation(
        standard_id="1926.503(a)(1)",
        title="Fall Protection Training",
        subpart="M",
        subpart_name="Fall Protection",
        requirement_text=(
            "The employer shall provide a training program for each employee who might be exposed to "
            "fall hazards. The program shall enable each employee to recognize the hazards of falling "
            "and shall train each employee in the procedures to be followed in order to minimize these "
            "hazards."
        ),
        violation_type="other",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Document training with employee name, date, signature of trainer, and topics covered. "
            "Retrain when conditions change or competence is in question."
        ),
        fatal_four_category="Falls",
        visual_indicators="Indirect — observed by lack of competent behavior (improper anchorage, no harness use).",
        keywords="training fall protection hazard recognition",
    ),

    # ------------------- SCAFFOLDING (Subpart L) ---------------------------
    Regulation(
        standard_id="1926.451(a)(1)",
        title="Scaffold Capacity",
        subpart="L",
        subpart_name="Scaffolds",
        requirement_text=(
            "Each scaffold and scaffold component shall be capable of supporting, without failure, "
            "its own weight and at least 4 times the maximum intended load applied or transmitted to it."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Have a qualified person calculate or verify the rated capacity of the scaffold "
            "configuration and post it. Replace damaged frames/braces immediately."
        ),
        fatal_four_category="Falls",
        visual_indicators="Bowed planks, rusted/bent frames, scaffold loaded with material near or above rated capacity.",
        keywords="scaffold capacity 4x intended load",
    ),
    Regulation(
        standard_id="1926.451(b)(1)",
        title="Scaffold Platform Construction",
        subpart="L",
        subpart_name="Scaffolds",
        requirement_text=(
            "Each platform on all working levels of scaffolds shall be fully planked or decked between "
            "the front uprights and the guardrail supports. Planks shall be at least 18 inches wide "
            "with no more than a 1-inch gap, except where the work area requires otherwise."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Replace missing planks; gaps > 1 inch must be closed except where space is needed for "
            "tools/equipment."
        ),
        fatal_four_category="Falls",
        visual_indicators="Scaffold platforms with missing planks, large gaps, or split/cracked planks.",
        keywords="scaffold platform plank decking 18 inches",
    ),
    Regulation(
        standard_id="1926.451(e)(1)",
        title="Scaffold Access",
        subpart="L",
        subpart_name="Scaffolds",
        requirement_text=(
            "When scaffold platforms are more than 2 feet above or below a point of access, portable "
            "ladders, hook-on ladders, attachable ladders, stair towers (scaffold stairways/towers), "
            "stairway-type ladders, ramps, walkways, or integral prefabricated scaffold access shall "
            "be used. Cross braces shall not be used as a means of access."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Install a manufacturer-supplied stair tower or hook-on ladder. Prohibit climbing on "
            "cross-braces."
        ),
        fatal_four_category="Falls",
        visual_indicators="Workers seen climbing scaffold cross-braces with no ladder visible.",
        keywords="scaffold access ladder stair tower cross brace climbing",
    ),
    Regulation(
        standard_id="1926.451(g)(1)",
        title="Scaffold Fall Protection — General",
        subpart="L",
        subpart_name="Scaffolds",
        requirement_text=(
            "Each employee on a scaffold more than 10 feet above a lower level shall be protected "
            "from falling to that lower level."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Provide guardrails on all open sides and ends, or PFAS for suspension scaffolds and "
            "single-/two-point adjustable suspension."
        ),
        fatal_four_category="Falls",
        visual_indicators="Scaffold > 10 ft tall with no rails on the working face.",
        keywords="scaffold fall protection 10 feet guardrail",
    ),
    Regulation(
        standard_id="1926.451(g)(4)",
        title="Scaffold Guardrail Systems",
        subpart="L",
        subpart_name="Scaffolds",
        requirement_text=(
            "Guardrail systems installed to meet the requirements of this section shall comply with "
            "the following: top rails between 38 and 45 inches above the platform surface; midrails "
            "at approximately mid-height; toeboards 3.5 inches minimum where employees may be below."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Add or correct rail heights to 38–45 in. top rail and midrail. Add toeboards on platforms "
            "above walkways."
        ),
        fatal_four_category="Falls",
        visual_indicators="Scaffold rails too high or low, missing midrail, or no toeboard with workers below.",
        keywords="scaffold guardrail top rail midrail toeboard 38 45 inches",
    ),
    Regulation(
        standard_id="1926.451(f)(3)",
        title="Scaffold Erection by Competent Person",
        subpart="L",
        subpart_name="Scaffolds",
        requirement_text=(
            "Scaffolds shall be erected, moved, dismantled, or altered only under the supervision and "
            "direction of a competent person qualified in scaffold erection, moving, dismantling, or "
            "alteration."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Document the competent person's qualifications and have them sign off on each "
            "configuration."
        ),
        fatal_four_category="Falls",
        visual_indicators="No designated competent person on site; workers self-directing scaffold work.",
        keywords="scaffold erection competent person dismantling",
    ),
    Regulation(
        standard_id="1926.452(w)",
        title="Mobile Scaffolds — Inspection",
        subpart="L",
        subpart_name="Scaffolds",
        requirement_text=(
            "Scaffolds shall be inspected for visible defects by a competent person before each work "
            "shift, and after any occurrence which could affect a scaffold's structural integrity."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Implement a daily scaffold inspection checklist. Tag defective components red and "
            "remove from service."
        ),
        fatal_four_category="Falls",
        visual_indicators="Visibly damaged components in active use; no inspection tag on the scaffold.",
        keywords="scaffold inspection competent person shift",
    ),
    Regulation(
        standard_id="1926.454(a)",
        title="Scaffold User Training",
        subpart="L",
        subpart_name="Scaffolds",
        requirement_text=(
            "The employer shall have each employee who performs work while on a scaffold trained by "
            "a person qualified in the subject matter to recognize the hazards associated with the "
            "type of scaffold being used and to understand the procedures to control or minimize "
            "those hazards."
        ),
        violation_type="other",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Provide and document scaffold-user training before assignment to scaffold work."
        ),
        fatal_four_category="Falls",
        visual_indicators="Indirect — observed by improper use of scaffold or PFAS.",
        keywords="scaffold training user qualification",
    ),

    # ------------------- LADDERS (Subpart X) -------------------------------
    Regulation(
        standard_id="1926.1053(b)(1)",
        title="Ladder Extension Above Landing",
        subpart="X",
        subpart_name="Stairways and Ladders",
        requirement_text=(
            "When portable ladders are used for access to an upper landing surface, the ladder side "
            "rails shall extend at least 3 feet (0.9 m) above the upper landing surface to which the "
            "ladder is used to gain access."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Replace short ladders with longer ones; or install grab rails extending 3 ft above the "
            "landing."
        ),
        fatal_four_category="Falls",
        visual_indicators="Ladder top flush with or below landing edge — workers must transition awkwardly.",
        keywords="ladder extension 3 feet landing access",
    ),
    Regulation(
        standard_id="1926.1053(b)(4)",
        title="Securing Portable Ladders",
        subpart="X",
        subpart_name="Stairways and Ladders",
        requirement_text=(
            "Ladders shall be used only on stable and level surfaces unless secured to prevent "
            "accidental displacement."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Tie off the top of the ladder; use a stabilizer; or assign a person at the base to hold "
            "while in use."
        ),
        fatal_four_category="Falls",
        visual_indicators="Ladder on uneven ground, in mud, or visibly leaning; not tied off at top.",
        keywords="ladder secure stable level tied off",
    ),
    Regulation(
        standard_id="1926.1053(b)(13)",
        title="Stepladder Top — No Standing",
        subpart="X",
        subpart_name="Stairways and Ladders",
        requirement_text=(
            "The top or top step of a stepladder shall not be used as a step."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Provide an appropriately sized ladder; train workers on the second-from-top maximum step."
        ),
        fatal_four_category="Falls",
        visual_indicators="Worker standing on top cap of A-frame stepladder.",
        keywords="stepladder top step prohibited",
    ),
    Regulation(
        standard_id="1926.1053(b)(16)",
        title="Ladder Capacity",
        subpart="X",
        subpart_name="Stairways and Ladders",
        requirement_text=(
            "Ladders shall not be loaded beyond the maximum intended load for which they were built, "
            "or beyond their manufacturer's rated capacity."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Match ladder Type rating (IAA/IA/I/II/III) to combined worker + tool weight. Replace "
            "household-grade Type III ladders with Type IA on construction sites."
        ),
        fatal_four_category="Falls",
        visual_indicators="Light-duty (Type III) household ladder in use on site.",
        keywords="ladder capacity rating type maximum load",
    ),
    Regulation(
        standard_id="1926.1060(a)",
        title="Ladder User Training",
        subpart="X",
        subpart_name="Stairways and Ladders",
        requirement_text=(
            "The employer shall provide a training program for each employee using ladders and "
            "stairways. The program shall enable each employee to recognize hazards related to "
            "ladders and stairways."
        ),
        violation_type="other",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Provide documented ladder safety training; cover three-points-of-contact rule.",
        fatal_four_category="Falls",
        visual_indicators="Indirect — observed by misuse of ladders.",
        keywords="ladder training",
    ),

    # ------------------- PPE (Subpart E) -----------------------------------
    Regulation(
        standard_id="1926.95(a)",
        title="Personal Protective Equipment — General",
        subpart="E",
        subpart_name="Personal Protective and Life Saving Equipment",
        requirement_text=(
            "Protective equipment, including personal protective equipment for eyes, face, head, and "
            "extremities, protective clothing, respiratory devices, and protective shields and "
            "barriers, shall be provided, used, and maintained in a sanitary and reliable condition "
            "wherever it is necessary by reason of hazards of processes or environment."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Conduct a written hazard assessment per 29 CFR 1926.28; supply appropriate PPE at no "
            "cost to employees."
        ),
        fatal_four_category=None,
        visual_indicators="Workers in active hazard zones without high-vis vests, eye protection, or hard hats.",
        keywords="PPE personal protective equipment hazard assessment",
    ),
    Regulation(
        standard_id="1926.100(a)",
        title="Head Protection",
        subpart="E",
        subpart_name="Personal Protective and Life Saving Equipment",
        requirement_text=(
            "Employees working in areas where there is a possible danger of head injury from impact, "
            "or from falling or flying objects, or from electrical shock and burns, shall be protected "
            "by protective helmets."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Issue ANSI Z89.1-2014 compliant helmets; enforce mandatory hard-hat policy in all active "
            "work zones; replace damaged helmets immediately."
        ),
        fatal_four_category="Struck-By",
        visual_indicators=(
            "Workers without hard hats in areas with overhead activity, falling-object risk, "
            "or electrical exposure."
        ),
        keywords="hard hat helmet head protection ANSI Z89.1",
    ),
    Regulation(
        standard_id="1926.100(b)",
        title="Helmet Performance Criteria",
        subpart="E",
        subpart_name="Personal Protective and Life Saving Equipment",
        requirement_text=(
            "Protective helmets shall comply with the requirements of ANSI Z89.1-2014, ANSI Z89.1-2009, "
            "or ANSI Z89.1-2003."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Audit helmet inventory; remove non-compliant or expired helmets.",
        fatal_four_category="Struck-By",
        visual_indicators="Bump caps, novelty helmets, or visibly damaged hard hats in use.",
        keywords="helmet ANSI Z89.1 type class",
    ),
    Regulation(
        standard_id="1926.102(a)(1)",
        title="Eye and Face Protection",
        subpart="E",
        subpart_name="Personal Protective and Life Saving Equipment",
        requirement_text=(
            "The employer shall ensure that each affected employee uses appropriate eye or face "
            "protection when exposed to eye or face hazards from flying particles, molten metal, "
            "liquid chemicals, acids or caustic liquids, chemical gases or vapors, or potentially "
            "injurious light radiation."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Provide ANSI Z87.1-rated safety glasses, goggles, or face shields based on the hazard.",
        fatal_four_category=None,
        visual_indicators="Workers grinding, cutting, or welding without eye protection.",
        keywords="eye face protection safety glasses ANSI Z87.1",
    ),
    Regulation(
        standard_id="1926.28(a)",
        title="PPE Enforcement",
        subpart="C",
        subpart_name="General Safety and Health Provisions",
        requirement_text=(
            "The employer is responsible for requiring the wearing of appropriate personal protective "
            "equipment in all operations where there is an exposure to hazardous conditions or where "
            "this part indicates the need for using such equipment to reduce the hazards."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Adopt a written PPE policy with progressive discipline for non-compliance.",
        fatal_four_category=None,
        visual_indicators="Multiple workers without required PPE — pattern of non-enforcement.",
        keywords="PPE enforcement employer responsibility",
    ),
    Regulation(
        standard_id="1926.95(c)",
        title="PPE Hazard Assessment",
        subpart="E",
        subpart_name="Personal Protective and Life Saving Equipment",
        requirement_text=(
            "The employer shall verify that the required workplace hazard assessment has been performed "
            "through a written certification that identifies the workplace evaluated, the person "
            "certifying that the evaluation has been performed, the date(s) of the hazard assessment, "
            "and which identifies the document as a certification of hazard assessment."
        ),
        violation_type="other",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Maintain a written PPE hazard assessment for each task on site, signed and dated.",
        fatal_four_category=None,
        visual_indicators="Indirect — verified via paperwork audit.",
        keywords="PPE hazard assessment written certification",
    ),

    # ------------------- ELECTRICAL (Subpart K) ----------------------------
    Regulation(
        standard_id="1926.405(a)(2)(ii)(I)",
        title="GFCI on Temporary 120V Outlets",
        subpart="K",
        subpart_name="Electrical",
        requirement_text=(
            "All 120-volt, single-phase 15- and 20-ampere receptacle outlets on construction sites "
            "which are not part of the permanent wiring of the building or structure and which are in "
            "use by employees shall have ground-fault circuit interrupters for personnel protection."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Replace temporary outlets with GFCI-protected units, or use portable in-line GFCIs.",
        fatal_four_category="Electrocution",
        visual_indicators="Temporary cords and outlets without GFCI breaker; standard receptacles in use.",
        keywords="GFCI ground fault temporary 120V outlet",
    ),
    Regulation(
        standard_id="1926.405(g)(2)(iv)",
        title="Flexible Cord Splices",
        subpart="K",
        subpart_name="Electrical",
        requirement_text=(
            "Flexible cords shall be used only in continuous lengths without splice or tap. Hard-service "
            "flexible cords No. 12 or larger may be repaired if spliced so that the splice retains the "
            "insulation, outer sheath properties, and usage characteristics of the cord being spliced."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Replace spliced cords with continuous-length cords. Tag damaged cords out of service.",
        fatal_four_category="Electrocution",
        visual_indicators="Extension cords with electrical tape splices, exposed conductors, or cracked sheaths.",
        keywords="flexible cord splice tape damage",
    ),
    Regulation(
        standard_id="1926.416(a)(1)",
        title="Working Near Energized Parts",
        subpart="K",
        subpart_name="Electrical",
        requirement_text=(
            "No employer shall permit an employee to work in such proximity to any part of an electric "
            "power circuit that the employee could contact the circuit in the course of work, unless "
            "the employee is protected against electric shock by deenergizing the circuit and grounding "
            "it or by guarding it effectively by insulation or other means."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="De-energize circuits and apply lockout/tagout per 29 CFR 1926.417 before work begins.",
        fatal_four_category="Electrocution",
        visual_indicators="Workers in panels with live conductors exposed; no LOTO tags; no insulating barriers.",
        keywords="energized parts electrical shock deenergize grounding",
    ),
    Regulation(
        standard_id="1926.416(a)(3)",
        title="Electrical Barriers",
        subpart="K",
        subpart_name="Electrical",
        requirement_text=(
            "Barriers or other means of guarding shall be provided to ensure that workspace for "
            "electrical equipment will not be used as a passageway during periods when energized parts "
            "of electrical equipment are exposed."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Erect physical barriers and 'DANGER — Energized Equipment' signs around exposed live work.",
        fatal_four_category="Electrocution",
        visual_indicators="Open electrical panels in walking corridors with no barricade.",
        keywords="electrical barriers guarding workspace",
    ),
    Regulation(
        standard_id="1926.404(b)(1)",
        title="Equipment Grounding",
        subpart="K",
        subpart_name="Electrical",
        requirement_text=(
            "The employer shall use either ground-fault circuit interrupters or an assured equipment "
            "grounding conductor program covering all cord sets, receptacles which are not part of the "
            "permanent wiring of the building or structure, and equipment connected by cord and plug "
            "which are available for use or used by employees."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Adopt either GFCI-on-everything or document an assured grounding conductor program.",
        fatal_four_category="Electrocution",
        visual_indicators="Two-pronged adapters in use; missing ground pin on plugs.",
        keywords="grounding equipment AEGP GFCI cord plug",
    ),
    Regulation(
        standard_id="1926.431",
        title="Maintenance of Electrical Equipment",
        subpart="K",
        subpart_name="Electrical",
        requirement_text=(
            "The employer shall ensure that electrical equipment is maintained in a safe condition. "
            "Wiring methods, components, and equipment shall be free from recognized hazards that are "
            "likely to cause death or serious physical harm."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Establish a preventive maintenance schedule; tag and remove damaged equipment immediately.",
        fatal_four_category="Electrocution",
        visual_indicators="Charred outlets, missing covers, exposed conductors.",
        keywords="electrical maintenance equipment safe condition",
    ),
    Regulation(
        standard_id="1926.417(a)",
        title="Lockout / Tagout",
        subpart="K",
        subpart_name="Electrical",
        requirement_text=(
            "Controls that are to be deactivated during the course of work on energized or "
            "deenergized equipment or circuits shall be tagged."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Implement a written LOTO program; train authorized employees; audit annually.",
        fatal_four_category="Electrocution",
        visual_indicators="Disconnect switches without LOTO devices when work is in progress on the equipment.",
        keywords="lockout tagout LOTO disconnect tagged",
    ),
    Regulation(
        standard_id="1926.416(e)(1)",
        title="Power Line Clearance",
        subpart="K",
        subpart_name="Electrical",
        requirement_text=(
            "When operating equipment near energized overhead lines, a minimum clearance of 10 feet "
            "shall be maintained for lines rated 50 kV or below; greater clearances are required for "
            "higher voltages."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=161323,
        corrective_action=(
            "De-energize lines, install insulating barriers, or maintain documented minimum clearance "
            "with a dedicated spotter."
        ),
        fatal_four_category="Electrocution",
        visual_indicators="Crane booms, scaffolding, or aerial lifts within 10 ft of overhead distribution lines.",
        keywords="power line overhead clearance 10 feet 50kV crane",
    ),

    # ------------------- EXCAVATIONS (Subpart P) ---------------------------
    Regulation(
        standard_id="1926.651(c)(2)",
        title="Locating Underground Utilities",
        subpart="P",
        subpart_name="Excavations",
        requirement_text=(
            "Estimated location of utility installations, such as sewer, telephone, fuel, electric, "
            "water lines, or any other underground installations that reasonably may be expected to "
            "be encountered during excavation work, shall be determined prior to opening an excavation."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Call 811 (or local one-call). Document utility tickets before mobilizing equipment.",
        fatal_four_category="Caught-In/Between",
        visual_indicators="Excavation in progress without paint marks or flag indicators for utilities.",
        keywords="underground utilities locate 811 excavation",
    ),
    Regulation(
        standard_id="1926.651(j)(2)",
        title="Excavation Egress",
        subpart="P",
        subpart_name="Excavations",
        requirement_text=(
            "A stairway, ladder, ramp, or other safe means of egress shall be located in trench "
            "excavations that are 4 feet or more in depth so as to require no more than 25 feet of "
            "lateral travel for employees."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Place ladders/ramps such that no worker is more than 25 ft of lateral travel from egress.",
        fatal_four_category="Caught-In/Between",
        visual_indicators="Long trench with workers but only one ladder at the far end.",
        keywords="excavation trench egress ladder 25 feet 4 feet",
    ),
    Regulation(
        standard_id="1926.651(k)(1)",
        title="Daily Excavation Inspections",
        subpart="P",
        subpart_name="Excavations",
        requirement_text=(
            "Daily inspections of excavations, the adjacent areas, and protective systems shall be "
            "made by a competent person for evidence of a situation that could result in possible "
            "cave-ins, indications of failure of protective systems, hazardous atmospheres, or other "
            "hazardous conditions."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Document daily inspections; halt work when conditions change (rain, vibration, additional "
            "loading)."
        ),
        fatal_four_category="Caught-In/Between",
        visual_indicators="Work proceeding after rainfall with no documented re-inspection of trench walls.",
        keywords="excavation daily inspection competent person cave in",
    ),
    Regulation(
        standard_id="1926.652(a)(1)",
        title="Excavation Protective Systems",
        subpart="P",
        subpart_name="Excavations",
        requirement_text=(
            "Each employee in an excavation shall be protected from cave-ins by an adequate protective "
            "system designed in accordance with paragraph (b) or (c) of this section. Excavations less "
            "than 5 feet in depth and examined by a competent person providing no indication of a "
            "potential cave-in are excepted."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Use sloping/benching, shoring, or shielding (trench box) for any trench 5 ft or deeper. "
            "Refer to Appendices A–F for soil classification."
        ),
        fatal_four_category="Caught-In/Between",
        visual_indicators="Trench > 5 ft with vertical walls, no shoring/box, no benching visible.",
        keywords="excavation cave in protective system trench box shoring sloping 5 feet",
    ),
    Regulation(
        standard_id="1926.651(i)(1)",
        title="Excavation Water Accumulation",
        subpart="P",
        subpart_name="Excavations",
        requirement_text=(
            "Employees shall not work in excavations in which there is accumulated water, or in "
            "excavations in which water is accumulating, unless adequate precautions have been taken "
            "to protect employees against the hazards posed by water accumulation."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action=(
            "Provide special support, shield systems, water removal (pumps), or safety harnesses with "
            "lifelines."
        ),
        fatal_four_category="Caught-In/Between",
        visual_indicators="Standing water at the bottom of an excavation while work continues.",
        keywords="excavation water accumulation pump safety harness",
    ),

    # ------------------- STRUCK-BY / MATERIAL HANDLING ---------------------
    Regulation(
        standard_id="1926.602(a)(9)(ii)",
        title="Earthmoving Equipment Seat Belts",
        subpart="O",
        subpart_name="Motor Vehicles, Mechanized Equipment, and Marine Operations",
        requirement_text=(
            "Earthmoving equipment shall be equipped with a seat belt as required by Society of "
            "Automotive Engineers, J386-1969, Seat Belts for Construction Equipment."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Repair or replace missing/damaged seat belts before equipment is returned to service.",
        fatal_four_category="Struck-By",
        visual_indicators="Operators in dozers, loaders, scrapers without visible seat belt use.",
        keywords="seat belt earthmoving equipment dozer loader scraper",
    ),
    Regulation(
        standard_id="1926.550(a)(1)",
        title="Crane Annual Inspection",
        subpart="N",
        subpart_name="Helicopters, Hoists, Elevators, and Conveyors",
        requirement_text=(
            "The employer shall comply with the manufacturer's specifications and limitations applicable "
            "to the operation of any and all cranes and derricks. Where manufacturer's specifications "
            "are not available, the limitations assigned to the equipment shall be based on the "
            "determinations of a qualified engineer competent in this field. Annual thorough "
            "inspections by a competent person are required."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Maintain annual inspection records on the crane; retain for the equipment's life.",
        fatal_four_category="Struck-By",
        visual_indicators="Crane in service without visible annual inspection sticker / current paperwork.",
        keywords="crane annual inspection competent person",
    ),
    Regulation(
        standard_id="1926.550(a)(5)",
        title="Crane Load Limits",
        subpart="N",
        subpart_name="Helicopters, Hoists, Elevators, and Conveyors",
        requirement_text=(
            "The employer shall designate a competent person who shall inspect all machinery and "
            "equipment prior to each use, and during use, to make sure it is in safe operating "
            "condition. The rated load capacity shall not be exceeded."
        ),
        violation_type="willful",
        min_penalty=1190,
        max_penalty=161323,
        corrective_action="Use load charts; weigh loads when uncertain; install load-moment indicators.",
        fatal_four_category="Struck-By",
        visual_indicators="Crane operating in suspect-overload posture; no load chart in cab.",
        keywords="crane load limit rated capacity overload",
    ),
    Regulation(
        standard_id="1926.250(a)(1)",
        title="Material Storage",
        subpart="H",
        subpart_name="Materials Handling, Storage, Use, and Disposal",
        requirement_text=(
            "All materials stored in tiers shall be stacked, racked, blocked, interlocked, or otherwise "
            "secured to prevent sliding, falling, or collapse."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Band, strap, or block stacked materials. Cap maximum stack height per material spec.",
        fatal_four_category="Struck-By",
        visual_indicators="Bricks, lumber, or pipes stacked unsecured; pallets leaning.",
        keywords="material storage stacking blocking sliding falling",
    ),
    Regulation(
        standard_id="1926.251(a)(1)",
        title="Rigging Equipment Inspection",
        subpart="H",
        subpart_name="Materials Handling, Storage, Use, and Disposal",
        requirement_text=(
            "Rigging equipment for material handling shall be inspected prior to use on each shift "
            "and as necessary during its use to ensure that it is safe."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Implement pre-shift rigging inspections; tag-and-remove defective gear.",
        fatal_four_category="Struck-By",
        visual_indicators="Slings, hooks, or shackles in use without inspection tag.",
        keywords="rigging inspection sling shackle hook shift",
    ),
    Regulation(
        standard_id="1926.251(c)(4)(i)",
        title="Wire Rope Removal Criteria",
        subpart="H",
        subpart_name="Materials Handling, Storage, Use, and Disposal",
        requirement_text=(
            "Wire rope shall be taken out of service when in any length of 8 diameters there are "
            "ten randomly distributed broken wires, or five broken wires in one strand."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Discard wire rope meeting any removal criterion; do not splice or repair.",
        fatal_four_category="Struck-By",
        visual_indicators="Visible broken wires, kinks, birdcaging, or severe corrosion on wire rope.",
        keywords="wire rope broken wires removal criteria",
    ),

    # ------------------- HOUSEKEEPING & GENERAL ----------------------------
    Regulation(
        standard_id="1926.25(a)",
        title="Housekeeping",
        subpart="C",
        subpart_name="General Safety and Health Provisions",
        requirement_text=(
            "During the course of construction, alteration, or repairs, form and scrap lumber with "
            "protruding nails, and all other debris, shall be kept cleared from work areas, passageways, "
            "and stairs in and around buildings or other structures."
        ),
        violation_type="other",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Daily clean-up; designate scrap bins; bend or remove protruding nails immediately.",
        fatal_four_category=None,
        visual_indicators="Lumber piles in walkways, protruding nails, debris on stairs.",
        keywords="housekeeping debris nails passageway stairs",
    ),
    Regulation(
        standard_id="1926.20(b)(1)",
        title="Accident Prevention Program",
        subpart="C",
        subpart_name="General Safety and Health Provisions",
        requirement_text=(
            "It shall be the responsibility of the employer to initiate and maintain such programs "
            "as may be necessary to comply with this part."
        ),
        violation_type="other",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Document a written safety program; review with all new hires.",
        fatal_four_category=None,
        visual_indicators="Indirect — verified by paperwork audit.",
        keywords="accident prevention program written safety",
    ),
    Regulation(
        standard_id="1926.20(b)(2)",
        title="Frequent and Regular Inspections",
        subpart="C",
        subpart_name="General Safety and Health Provisions",
        requirement_text=(
            "Such programs shall provide for frequent and regular inspections of the job sites, "
            "materials, and equipment to be made by competent persons designated by the employers."
        ),
        violation_type="other",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Schedule and document at least weekly competent-person inspections.",
        fatal_four_category=None,
        visual_indicators="Indirect — verified by inspection logs.",
        keywords="frequent regular inspections competent person",
    ),
    Regulation(
        standard_id="1926.200(b)(1)",
        title="Danger Signs",
        subpart="G",
        subpart_name="Signs, Signals, and Barricades",
        requirement_text=(
            "Danger signs shall be used only where an immediate hazard exists, and shall have red "
            "as the predominating color for the upper panel."
        ),
        violation_type="other",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Post compliant danger signs at immediate-hazard locations.",
        fatal_four_category=None,
        visual_indicators="Hazard zones (live electrical, fall edges) with no danger signage.",
        keywords="danger sign red immediate hazard",
    ),
    Regulation(
        standard_id="1926.200(c)(1)",
        title="Caution Signs",
        subpart="G",
        subpart_name="Signs, Signals, and Barricades",
        requirement_text=(
            "Caution signs shall be used only to warn against potential hazards or to caution against "
            "unsafe practices."
        ),
        violation_type="other",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Use yellow caution signs for non-immediate hazards (slippery floor, low clearance).",
        fatal_four_category=None,
        visual_indicators="Potential-hazard areas without caution signage.",
        keywords="caution sign yellow potential hazard",
    ),
    Regulation(
        standard_id="1926.200(g)(1)",
        title="Accident Prevention Tags",
        subpart="G",
        subpart_name="Signs, Signals, and Barricades",
        requirement_text=(
            "Accident prevention tags shall be used as a temporary means of warning employees of an "
            "existing hazard, such as defective tools, equipment, etc. They shall not be used in place "
            "of, or as a substitute for, accident prevention signs."
        ),
        violation_type="other",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Tag defective tools and equipment 'Out of Service' immediately upon discovery.",
        fatal_four_category=None,
        visual_indicators="Damaged tools in active use without tags.",
        keywords="accident prevention tag defective tool out of service",
    ),

    # ------------------- FIRE PROTECTION (Subpart F) ----------------------
    Regulation(
        standard_id="1926.150(a)",
        title="Fire Protection Plan",
        subpart="F",
        subpart_name="Fire Protection and Prevention",
        requirement_text=(
            "The employer shall be responsible for the development of a fire protection program to be "
            "followed throughout all phases of the construction and demolition work, and shall provide "
            "for the firefighting equipment as specified in this subpart."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Adopt a written fire-prevention plan; train workers on hot-work permits.",
        fatal_four_category=None,
        visual_indicators="Hot work in progress without fire watch, blankets, or extinguisher nearby.",
        keywords="fire protection plan hot work prevention",
    ),
    Regulation(
        standard_id="1926.150(c)(1)(i)",
        title="Fire Extinguisher Distribution",
        subpart="F",
        subpart_name="Fire Protection and Prevention",
        requirement_text=(
            "A fire extinguisher, rated not less than 2A, shall be provided for each 3,000 square feet "
            "of the protected building area, or major fraction thereof. Travel distance from any point "
            "of the protected area to the nearest fire extinguisher shall not exceed 100 feet."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Map and post extinguisher locations; inspect monthly.",
        fatal_four_category=None,
        visual_indicators="Large work area with no visible fire extinguisher within 100 ft.",
        keywords="fire extinguisher 2A 100 feet travel distance",
    ),
    Regulation(
        standard_id="1926.152(a)(1)",
        title="Flammable Liquid Containers",
        subpart="F",
        subpart_name="Fire Protection and Prevention",
        requirement_text=(
            "Only approved containers and portable tanks shall be used for storage and handling of "
            "flammable liquids. Approved metal safety cans shall be used for the handling and use of "
            "flammable liquids in quantities greater than 1 gallon."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Replace plastic jugs/buckets with UL-listed safety cans for fuel and solvents.",
        fatal_four_category=None,
        visual_indicators="Gasoline in plastic milk jugs or open buckets.",
        keywords="flammable liquid safety can UL listed gasoline",
    ),

    # ------------------- STAIRWAYS (Subpart X) ----------------------------
    Regulation(
        standard_id="1926.1052(a)(1)",
        title="Stairways Required",
        subpart="X",
        subpart_name="Stairways and Ladders",
        requirement_text=(
            "A stairway or ladder shall be provided at all personnel points of access where there is "
            "a break in elevation of 19 inches or more, and no ramp, runway, sloped embankment, or "
            "personnel hoist is provided."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Install temporary stairs or ladders at every elevation change of 19+ inches.",
        fatal_four_category="Falls",
        visual_indicators="Workers jumping from elevated platforms instead of using a stair or ladder.",
        keywords="stairway ladder 19 inches elevation change",
    ),
    Regulation(
        standard_id="1926.1052(c)(1)",
        title="Stair Rails and Handrails",
        subpart="X",
        subpart_name="Stairways and Ladders",
        requirement_text=(
            "Stairways having four or more risers or rising more than 30 inches, whichever is less, "
            "shall be equipped with at least one handrail and one stair rail system along each "
            "unprotected side or edge."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Install stair rails and handrails on temporary stairs above 30 inches / 4 risers.",
        fatal_four_category="Falls",
        visual_indicators="Temporary stairs without rails on the open side.",
        keywords="stair rail handrail 4 risers 30 inches",
    ),

    # ------------------- TRAINING (general) -------------------------------
    Regulation(
        standard_id="1926.21(b)(2)",
        title="Hazard Recognition Training",
        subpart="C",
        subpart_name="General Safety and Health Provisions",
        requirement_text=(
            "The employer shall instruct each employee in the recognition and avoidance of unsafe "
            "conditions and the regulations applicable to his work environment to control or eliminate "
            "any hazards or other exposure to illness or injury."
        ),
        violation_type="other",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Document hazard-recognition training for every employee, including OSHA 10/30 if applicable.",
        fatal_four_category=None,
        visual_indicators="Indirect — verified by training records.",
        keywords="hazard recognition training instruction",
    ),
    Regulation(
        standard_id="1926.21(b)(6)(i)",
        title="Confined Space Training",
        subpart="C",
        subpart_name="General Safety and Health Provisions",
        requirement_text=(
            "All employees required to enter into confined or enclosed spaces shall be instructed "
            "as to the nature of the hazards involved, the necessary precautions to be taken, and in "
            "the use of protective and emergency equipment required."
        ),
        violation_type="serious",
        min_penalty=1190,
        max_penalty=16131,
        corrective_action="Implement a confined-space entry program; train entrants, attendants, and supervisors.",
        fatal_four_category=None,
        visual_indicators="Workers entering tanks, vaults, or trenches without permits or attendants.",
        keywords="confined space training entry permit attendant",
    ),
]


# ---------------------------------------------------------------------------
# Schema + writer
# ---------------------------------------------------------------------------

SCHEMA = """
DROP TABLE IF EXISTS regulations;
DROP TABLE IF EXISTS penalty_schedule;
DROP TABLE IF EXISTS fatal_four_stats;

CREATE TABLE regulations (
    standard_id          TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    subpart              TEXT NOT NULL,
    subpart_name         TEXT NOT NULL,
    requirement_text     TEXT NOT NULL,
    violation_type       TEXT NOT NULL CHECK(violation_type IN ('serious','willful','other','repeat')),
    min_penalty          INTEGER NOT NULL,
    max_penalty          INTEGER NOT NULL,
    corrective_action    TEXT NOT NULL,
    fatal_four_category  TEXT CHECK(fatal_four_category IN ('Falls','Struck-By','Electrocution','Caught-In/Between') OR fatal_four_category IS NULL),
    visual_indicators    TEXT,
    keywords             TEXT
);

CREATE INDEX idx_regulations_subpart   ON regulations(subpart);
CREATE INDEX idx_regulations_fatal_four ON regulations(fatal_four_category);
CREATE INDEX idx_regulations_violation_type ON regulations(violation_type);

CREATE TABLE penalty_schedule (
    violation_type TEXT PRIMARY KEY,
    min_penalty    INTEGER NOT NULL,
    max_penalty    INTEGER NOT NULL,
    description    TEXT NOT NULL
);

CREATE TABLE fatal_four_stats (
    category       TEXT PRIMARY KEY,
    pct_of_deaths  REAL NOT NULL,
    annual_deaths  INTEGER NOT NULL,
    description    TEXT NOT NULL
);
"""


def build_database(db_path: Path = DB_PATH) -> None:
    """(Re)build the OSHA SQLite database from the in-file regulation table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing database: %s", db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)

        conn.executemany(
            """
            INSERT INTO regulations (
                standard_id, title, subpart, subpart_name, requirement_text,
                violation_type, min_penalty, max_penalty, corrective_action,
                fatal_four_category, visual_indicators, keywords
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.standard_id,
                    r.title,
                    r.subpart,
                    r.subpart_name,
                    r.requirement_text,
                    r.violation_type,
                    r.min_penalty,
                    r.max_penalty,
                    r.corrective_action,
                    r.fatal_four_category,
                    r.visual_indicators,
                    r.keywords.lower(),
                )
                for r in REGULATIONS
            ],
        )

        conn.executemany(
            "INSERT INTO penalty_schedule VALUES (?, ?, ?, ?)",
            PENALTY_TABLE,
        )

        conn.executemany(
            "INSERT INTO fatal_four_stats VALUES (?, ?, ?, ?)",
            FATAL_FOUR,
        )

        conn.commit()
    finally:
        conn.close()

    log.info(
        "Inserted %d regulations across %d Subparts.",
        len(REGULATIONS),
        len({r.subpart for r in REGULATIONS}),
    )


# ---------------------------------------------------------------------------
# Convenience query functions (re-used by the function-calling tools)
# ---------------------------------------------------------------------------

def query_regulation(
    standard_id: Optional[str] = None,
    keyword: Optional[str] = None,
    fatal_four_category: Optional[str] = None,
    db_path: Path = DB_PATH,
    limit: int = 5,
) -> list[dict]:
    """Look up regulations by exact standard ID, keyword search, or Fatal Four category.

    Returns a list of row dicts. ``standard_id`` matches are returned first;
    keyword matches use a SQL LIKE OR-search across title, requirement_text,
    visual_indicators, and the keywords column.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"OSHA DB not found at {db_path}. Run `python data/build_osha_db.py` first."
        )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if standard_id:
            cur = conn.execute(
                "SELECT * FROM regulations WHERE standard_id = ?",
                (standard_id,),
            )
            rows = [dict(r) for r in cur.fetchall()]
            if rows:
                return rows[:limit]

        if fatal_four_category:
            cur = conn.execute(
                "SELECT * FROM regulations WHERE fatal_four_category = ? LIMIT ?",
                (fatal_four_category, limit),
            )
            rows = [dict(r) for r in cur.fetchall()]
            if rows:
                return rows

        if keyword:
            kw = f"%{keyword.lower()}%"
            cur = conn.execute(
                """
                SELECT * FROM regulations
                WHERE LOWER(title)             LIKE ?
                   OR LOWER(requirement_text)  LIKE ?
                   OR LOWER(visual_indicators) LIKE ?
                   OR keywords                 LIKE ?
                LIMIT ?
                """,
                (kw, kw, kw, kw, limit),
            )
            return [dict(r) for r in cur.fetchall()]

        return []
    finally:
        conn.close()


def get_fatal_four_info(category: str, db_path: Path = DB_PATH) -> Optional[dict]:
    """Return a dict for a Fatal Four category, including the relevant regulations."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        stat = conn.execute(
            "SELECT * FROM fatal_four_stats WHERE category = ?",
            (category,),
        ).fetchone()
        if stat is None:
            return None

        regs = conn.execute(
            "SELECT standard_id, title FROM regulations WHERE fatal_four_category = ?",
            (category,),
        ).fetchall()

        return {
            "category": stat["category"],
            "pct_of_deaths": stat["pct_of_deaths"],
            "annual_deaths": stat["annual_deaths"],
            "description": stat["description"],
            "primary_regulations": [dict(r) for r in regs],
        }
    finally:
        conn.close()


def get_penalty_info(violation_type: str, db_path: Path = DB_PATH) -> Optional[dict]:
    """Return penalty range + description for a violation severity class."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM penalty_schedule WHERE violation_type = ?",
            (violation_type.lower(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    log.info("Running smoke test against the freshly built DB")
    rows = query_regulation(standard_id="1926.501(b)(1)")
    assert rows, "Expected to find 1926.501(b)(1)"
    assert rows[0]["fatal_four_category"] == "Falls"

    kw = query_regulation(keyword="hard hat")
    assert any(r["standard_id"] == "1926.100(a)" for r in kw), "Expected 1926.100(a) for 'hard hat'"

    info = get_fatal_four_info("Falls")
    assert info and info["primary_regulations"], "Expected Fatal Four info for Falls"

    pen = get_penalty_info("serious")
    assert pen and pen["max_penalty"] == 16131
    log.info("Smoke test OK")


if __name__ == "__main__":
    build_database()
    _smoke_test()
