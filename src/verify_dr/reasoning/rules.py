"""M3 — the ICDR reasoner.

Deterministic Python, **no learned parameters**. Every output traceable to a rule
that fired, and every rule quotable in the thesis.

    R1  no lesions of any type                       -> 0
    R2  MA present, no haemorrhage or exudate        -> 1  mild NPDR
    R3  MA + haemorrhage or exudate, 4-2-1 not met   -> 2  moderate NPDR
    R4  4-2-1 severe criterion met                   -> 3  severe NPDR
    R5  neovascularisation or vitreous haemorrhage   -> 4  PDR

**What this module refuses to do is the point.** Full ICDR fires R4 on venous
beading in >=2 quadrants or IRMA in >=1, and R5 on neovascularisation. None of
those three are annotated on any Kaggle-reachable dataset, so M2 cannot see them
and M3 does not pretend otherwise: they are emitted as `unobservable`, and
`max_excludable_grade` states the highest grade the evidence can actually rule
out. Existing systems have the same blind spots -- a single 45-degree
macula-centred field cannot support real four-quadrant assessment either -- and
emit a confident grade anyway. Naming the blind spot turns a dataset limitation
into a safety property.

R4 is further reduced to its **haemorrhage arm only**, and needs quadrants, which
need M2b's geometry. When the geometry is unavailable or untrusted, R4 declines
to fire and says so: the count-only fallback that docs/00_START_HERE.md names for
exactly this case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from verify_dr.reasoning.facts import Facts

#: Findings ICDR needs that no available dataset annotates.
UNOBSERVABLE = ("venous_beading", "IRMA", "neovascularisation")

#: The 4-2-1 severe-NPDR criterion, haemorrhage arm: >= 20 intraretinal
#: haemorrhages in each of the four quadrants.
SEVERE_HAEMORRHAGE_PER_QUADRANT = 20


@dataclass
class Verdict:
    evidence_grade: int
    rules_fired: List[str]
    unobservable: List[str] = field(default_factory=lambda: list(UNOBSERVABLE))
    max_excludable_grade: int = 2
    abstain_upward: bool = True
    quadrants_used: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "evidence_grade": self.evidence_grade,
            "rules_fired": self.rules_fired,
            "unobservable": self.unobservable,
            "max_excludable_grade": self.max_excludable_grade,
            "abstain_upward": self.abstain_upward,
            "quadrants_used": self.quadrants_used,
            "notes": self.notes,
        }


def grade(facts: Facts,
          severe_threshold: int = SEVERE_HAEMORRHAGE_PER_QUADRANT) -> Verdict:
    """Apply R1-R5 to one image's facts."""
    ma = facts.present("microaneurysm")
    he = facts.present("haemorrhage")
    ex = facts.present("hard_exudate") or facts.present("soft_exudate")
    notes = list(facts.notes)

    # R5 cannot fire: neovascularisation is unobservable here. Stated rather than
    # silently skipped, so a reader is not left to infer that grade 4 was ruled out.
    notes.append("R5 not evaluable: neovascularisation is not annotated in any "
                 "available dataset")

    # ---- R4, the only rule that needs a coordinate frame -------------------
    severe = False
    quadrants_used = False
    if facts.quadrant_counts is not None:
        per_quadrant = facts.quadrant_counts.get("haemorrhage", {})
        if per_quadrant:
            quadrants_used = True
            severe = all(n >= severe_threshold for n in per_quadrant.values())
            notes.append(
                f"R4 evaluated on haemorrhages per quadrant {per_quadrant} "
                f"against >= {severe_threshold} in each; "
                "venous beading and IRMA arms unobservable")
    else:
        notes.append("R4 declined: no trusted quadrant frame, so the 4-2-1 "
                     "criterion cannot be assessed. Grading is count-only.")

    # ---- the ladder --------------------------------------------------------
    if severe:
        fired = [f"R4: >= {severe_threshold} haemorrhages in each of 4 quadrants "
                 "(haemorrhage arm only)"]
        evidence_grade = 3
    elif ma and (he or ex):
        fired = ["R3: MA plus haemorrhage or exudate, 4-2-1 haemorrhage criterion "
                 + ("not met" if quadrants_used else "not assessable")]
        evidence_grade = 2
    elif ma:
        fired = ["R2: MA present, no haemorrhage or exudate"]
        evidence_grade = 1
    elif he or ex:
        # Not a clean ICDR rung: ICDR's ladder presumes MA accompanies anything
        # further along. Segmentation can miss MA while finding a haemorrhage, so
        # this is recorded as R3 with the discrepancy noted rather than forced to 0.
        fired = ["R3*: haemorrhage or exudate without detected MA - graded as "
                 "moderate, since ICDR has no rung for this and M2a misses MA more "
                 "often than larger lesions"]
        evidence_grade = 2
    else:
        fired = ["R1: no lesions of any type detected"]
        evidence_grade = 0

    # ---- what can honestly be ruled out ------------------------------------
    #
    # Without quadrants, nothing above moderate can be excluded: severe NPDR is
    # defined by a criterion that was not assessed. With quadrants, the
    # haemorrhage arm was checked but beading and IRMA were not, so severe still
    # cannot be excluded. Either way grade 4 is out of reach.
    max_excludable = 3 if (quadrants_used and evidence_grade >= 3) else 2

    return Verdict(
        evidence_grade=evidence_grade,
        rules_fired=fired,
        max_excludable_grade=max_excludable,
        abstain_upward=True,
        quadrants_used=quadrants_used,
        notes=notes,
    )
