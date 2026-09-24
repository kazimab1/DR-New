"""D12: M3's image-level operating point (src/verify_dr/reasoning/operating_point.py).

The amended rule must be M3 with one condition added, nothing else: the ladder has to
agree with `rules.grade` on every presence pattern, and the frozen setting has to give
back exactly the evidence grade the real pipeline (extract_facts -> grade) produced.
"""

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verify_dr.data.segmentation import LESION_NAMES  # noqa: E402
from verify_dr.reasoning import extract_facts, grade  # noqa: E402
from verify_dr.reasoning.facts import Facts  # noqa: E402
from verify_dr.reasoning.operating_point import (  # noqa: E402
    AREA_GRID, FROZEN_AREA_MIN, evidence, fit_area_minimums, ladder,
)

GEOMETRY = "C1 failed its gate: count-only"


def row(counts, areas):
    out = {}
    for name in LESION_NAMES:
        out[f"n_{name}"] = counts.get(name, 0)
        out[f"area_{name}"] = areas.get(name, 0)
    return out


class Ladder(unittest.TestCase):

    def test_agrees_with_rules_grade_on_all_16_presence_patterns(self):
        for pattern in itertools.product((0, 1), repeat=4):
            counts = dict(zip(LESION_NAMES, pattern))
            verdict = grade(Facts(counts=counts, areas={n: 10 * c for n, c in counts.items()}))
            e, rule = ladder(np.array([pattern[0]]), np.array([pattern[1]]),
                             np.array([pattern[2] or pattern[3]]))
            self.assertEqual(e[0], verdict.evidence_grade, f"{pattern}")
            self.assertEqual(rule[0], verdict.rules_fired[0].split(":")[0], f"{pattern}")

    def test_the_frozen_setting_reproduces_the_real_pipeline(self):
        rng = np.random.default_rng(0)
        rows, grades = [], []
        for _ in range(60):
            probs = np.zeros((4, 64, 64))
            for c in range(4):
                for _ in range(rng.integers(0, 3)):
                    y0, x0 = rng.integers(0, 58, 2)
                    h, w = rng.integers(1, 7, 2)             # some blobs below 4 px
                    probs[c, y0:y0 + h, x0:x0 + w] = rng.uniform(0.3, 1.0)
            facts = extract_facts(probs, geometry=None, geometry_reason=GEOMETRY)
            grades.append(grade(facts).evidence_grade)
            rows.append(row(facts.counts, facts.areas))
        e, _ = evidence(pd.DataFrame(rows), FROZEN_AREA_MIN)
        self.assertEqual(e.tolist(), grades)
        self.assertGreater(len(set(grades)), 1, "the fixture should exercise several rules")


class Minimums(unittest.TestCase):

    def test_a_type_below_its_minimum_is_absent(self):
        images = pd.DataFrame([row({"haemorrhage": 1}, {"haemorrhage": 20})])
        self.assertEqual(evidence(images, FROZEN_AREA_MIN)[0][0], 2)          # R3*
        raised = {**FROZEN_AREA_MIN, "haemorrhage": 64}
        self.assertEqual(evidence(images, raised)[0][0], 0)                   # R1

    def test_the_fit_removes_small_false_positives_and_keeps_real_lesions(self):
        """Healthy eyes carry small spurious exudate; diseased ones carry large lesions."""
        rng = np.random.default_rng(1)
        rows, y = [], []
        for i in range(400):
            sick = i % 4 == 0
            if sick:
                counts = {"microaneurysm": 2, "haemorrhage": 3}
                areas = {"microaneurysm": 40, "haemorrhage": int(rng.integers(300, 900))}
                y.append(2)
            else:
                counts = {"hard_exudate": 1}
                areas = {"hard_exudate": int(rng.integers(4, 40))}
                y.append(0)
            rows.append(row(counts, areas))
        images, y = pd.DataFrame(rows), np.array(y)
        fit = fit_area_minimums(images, y)
        self.assertLess(fit["qwk_frozen"], 0.5)
        self.assertGreater(fit["qwk"], 0.99)
        self.assertGreaterEqual(fit["area_min"]["hard_exudate"], 64)
        self.assertEqual(fit["combinations"], len(AREA_GRID) ** 4)
        # soft exudate never appears: every minimum ties, and the tie goes to 4 px
        self.assertEqual(fit["area_min"]["soft_exudate"], 4)

    def test_nothing_to_gain_means_the_frozen_rule(self):
        rows = [row({"microaneurysm": 1}, {"microaneurysm": 30}), row({}, {})] * 50
        y = np.array([1, 0] * 50)
        fit = fit_area_minimums(pd.DataFrame(rows), y)
        self.assertEqual(fit["area_min"], FROZEN_AREA_MIN)
        self.assertAlmostEqual(fit["qwk"], fit["qwk_frozen"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
