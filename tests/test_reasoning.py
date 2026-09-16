"""M3 — the ICDR reasoner.

Deterministic and unlearned, so every rule is directly testable. The properties
pinned here are the ones the thesis quotes: that unobservable findings are
declared rather than assumed absent, that R4 declines without a trusted
coordinate frame, and that quadrant assignment is correct in either eye.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verify_dr.reasoning import UNOBSERVABLE, extract_facts, grade, quadrant_of


def stack(size=256):
    # 256, not 128: the disc/fovea and lesion coordinates below are chosen in a
    # 256 px frame, and a blob placed outside the array silently lands nowhere --
    # which reads as "the rule did not fire" rather than "the test is wrong".
    return np.zeros((4, size, size), np.float32)


def blob(probs, channel, x, y, r=3):
    yy, xx = np.ogrid[:probs.shape[1], :probs.shape[2]]
    probs[channel][(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = 1.0


class TestQuadrants(unittest.TestCase):
    """Superior/inferior is anatomically fixed; temporal/nasal follows the fovea.
    An earlier version rotated the perpendicular with the axis, which labelled a
    superior lesion 'infero' in one eye and 'supero' in the other -- the same
    laterality trap C1 fell into, one layer up."""

    RIGHT = ((60.0, 128.0), (190.0, 128.0))     # fovea temporal, to the right
    LEFT = ((190.0, 128.0), (60.0, 128.0))      # mirrored

    def test_right_eye(self):
        disc, fovea = self.RIGHT
        self.assertEqual(quadrant_of(220, 80, disc, fovea), "superotemporal")
        self.assertEqual(quadrant_of(220, 180, disc, fovea), "inferotemporal")
        self.assertEqual(quadrant_of(20, 80, disc, fovea), "superonasal")
        self.assertEqual(quadrant_of(20, 180, disc, fovea), "inferonasal")

    def test_left_eye_mirrors_temporal_but_not_superior(self):
        disc, fovea = self.LEFT
        self.assertEqual(quadrant_of(20, 80, disc, fovea), "superotemporal")
        self.assertEqual(quadrant_of(230, 80, disc, fovea), "superonasal")

    def test_a_tilted_axis_still_resolves(self):
        import math
        disc = (60.0, 128.0)
        fovea = (60 + 130 * math.cos(math.radians(20)),
                 128 + 130 * math.sin(math.radians(20)))
        self.assertEqual(quadrant_of(200, 60, disc, fovea), "superotemporal")
        self.assertEqual(quadrant_of(200, 230, disc, fovea), "inferotemporal")

    def test_coincident_landmarks_are_rejected(self):
        with self.assertRaises(ValueError):
            quadrant_of(10, 10, (50.0, 50.0), (50.0, 50.0))


class TestRules(unittest.TestCase):
    DISC, FOVEA = (60.0, 128.0), (190.0, 128.0)

    def facts(self, probs, trusted=False):
        return extract_facts(probs, geometry=(*self.DISC, *self.FOVEA),
                             geometry_trusted=trusted)

    def test_R1_no_lesions(self):
        v = grade(self.facts(stack()))
        self.assertEqual(v.evidence_grade, 0)
        self.assertTrue(v.rules_fired[0].startswith("R1"))

    def test_R2_microaneurysms_alone(self):
        p = stack()
        for i in range(3):
            blob(p, 0, 100 + 10 * i, 90)
        v = grade(self.facts(p))
        self.assertEqual(v.evidence_grade, 1)
        self.assertTrue(v.rules_fired[0].startswith("R2"))

    def test_R3_microaneurysms_with_haemorrhage(self):
        p = stack()
        blob(p, 0, 100, 90)
        blob(p, 1, 140, 150, 6)
        v = grade(self.facts(p))
        self.assertEqual(v.evidence_grade, 2)

    def test_a_lesion_without_MA_is_not_graded_zero(self):
        # ICDR presumes MA accompanies anything further along, but M2a misses MA
        # more often than larger lesions. Grading such an image 0 would be a
        # segmentation artefact reported as health.
        p = stack()
        blob(p, 1, 140, 150, 6)
        v = grade(self.facts(p))
        self.assertEqual(v.evidence_grade, 2)
        self.assertIn("R3*", v.rules_fired[0])

    def _four_二_one(self, p):
        for qx, qy in ((210, 60), (210, 200), (20, 60), (20, 200)):
            for k in range(22):
                blob(p, 1, qx + (k % 6) * 5 - 12, qy + (k // 6) * 5 - 6, 1)
        return p

    def test_R4_needs_a_trusted_frame(self):
        p = self._four_二_one(stack())
        blob(p, 0, 120, 120)
        without = grade(self.facts(p, trusted=False))
        self.assertEqual(without.evidence_grade, 2, "R4 must not fire untrusted")
        self.assertFalse(without.quadrants_used)
        self.assertTrue(any("R4 declined" in n for n in without.notes))

        with_frame = grade(self.facts(p, trusted=True))
        self.assertEqual(with_frame.evidence_grade, 3)
        self.assertTrue(with_frame.quadrants_used)

    def test_severe_is_unreachable_without_geometry(self):
        """The concrete cost of C1 failing: every grade-3 image reads as grade 2."""
        p = self._four_二_one(stack())
        blob(p, 0, 120, 120)
        self.assertLess(grade(self.facts(p, trusted=False)).evidence_grade, 3)


class TestHonestyProperties(unittest.TestCase):
    """The part of M3 the thesis actually rests on."""

    def test_unobservable_findings_are_always_declared(self):
        for probs in (stack(), stack()):
            v = grade(extract_facts(probs))
            self.assertEqual(set(v.unobservable), set(UNOBSERVABLE))
        self.assertIn("neovascularisation", UNOBSERVABLE)
        self.assertIn("IRMA", UNOBSERVABLE)
        self.assertIn("venous_beading", UNOBSERVABLE)

    def test_grade_4_is_never_claimed_nor_excluded(self):
        # R5 needs neovascularisation, which nothing annotates. The reasoner must
        # not emit grade 4, and must not claim to have ruled it out either.
        p = stack()
        blob(p, 0, 100, 90)
        blob(p, 1, 140, 150, 6)
        v = grade(extract_facts(p))
        self.assertLess(v.evidence_grade, 4)
        self.assertLess(v.max_excludable_grade, 4)
        self.assertTrue(v.abstain_upward)

    def test_a_clean_image_still_cannot_exclude_severe(self):
        """No lesions found is not the same as no disease. With beading, IRMA and
        neovascularisation unobservable, nothing above moderate can be ruled out
        however clean the image looks."""
        v = grade(extract_facts(stack()))
        self.assertEqual(v.evidence_grade, 0)
        self.assertEqual(v.max_excludable_grade, 2)

    def test_every_verdict_cites_a_rule(self):
        p = stack()
        blob(p, 2, 100, 100, 5)
        v = grade(extract_facts(p))
        self.assertTrue(v.rules_fired)
        self.assertRegex(v.rules_fired[0], r"^R\d")


if __name__ == "__main__":
    unittest.main()
