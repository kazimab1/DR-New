"""M4b faithfulness (ANALYSIS_PLAN.md section 5.4) and the region it inpaints."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verify_dr.reasoning import extract_facts, lesion_region_mask  # noqa: E402
from verify_dr.triage import faithfulness as F  # noqa: E402


def disc(size, cy, cx, r):
    yy, xx = np.mgrid[:size, :size]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def fundus(size=128):
    """A lit disc on black padding, like the 512 px cache."""
    rgb = np.zeros((size, size, 3), np.uint8)
    inside = disc(size, size // 2, size // 2, size // 2 - 4)
    rng = np.random.default_rng(0)
    rgb[inside] = rng.integers(60, 200, size=(inside.sum(), 3))
    return rgb


class RegionMask(unittest.TestCase):
    """The inpainted region must be exactly the lesions M3 counted."""

    def test_specks_below_the_minimum_are_excluded_like_the_counts(self):
        probs = np.zeros((4, 64, 64), np.float32)
        probs[0, 5, 5] = probs[0, 5, 6] = 1.0                 # 2 px speck
        probs[0][disc(64, 30, 30, 3)] = 1.0                   # a real MA
        facts = extract_facts(probs, min_lesion_px=4)
        region = lesion_region_mask(probs, min_lesion_px=4)
        self.assertEqual(facts.counts["microaneurysm"], 1)
        self.assertFalse(region[5, 5] or region[5, 6], "speck inpainted but not counted")
        self.assertEqual(int(region.sum()), int(disc(64, 30, 30, 3).sum()))

    def test_channels_are_unioned_so_overlap_is_not_double_counted(self):
        probs = np.zeros((4, 64, 64), np.float32)
        a, b = disc(64, 30, 30, 6), disc(64, 30, 36, 6)
        probs[0][a] = 1.0
        probs[1][b] = 1.0
        self.assertEqual(int(lesion_region_mask(probs).sum()), int((a | b).sum()))

    def test_component_count_matches_M3_on_random_maps(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            probs = np.zeros((4, 48, 48), np.float32)
            probs[2] = (rng.random((48, 48)) > 0.93).astype(np.float32)
            import cv2
            n, _ = cv2.connectedComponents(
                lesion_region_mask(probs).astype(np.uint8), connectivity=8)
            self.assertEqual(n - 1, extract_facts(probs).counts["hard_exudate"], seed)

    def test_threshold_is_inclusive_like_extract_facts(self):
        probs = np.zeros((4, 32, 32), np.float32)
        probs[3][disc(32, 16, 16, 3)] = 0.5
        self.assertEqual(extract_facts(probs).counts["soft_exudate"], 1)
        self.assertTrue(lesion_region_mask(probs).any())


class Controls(unittest.TestCase):

    def setUp(self):
        self.rgb = fundus(128)
        self.fov = F.field_of_view(self.rgb)
        lesions = disc(128, 50, 50, 3) | disc(128, 80, 70, 5)
        self.removed = F.dilate(lesions)
        self.shapes = F.regions(self.removed)

    def test_dilation_grows_by_three_pixels_as_a_disc(self):
        grown = F.dilate(disc(64, 32, 32, 0))
        self.assertTrue(grown[32, 35] and grown[35, 32])
        self.assertFalse(grown[35, 35], "a square kernel would reach the corner")

    def test_regions_cover_the_mask_largest_first(self):
        self.assertEqual(len(self.shapes), 2)
        self.assertEqual(sum(s.area for s in self.shapes), int(self.removed.sum()))
        self.assertGreaterEqual(self.shapes[0].area, self.shapes[1].area)

    def test_a_control_is_equal_area_inside_the_retina_and_off_the_lesions(self):
        for draw in range(F.CONTROLS):
            c = F.place_control(self.shapes, self.removed, self.fov, "EyePACS::10_left", draw)
            self.assertIsNotNone(c)
            self.assertEqual(int(c.sum()), int(self.removed.sum()), "not equal area")
            self.assertTrue(self.fov[c].all(), "control outside the field of view")
            self.assertFalse((c & self.removed).any(), "control overlaps the lesions")

    def test_draws_are_seeded_by_image_id(self):
        a = F.place_control(self.shapes, self.removed, self.fov, "x", 3)
        b = F.place_control(self.shapes, self.removed, self.fov, "x", 3)
        c = F.place_control(self.shapes, self.removed, self.fov, "x", 4)
        d = F.place_control(self.shapes, self.removed, self.fov, "y", 3)
        self.assertTrue((a == b).all())
        self.assertFalse((a == c).all())
        self.assertFalse((a == d).all())

    def test_an_impossible_draw_fails_rather_than_shrinking(self):
        tiny = np.zeros_like(self.fov)
        tiny[60:64, 60:64] = True
        self.assertIsNone(F.place_control(self.shapes, self.removed, tiny, "x", 0))

    def test_inpainting_changes_only_the_masked_pixels(self):
        v = F.make_variants(self.rgb, disc(128, 50, 50, 3) | disc(128, 80, 70, 5), "img")
        self.assertEqual(v.region_px, int(self.removed.sum()))
        self.assertTrue((v.lesion[~self.removed] == self.rgb[~self.removed]).all())
        self.assertEqual(len(v.controls), F.CONTROLS)
        self.assertEqual(v.controls_ok, F.CONTROLS)

    def test_no_lesions_means_nothing_to_test(self):
        self.assertIsNone(F.make_variants(self.rgb, np.zeros((128, 128), bool), "img"))


class Decision(unittest.TestCase):

    def test_expected_grade_spans_zero_to_four(self):
        self.assertAlmostEqual(float(F.expected_grade(np.full(4, 30.0))), 4.0, places=6)
        self.assertAlmostEqual(float(F.expected_grade(np.full(4, -30.0))), 0.0, places=6)

    def test_faithful_needs_every_control_beaten(self):
        self.assertTrue(F.faithful(0.5, [0.1] * 19))
        self.assertFalse(F.faithful(0.5, [0.1] * 18 + [0.6]))

    def test_too_few_controls_is_undetermined_not_unfaithful(self):
        self.assertIsNone(F.faithful(0.5, [0.1] * 14 + [np.nan] * 5))
        self.assertTrue(F.faithful(0.5, [0.1] * 15 + [np.nan] * 4))

    def test_a_lesion_blind_network_passes_about_one_time_in_twenty(self):
        """The plan's claim: p <= 1/(K + 1) when lesion and control deltas are
        exchangeable. Checked, not assumed."""
        rng = np.random.default_rng(7)
        trials = 20000
        deltas = rng.normal(size=(trials, F.CONTROLS + 1))
        rate = np.mean([F.faithful(d[0], d[1:]) for d in deltas])
        self.assertAlmostEqual(rate, 1 / (F.CONTROLS + 1), delta=0.006)


if __name__ == "__main__":
    unittest.main(verbosity=2)
