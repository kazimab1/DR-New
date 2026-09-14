"""Properties the Phase 4 evidence pathway must not lose.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verify_dr.data.geometry import GeometryDataset, patient_split  # noqa: E402
from verify_dr.evaluation.geometry_metrics import (  # noqa: E402
    C1_GATE_DD, DISC_TO_FOVEA_IN_DIAMETERS, disc_diameters, geometry_metrics,
)
from verify_dr.models.evidence import (  # noqa: E402
    EvidenceEncoder, EvidenceModel, GeometryModel, LesionSegmenter,
)

REPO = Path(__file__).resolve().parent.parent


class TestIndependenceFromM1(unittest.TestCase):
    def test_evidence_never_imports_the_grading_pathway(self):
        """CLAUDE.md rule 4. M1 and M2 sharing weights would make them fail
        together, and disagreement between them is the entire thesis. A shared
        import is how that would creep in, so it is checked statically rather than
        left to reviewer vigilance."""
        source = (REPO / "src/verify_dr/models/evidence.py").read_text()
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
        offenders = [m for m in imported if "grading" in m or "losses" in m]
        self.assertEqual(offenders, [], f"evidence.py imports from M1: {offenders}")


class TestEncoder(unittest.TestCase):
    def test_skips_match_the_specified_ladder(self):
        """docs/03 asks for skips at 256, 128, 64 and 32 from a 512 px input. The
        UNet decoder's channel arithmetic is built on those exact widths."""
        encoder = EvidenceEncoder(pretrained=False)
        bottleneck, skips = encoder(torch.randn(1, 3, 512, 512))
        self.assertEqual(tuple(bottleneck.shape), (1, EvidenceEncoder.BOTTLENECK, 16, 16))
        for skip, (channels, size) in zip(skips, EvidenceEncoder.SKIPS):
            self.assertEqual((skip.shape[1], skip.shape[2], skip.shape[3]),
                             (channels, size, size))


class TestGeometryModel(unittest.TestCase):
    def test_output_is_four_coordinates_inside_the_unit_square(self):
        out = GeometryModel(pretrained=False)(torch.randn(2, 3, 256, 256))
        self.assertEqual(tuple(out.shape), (2, 4))
        self.assertGreaterEqual(float(out.min().detach()), 0.0)
        self.assertLessEqual(float(out.max().detach()), 1.0)

    def test_extreme_features_still_cannot_leave_the_frame(self):
        """The sigmoid is the point: a centre outside [0,1] is not representable,
        so the head never spends capacity learning that constraint."""
        model = GeometryModel(pretrained=False)
        out = model(torch.randn(4, 3, 256, 256) * 50)
        self.assertTrue(bool(((out >= 0) & (out <= 1)).all()))


class TestSegmenter(unittest.TestCase):
    def test_four_channels_at_full_resolution(self):
        logits = LesionSegmenter(pretrained=False)(torch.randn(1, 3, 512, 512))
        self.assertEqual(tuple(logits.shape), (1, 4, 512, 512))

    def test_combined_model_emits_both_heads(self):
        out = EvidenceModel(pretrained=False)(torch.randn(1, 3, 512, 512))
        self.assertEqual(tuple(out["lesion_logits"].shape), (1, 4, 512, 512))
        self.assertEqual(tuple(out["geometry"].shape), (1, 4))

    def test_both_heads_read_one_encoder(self):
        model = EvidenceModel(pretrained=False)
        self.assertIs(model.encoder, model.segmenter.encoder)


class TestHorizontalFlip(unittest.TestCase):
    """The flip is the only augmentation that moves the targets, so it is the only
    one that can corrupt them. A sign error here trains the model to predict a
    mirrored landmark and looks exactly like a model that failed to learn."""

    def _fixture(self, tmp: Path, size: int = 64, mark_x: int = 10, mark_y: int = 20):
        img = Image.new("RGB", (size, size), (0, 0, 0))
        img.paste((255, 255, 255), (mark_x - 2, mark_y - 2, mark_x + 2, mark_y + 2))
        path = tmp / "IDRiD_000.jpg"
        img.save(path, quality=100)
        frame = pd.DataFrame([{
            "image_path": str(path), "patient_id": "p0",
            "od_x": mark_x, "od_y": mark_y, "od_in_frame": 1,
            "fovea_x": mark_x + 20, "fovea_y": mark_y, "fovea_in_frame": 1,
        }])
        return frame, size

    def test_flip_moves_the_target_onto_the_flipped_landmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, size = self._fixture(Path(tmp))
            ds = GeometryDataset(frame, image_size=size, cache_size=size,
                                 train=True, hflip=1.0, jitter=0.0)
            item = ds[0]

            # Where the bright mark actually ended up in the returned image.
            brightest_col = int(item["image"].mean(0).mean(0).argmax())
            predicted_col = float(item["target"][0]) * size
            self.assertAlmostEqual(brightest_col, predicted_col, delta=2.0)

    def test_flip_reflects_x_and_leaves_y_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, size = self._fixture(Path(tmp))
            flipped = GeometryDataset(frame, image_size=size, cache_size=size,
                                      train=True, hflip=1.0, jitter=0.0)[0]["target"]
            plain = GeometryDataset(frame, image_size=size, cache_size=size,
                                    train=False)[0]["target"]
            self.assertAlmostEqual(float(flipped[0]), 1.0 - float(plain[0]), places=5)
            self.assertAlmostEqual(float(flipped[2]), 1.0 - float(plain[2]), places=5)
            self.assertAlmostEqual(float(flipped[1]), float(plain[1]), places=5)
            self.assertAlmostEqual(float(flipped[3]), float(plain[3]), places=5)

    def test_validation_never_flips(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, size = self._fixture(Path(tmp))
            ds = GeometryDataset(frame, image_size=size, cache_size=size, train=False)
            self.assertEqual(ds.hflip, 0.0)
            self.assertEqual(ds.jitter, 0.0)


class TestGeometryMetrics(unittest.TestCase):
    def test_disc_diameter_comes_from_the_landmark_separation(self):
        truth = np.array([[0.2, 0.5, 0.2 + 0.25, 0.5]])      # 0.25 apart
        self.assertAlmostEqual(float(disc_diameters(truth)[0]),
                               0.25 / DISC_TO_FOVEA_IN_DIAMETERS, places=6)

    def test_perfect_prediction_scores_zero(self):
        truth = np.array([[0.2, 0.5, 0.45, 0.5], [0.8, 0.4, 0.55, 0.45]])
        m = geometry_metrics(truth, truth)
        self.assertAlmostEqual(m["mean_error_dd"], 0.0)
        self.assertAlmostEqual(m["od_error_px"], 0.0)
        self.assertTrue(m["c1_pass"])

    def test_error_is_measured_in_disc_diameters_not_pixels(self):
        """A fixed pixel tolerance would mean different things at different fields
        of view, which is why C1 is stated in disc diameters. Same absolute error,
        different disc size, different verdict."""
        tight = np.array([[0.2, 0.5, 0.45, 0.5]])       # separation 0.25 -> DD 0.10
        wide = np.array([[0.2, 0.5, 0.70, 0.5]])        # separation 0.50 -> DD 0.20
        shift = 0.05                                     # the same 0.05 in both cases

        for truth, expected_dd in ((tight, 0.5), (wide, 0.25)):
            pred = truth.copy()
            pred[0, 0] += shift
            self.assertAlmostEqual(
                geometry_metrics(pred, truth)["od_error_dd"], expected_dd, places=3)

    def test_the_gate_is_the_mean_over_both_landmarks(self):
        truth = np.array([[0.2, 0.5, 0.45, 0.5]])       # DD = 0.1

        one_landmark_off = truth.copy()
        one_landmark_off[0, 0] += 0.05                   # OD 0.5 DD out, fovea exact
        m = geometry_metrics(one_landmark_off, truth)
        self.assertAlmostEqual(m["od_error_dd"], C1_GATE_DD, places=3)
        self.assertAlmostEqual(m["fovea_error_dd"], 0.0, places=6)
        self.assertAlmostEqual(m["mean_error_dd"], 0.25, places=3)
        self.assertTrue(m["c1_pass"], "one perfect landmark should carry the mean under")

        both_off = truth.copy()
        both_off[0, [0, 2]] += 0.06                      # 0.6 DD each
        failed = geometry_metrics(both_off, truth)
        self.assertAlmostEqual(failed["mean_error_dd"], 0.6, places=3)
        self.assertFalse(failed["c1_pass"])

    def test_degenerate_images_are_excluded_not_averaged_in(self):
        """Both centres marked at the same point gives a zero disc diameter. Left
        in, it divides by zero and poisons the mean for every other image."""
        truth = np.array([[0.3, 0.3, 0.3, 0.3], [0.2, 0.5, 0.45, 0.5]])
        pred = truth + 0.01
        m = geometry_metrics(pred, truth)
        self.assertEqual(m["degenerate"], 1)
        self.assertTrue(np.isfinite(m["mean_error_dd"]))


class TestPatientSplit(unittest.TestCase):
    def test_no_patient_appears_in_both_splits(self):
        frame = pd.DataFrame({"patient_id": [f"p{i // 2}" for i in range(40)],
                              "image_path": [f"/x/{i}.jpg" for i in range(40)]})
        splits = patient_split(frame, val_frac=0.25, seed=1)
        self.assertFalse(set(splits["train"]["patient_id"]) &
                         set(splits["val"]["patient_id"]))
        self.assertEqual(len(splits["train"]) + len(splits["val"]), len(frame))

    def test_a_split_is_always_non_empty(self):
        frame = pd.DataFrame({"patient_id": ["a", "b"], "image_path": ["/x/1", "/x/2"]})
        splits = patient_split(frame, val_frac=0.01, seed=0)
        self.assertGreaterEqual(len(splits["val"]), 1)


if __name__ == "__main__":
    unittest.main()
