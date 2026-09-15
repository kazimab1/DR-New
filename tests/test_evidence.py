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
from verify_dr.data.segmentation import (  # noqa: E402
    MASK_DIRS, SegmentationDataset, channel_presence, mask_path,
)
from verify_dr.evaluation.segmentation_metrics import segmentation_metrics  # noqa: E402
from verify_dr.models.seg_losses import SegmentationLoss, dice_loss  # noqa: E402
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


class TestMaskPaths(unittest.TestCase):
    def test_mask_sits_beside_images_under_the_dataset_directory(self):
        img = Path("/cache/ddr/images/ddr_0001.jpg")
        self.assertEqual(mask_path(img, MASK_DIRS[0]),
                         Path(f"/cache/ddr/masks/{MASK_DIRS[0]}/ddr_0001.png"))

    def test_nesting_below_images_does_not_move_the_mask(self):
        """EyePACS nests under images/; build_cache.py still writes masks flat
        under masks/<CH>/ keyed by stem, so the anchor is the images directory."""
        img = Path("/cache/eyepacs/images/ORIG/train/2/16_left.jpg")
        self.assertEqual(mask_path(img, "HE"),
                         Path("/cache/eyepacs/masks/HE/16_left.png"))


class TestSegmentationAugmentation(unittest.TestCase):
    """Image and mask must be flipped together. Flipping one and not the other
    trains against mirrored targets and is indistinguishable from a model that
    never converged -- the same failure class as the geometry flip."""

    def _fixture(self, tmp: Path, size: int = 32):
        root = tmp / "ddr"
        (root / "images").mkdir(parents=True)
        for channel in MASK_DIRS:
            (root / "masks" / channel).mkdir(parents=True)

        img = Image.new("RGB", (size, size), (0, 0, 0))
        img.paste((255, 0, 0), (2, 10, 8, 16))          # a mark on the LEFT
        img.save(root / "images" / "ddr_0000.jpg", quality=100)

        mask = Image.new("L", (size, size), 0)
        mask.paste(255, (2, 10, 8, 16))                 # the same pixels
        # MASK_DIRS[0], not a literal: the fixture restating the channel name is
        # what let this suite pass while mask_path looked in a directory the cache
        # does not have.
        mask.save(root / "masks" / MASK_DIRS[0] / "ddr_0000.png")

        frame = pd.DataFrame([{"image_path": str(root / "images" / "ddr_0000.jpg"),
                               "dataset": "ddr", "patient_id": "p0"}])
        return frame, size

    @staticmethod
    def _column_centroid(profile: torch.Tensor) -> float:
        """Intensity-weighted mean column.

        Compared rather than argmax because the image is a JPEG: compression
        softens the block's edges, so its row-mean peaks mid-block while the
        binary mask's peaks at the first column. Both describe the same region,
        and the centroid says so; argmax reports a spurious few-pixel gap.
        """
        weights = (profile - profile.min()).clamp_min(0)
        columns = torch.arange(len(weights), dtype=weights.dtype)
        return float((weights * columns).sum() / weights.sum().clamp_min(1e-9))

    def test_flip_moves_image_and_mask_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, size = self._fixture(Path(tmp))
            item = SegmentationDataset(frame, image_size=size, train=True,
                                       hflip=1.0, jitter=0.0)[0]
            red = self._column_centroid(item["image"][0].mean(0))
            mask = self._column_centroid(item["mask"][0].mean(0))
            self.assertAlmostEqual(red, mask, delta=2.0,
                                   msg="image and mask flipped independently")
            self.assertGreater(mask, size / 2, "mark should have moved right")

    def test_not_flipping_leaves_both_on_the_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, size = self._fixture(Path(tmp))
            item = SegmentationDataset(frame, image_size=size, train=True,
                                       hflip=0.0, jitter=0.0)[0]
            red = self._column_centroid(item["image"][0].mean(0))
            mask = self._column_centroid(item["mask"][0].mean(0))
            self.assertAlmostEqual(red, mask, delta=2.0)
            self.assertLess(mask, size / 2)

    def test_unflipped_mark_stays_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, size = self._fixture(Path(tmp))
            item = SegmentationDataset(frame, image_size=size, train=False)[0]
            self.assertLess(int(item["mask"][0].mean(0).argmax()), size // 2)

    def test_absent_channel_becomes_an_all_zero_target(self):
        """DDR ships a mask only where the lesion occurs, so a missing file means
        absent -- not unlabelled, and not a reason to skip the image."""
        with tempfile.TemporaryDirectory() as tmp:
            frame, size = self._fixture(Path(tmp))
            masks = SegmentationDataset(frame, image_size=size, train=False)[0]["mask"]
            self.assertEqual(tuple(masks.shape), (4, size, size))
            self.assertGreater(float(masks[0].sum()), 0)      # MA present
            for c in range(1, 4):
                self.assertEqual(float(masks[c].sum()), 0.0)  # HE, EX, SE absent

    def test_masks_stay_binary_after_resize(self):
        """Bilinear resizing of a binary mask invents partial-membership pixels on
        every boundary, and a microaneurysm is only a few pixels across."""
        with tempfile.TemporaryDirectory() as tmp:
            frame, size = self._fixture(Path(tmp))
            masks = SegmentationDataset(frame, image_size=size * 2, train=False)[0]["mask"]
            values = set(masks.unique().tolist())
            self.assertTrue(values <= {0.0, 1.0}, f"non-binary values: {values}")

    def test_channel_presence_counts_images_not_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, _ = self._fixture(Path(tmp))
            counts = channel_presence(frame)
            self.assertEqual(counts["microaneurysm"], 1)
            self.assertEqual(counts["soft_exudate"], 0)


class TestSegmentationLoss(unittest.TestCase):
    def test_empty_target_costs_nothing_when_prediction_is_empty(self):
        """Summing Dice over batch and space makes an empty target's denominator
        the accumulated predicted probability, so at 512 px a correct empty answer
        scored ~0.99. Empty channels are the norm here, so that dominated."""
        empty = torch.zeros(2, 4, 256, 256)
        confident_empty = torch.full((2, 4, 256, 256), -10.0)
        self.assertLess(float(dice_loss(confident_empty, empty)), 1e-4)

    def test_missing_a_present_lesion_is_fully_penalised(self):
        target = torch.zeros(2, 4, 64, 64)
        target[:, 1, 20:40, 20:40] = 1.0
        predicts_nothing = torch.full((2, 4, 64, 64), -10.0)
        self.assertGreater(float(dice_loss(predicts_nothing, target)), 0.9)

    def test_perfect_prediction_is_near_zero(self):
        target = torch.zeros(2, 4, 64, 64)
        target[:, 1, 20:40, 20:40] = 1.0
        perfect = torch.where(target > 0, 10.0, -10.0)
        self.assertLess(float(dice_loss(perfect, target)), 0.05)

    def test_bce_keeps_a_gradient_on_an_all_empty_batch(self):
        """Dice contributes nothing when there is no overlap to score, so BCE has
        to carry it -- that is the division of labour the 0.5/0.5 split is for."""
        logits = torch.randn(2, 4, 32, 32, requires_grad=True)
        SegmentationLoss()(logits, torch.zeros(2, 4, 32, 32))["loss"].backward()
        self.assertGreater(float(logits.grad.norm()), 0.0)


class TestSegmentationMetrics(unittest.TestCase):
    def test_dice_present_excludes_images_without_the_lesion(self):
        """Averaging over every image folds in empty/empty pairs that score 1.0 by
        convention, inflating the headline without segmenting anything."""
        truth = np.zeros((10, 4, 8, 8), dtype=np.float32)
        truth[0, 3, 2:5, 2:5] = 1.0                  # SE in one image of ten
        probs = np.zeros((10, 4, 8, 8), dtype=np.float32)   # predicts nothing ever

        m = segmentation_metrics(probs, truth)["per_lesion"]["soft_exudate"]
        self.assertEqual(m["images_with_lesion"], 1)
        self.assertAlmostEqual(m["dice_present"], 0.0)       # missed the only one
        self.assertAlmostEqual(m["dice_all"], 0.9, places=6)  # nine free 1.0s

    def test_perfect_segmentation_scores_one(self):
        truth = np.zeros((4, 4, 8, 8), dtype=np.float32)
        truth[:, 0, 1:4, 1:4] = 1.0
        m = segmentation_metrics(truth.copy(), truth)["per_lesion"]["microaneurysm"]
        self.assertAlmostEqual(m["dice_present"], 1.0)
        self.assertAlmostEqual(m["iou_present"], 1.0)

    def test_over_segmentation_is_counted(self):
        """A segmenter that finds lesions everywhere makes disagreement
        meaningless, so false-positive images are reported, not just Dice."""
        truth = np.zeros((5, 4, 8, 8), dtype=np.float32)
        probs = np.zeros((5, 4, 8, 8), dtype=np.float32)
        probs[:, 2, 0:2, 0:2] = 1.0                  # hallucinates EX everywhere
        m = segmentation_metrics(probs, truth)["per_lesion"]["hard_exudate"]
        self.assertEqual(m["false_positive_images"], 5)
        self.assertNotEqual(m["dice_all"], 1.0)


class TestChannelLabels(unittest.TestCase):
    """The per-epoch log abbreviates each lesion. Two channels printing under the
    same label would let a collapsed one hide behind a healthy one, which is exactly
    the failure the log exists to surface."""

    def test_abbreviations_are_distinct(self):
        from verify_dr.data.segmentation import LESION_NAMES, MASK_DIRS
        self.assertEqual(len(set(MASK_DIRS)), len(MASK_DIRS))
        self.assertEqual(len(MASK_DIRS), len(LESION_NAMES))

    def test_first_two_letters_would_have_collided(self):
        # Pins why MASK_DIRS is used rather than name[:2]: haemorrhage and
        # hard_exudate both truncate to "HA".
        from verify_dr.data.segmentation import LESION_NAMES
        truncated = [n[:2].upper() for n in LESION_NAMES]
        self.assertLess(len(set(truncated)), len(truncated))


class TestIDRiDTables(unittest.TestCase):
    """Reading IDRiD's markup tables. Every property pinned here has broken a run:
    the filename carries no reliable keyword, the header row moves, .xlsx needs an
    engine that is not always installed, and a grading table looks similar."""

    def setUp(self):
        import tempfile
        import numpy as np
        import pandas as pd
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp)
        rng = np.random.default_rng(0)
        ids = [f"IDRiD_{i:03d}" for i in range(1, 31)]
        coords = pd.DataFrame({"Image No": ids,
                               "X- Coordinate": rng.integers(200, 600, 30),
                               "Y- Coordinate": rng.integers(150, 450, 30)})
        self.coords = coords
        self.ids = set(ids)
        (self.root / "gt").mkdir()
        coords.to_csv(self.root / "gt" / "IDRiD_OD_Center_Markups.csv", index=False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_finds_a_plain_table(self):
        from verify_dr.data.idrid_tables import find_coord_tables
        found, _n, _u = find_coord_tables(self.root)
        self.assertEqual(len(found), 1)
        ids, col, (x_col, y_col) = next(iter(found.values()))
        self.assertEqual((x_col, y_col), ("X- Coordinate", "Y- Coordinate"))
        self.assertEqual(ids, self.ids)
        self.assertEqual(col, "Image No")

    def test_name_carries_no_keyword(self):
        # A mirror that calls it Localization_Groundtruth still resolves, because
        # detection reads the values rather than the filename.
        (self.root / "gt" / "IDRiD_OD_Center_Markups.csv").unlink()
        self.coords.to_csv(self.root / "gt" / "somefile.csv", index=False)
        from verify_dr.data.idrid_tables import find_coord_tables
        found, _n, _u = find_coord_tables(self.root)
        self.assertEqual(len(found), 1)

    def test_title_line_above_the_header(self):
        (self.root / "gt" / "IDRiD_OD_Center_Markups.csv").unlink()
        body = "Localization ground truth, IDRiD 2018\n" + self.coords.to_csv(index=False)
        (self.root / "gt" / "t.csv").write_text(body)
        from verify_dr.data.idrid_tables import find_coord_tables
        found, _n, _u = find_coord_tables(self.root)
        self.assertEqual(len(found), 1, "a preamble line must not hide the table")

    def test_a_grading_table_is_not_a_coordinate_table(self):
        # Same ids, one numeric column. It must be reported as readable-but-not-
        # coordinates, never counted as a coordinate source.
        import pandas as pd
        pd.DataFrame({"Image name": sorted(self.ids),
                      "Retinopathy grade": [1] * len(self.ids)}
                     ).to_csv(self.root / "gt" / "grades.csv", index=False)
        from verify_dr.data.idrid_tables import find_coord_tables
        found, not_coords, _u = find_coord_tables(self.root)
        self.assertEqual(len(found), 1)
        self.assertTrue(any("grades.csv" in p.name for p, _w in not_coords))

    def test_unreadable_is_distinct_from_absent(self):
        # The distinction that matters: an earlier version collapsed both into
        # "no Part C coordinate tables found", which sent the search the wrong way.
        (self.root / "gt" / "broken.xlsx").write_bytes(b"not really a workbook")
        from verify_dr.data.idrid_tables import find_coord_tables, report
        found, _n, unreadable = find_coord_tables(self.root)
        self.assertEqual(len(found), 1)
        self.assertTrue(any("broken.xlsx" in p.name for p, _w in unreadable))
        self.assertIn("UNREADABLE", report(found, _n, unreadable))

    def test_empty_root_says_so(self):
        import tempfile
        from verify_dr.data.idrid_tables import find_coord_tables, report
        with tempfile.TemporaryDirectory() as empty:
            f, n, u = find_coord_tables(Path(empty))
            self.assertEqual((f, n, u), ({}, [], []))
            self.assertIn("no .csv/.xlsx tables", report(f, n, u))

    def test_columns_not_named_x_and_y_still_resolve(self):
        # A mirror labelling them 'OD Center X' / 'OD Center Y' has the values but
        # fails a name test that requires the name to START with x or y.
        import pandas as pd
        (self.root / "gt" / "IDRiD_OD_Center_Markups.csv").unlink()
        pd.DataFrame({"Image name": sorted(self.ids),
                      "OD Center X": range(1500, 1500 + len(self.ids)),
                      "OD Center Y": range(900, 900 + len(self.ids))}
                     ).to_csv(self.root / "gt" / "t.csv", index=False)
        from verify_dr.data.idrid_tables import find_coord_tables
        found, _n, _u = find_coord_tables(self.root)
        self.assertEqual(len(found), 1)
        _ids, _col, pair = next(iter(found.values()))
        self.assertEqual(pair, ("OD Center X", "OD Center Y"))

    def test_opaque_column_names_resolve_by_value(self):
        import pandas as pd
        (self.root / "gt" / "IDRiD_OD_Center_Markups.csv").unlink()
        pd.DataFrame({"Image": sorted(self.ids),
                      "Col1": range(1500, 1500 + len(self.ids)),
                      "Col2": range(900, 900 + len(self.ids))}
                     ).to_csv(self.root / "gt" / "t.csv", index=False)
        from verify_dr.data.idrid_tables import find_coord_tables
        found, _n, _u = find_coord_tables(self.root)
        self.assertEqual(len(found), 1, "values alone must be enough")

    def test_two_small_int_columns_are_grades_not_coordinates(self):
        # IDRiD's Part B table has TWO numeric columns beside the same ids
        # (retinopathy grade, macular oedema risk). Magnitude is what separates
        # them from a coordinate pair, so a bare "2+ numeric columns" rule fails.
        import pandas as pd
        (self.root / "gt" / "IDRiD_OD_Center_Markups.csv").unlink()
        pd.DataFrame({"Image name": sorted(self.ids),
                      "Retinopathy grade": [i % 5 for i in range(len(self.ids))],
                      "Risk of macular edema": [i % 3 for i in range(len(self.ids))]}
                     ).to_csv(self.root / "gt" / "grades.csv", index=False)
        from verify_dr.data.idrid_tables import find_coord_tables
        found, not_coords, _u = find_coord_tables(self.root)
        self.assertEqual(found, {}, "a grading table is not a coordinate table")
        self.assertTrue(any("grades.csv" in p.name for p, _w in not_coords))


class TestChannelVocabulary(unittest.TestCase):
    """The mask directory names must be the ones build_cache.py writes.

    They were not: segmentation.py used DDR's source spelling (MA/HE/EX/SE) while
    the cache holds build_cache's LESION_CHANNELS (microaneurysm/...). mask_path
    then looked in directories that do not exist, and C2 reported "no masks on
    disk" against a cache holding every one of them. Nothing failed loudly --
    an absent mask is a legal all-zero target, so every image simply looked
    lesion-free.
    """

    def _build_cache_channels(self):
        # Imported by path: scripts/ is not a package, and build_cache pulls in
        # cv2 at call time rather than import time, so this stays cheap.
        import ast
        src = Path(__file__).resolve().parent.parent / "scripts" / "build_cache.py"
        tree = ast.parse(src.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "LESION_CHANNELS" for t in node.targets):
                return tuple(ast.literal_eval(node.value))
        self.fail("build_cache.py no longer defines LESION_CHANNELS")

    def test_mask_dirs_match_what_build_cache_writes(self):
        from verify_dr.data.segmentation import MASK_DIRS
        self.assertEqual(MASK_DIRS, self._build_cache_channels())

    def test_model_and_data_agree_on_channels(self):
        from verify_dr.data.segmentation import MASK_DIRS
        from verify_dr.models.evidence import MASK_DIRS as MODEL_DIRS
        self.assertEqual(MASK_DIRS, MODEL_DIRS)

    def test_short_labels_are_distinct_and_aligned(self):
        from verify_dr.data.segmentation import LESION_NAMES, SHORT_LABELS
        self.assertEqual(len(SHORT_LABELS), len(LESION_NAMES))
        self.assertEqual(len(set(SHORT_LABELS)), len(SHORT_LABELS))
