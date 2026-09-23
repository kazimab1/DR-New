"""The single pass (scripts/predict.py, ANALYSIS_PLAN.md s9).

Two kinds of check. Static: the source names a label in exactly one place and reads
a manifest in exactly one function, which drops the labels. End to end: a fixture
cache, real-format checkpoints and a manifest carrying grades go through the real
script, and the outputs are checked for what they must and must not contain.

M2 is replaced by a stand-in that "finds" a lesion wherever a fixture image has a
bright red blob, so the faithfulness pass runs on regions of a known size rather
than on whatever a randomly initialised segmenter happens to emit.
"""

import ast
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import predict  # noqa: E402
from verify_dr.data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms  # noqa: E402
from verify_dr.models.evidence import LesionSegmenter  # noqa: E402
from verify_dr.models.grading import GradingModel, cumulative_to_grade  # noqa: E402
from verify_dr.triage import faithfulness as F  # noqa: E402

SIZE = 64
SOURCE = (ROOT / "scripts" / "predict.py").read_text()


class LabelFreeByConstruction(unittest.TestCase):

    def test_a_label_is_named_only_in_LABEL_COLUMNS(self):
        tree = ast.parse(SOURCE)
        allowed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "LABEL_COLUMNS" for t in node.targets):
                allowed |= {id(c) for c in ast.walk(node.value)}
        stray = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and n.value in predict.LABEL_COLUMNS
                 and id(n) not in allowed]
        self.assertEqual(stray, [], f"a label column is named outside LABEL_COLUMNS: {stray}")

    def test_the_manifest_is_read_in_one_place(self):
        tree = ast.parse(SOURCE)
        callers = []
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Attribute) and n.attr == "read_csv":
                        callers.append(fn.name)
        # merge() re-reads the script's OWN shard outputs, which carry no label.
        self.assertEqual(sorted(set(callers)), ["merge", "read_unlabelled"])

    def test_every_label_spelling_is_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.csv"
            pd.DataFrame({"image_path": ["a"], "grade": [2], "true_grade": [2],
                          "adjudicated_grade": [2], "DR_Level": [2], "dataset": ["x"],
                          "evidence_grade_hint": [0], "upgraded": [0]}).to_csv(path, index=False)
            kept = list(predict.read_unlabelled(path).columns)
        # Conservative on purpose: "grade" as a whole word goes even when it may not
        # be a label; the same letters inside another word ("upgraded") stay.
        self.assertEqual(kept, ["image_path", "dataset", "upgraded"])


class NormalisationMatchesTheEvalTransforms(unittest.TestCase):
    """The pass normalises on the device; it must equal what training evaluated."""

    def test_device_normalisation_equals_build_transforms(self):
        from PIL import Image
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
        ours = predict.normalise(predict.to_chw(rgb)[None])[0]
        theirs = build_transforms(SIZE, train=False)(Image.fromarray(rgb))
        self.assertTrue(torch.allclose(ours, theirs, atol=1e-5))


def fixture_image(red_blob: bool) -> np.ndarray:
    yy, xx = np.mgrid[:SIZE, :SIZE]
    rgb = np.zeros((SIZE, SIZE, 3), np.uint8)
    inside = (yy - 32) ** 2 + (xx - 32) ** 2 <= 28 ** 2
    rgb[inside] = (90, 60, 40)
    if red_blob:
        blob = (yy - 26) ** 2 + (xx - 38) ** 2 <= 3 ** 2
        rgb[blob] = (250, 10, 10)
    return rgb


def fake_segmenter_forward(self, x):
    """Stand-in for M2: a 'microaneurysm' wherever the image is bright red."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    rgb = x * std + mean
    red = (rgb[:, 0] > 0.9) & (rgb[:, 1] < 0.2)
    logits = torch.full((x.shape[0], 4, x.shape[2], x.shape[3]), -10.0)
    logits[:, 0][red] = 10.0
    return logits


class EndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from PIL import Image
        cls.tmp = Path(tempfile.mkdtemp(prefix="predict_"))
        rows = []
        for i in range(8):
            blob = i % 2 == 0
            path = cls.tmp / "images" / f"{100 + i}_left.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(fixture_image(blob)).save(path)
            rows.append({"image_path": str(path), "grade": 3 if blob else 0,
                         "dataset": "EyePACS", "patient_id": f"EyePACS::{100 + i}",
                         "split": "calibration" if i < 6 else "test"})
        cls.manifest = cls.tmp / "eyepacs_full.csv"
        pd.DataFrame(rows).to_csv(cls.manifest, index=False)

        torch.manual_seed(0)
        cls.ckpts = []
        for seed in (42, 43):
            model = GradingModel(pretrained=False)
            path = cls.tmp / "results" / f"H1_eyepacs_full_s{seed}" / "best.pt"
            path.parent.mkdir(parents=True)
            torch.save({"model": model.state_dict(), "epoch": 3,
                        "config": {"image_size": SIZE, "backbone": "efficientnet_b0",
                                   "head": "ordinal_focal", "dropout": 0.3,
                                   "eye_pair_fusion": False}}, path)
            cls.ckpts.append(path)
        cls.m2 = cls.tmp / "results" / "C2_lesions" / "best.pt"
        cls.m2.parent.mkdir(parents=True)
        torch.save({"model": LesionSegmenter(pretrained=False).state_dict(),
                    "config": {"image_size": SIZE}}, cls.m2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_pass(self, out, *extra):
        argv = ["--manifest", str(self.manifest), "--out-dir", str(out),
                "--grading-checkpoints", *map(str, self.ckpts),
                "--device", "cpu", "--workers", "0", "--shard-size", "4", "--no-amp",
                *extra]
        if "--embeddings-only" not in extra:
            argv += ["--evidence-checkpoint", str(self.m2)]
        with mock.patch.object(LesionSegmenter, "forward", fake_segmenter_forward):
            return predict.main(argv)

    def test_calibration_pass_records_everything_and_no_label(self):
        out = self.tmp / "out_cal"
        self.assertEqual(self.run_pass(out, "--split", "calibration"), predict.EXIT_DONE)
        images = pd.read_csv(out / "images.csv")
        m1 = pd.read_csv(out / "m1.csv")

        self.assertEqual(len(images), 6)
        for frame in (images, m1):
            self.assertFalse(set(frame.columns) & set(predict.LABEL_COLUMNS),
                             "a label column reached the output")
        blob = images["image_id"].str.extract(r"::(\d+)_")[0].astype(int) % 2 == 0

        # M3 on M2's findings: a lone microaneurysm is R2, grade 1; nothing is R1.
        self.assertTrue((images.loc[blob, "evidence_grade"] == 1).all())
        self.assertTrue((images.loc[~blob, "evidence_grade"] == 0).all())
        self.assertTrue((images.loc[blob, "rule"] == "R2").all())
        self.assertTrue((images.loc[blob, "faith_status"] == "determined").all())
        self.assertTrue((images.loc[~blob, "faith_status"] == "none").all())
        self.assertTrue((images.loc[blob, "region_px"] > images.loc[blob, "lesion_px"]).all(),
                        "the inpainted region is the lesion dilated by 3 px")

        self.assertEqual(len(m1), 12)
        z = torch.tensor(m1[[f"z{j}" for j in range(4)]].to_numpy())
        self.assertTrue((cumulative_to_grade(z).numpy() == m1["yhat"].to_numpy()).all())
        self.assertTrue(np.allclose(F.expected_grade(z.numpy()), m1["e_orig"], atol=1e-5))
        merged = m1.merge(images[["image_id", "faith_status", "controls_ok"]], on="image_id")
        has = merged["faith_status"] != "none"
        self.assertTrue(merged.loc[has, "e_lesion"].notna().all())
        self.assertTrue(merged.loc[~has, "e_lesion"].isna().all())
        ctrl = merged[[f"e_c{j:02d}" for j in range(F.CONTROLS)]].notna().sum(axis=1)
        self.assertTrue((ctrl == merged["controls_ok"]).all())

        for ckpt in self.ckpts:
            emb = np.load(out / "emb" / f"{ckpt.parent.name}.npy")
            self.assertEqual(emb.shape, (6, 512))
            self.assertEqual(emb.dtype, np.float16)
        run = json.loads((out / "run.json").read_text())
        self.assertFalse(run["labels_written"])
        self.assertEqual(run["grading_checkpoints"][self.ckpts[0].parent.name]["sha256"],
                         predict.sha256(self.ckpts[0]))

    def test_a_stopped_run_resumes_to_the_same_answer(self):
        whole, split = self.tmp / "out_whole", self.tmp / "out_split"
        self.assertEqual(self.run_pass(whole, "--split", "calibration"), predict.EXIT_DONE)
        self.assertEqual(self.run_pass(split, "--split", "calibration", "--max-minutes",
                                       "0.00001"), predict.EXIT_PARTIAL)
        self.assertFalse((split / "images.csv").exists())
        self.assertEqual(self.run_pass(split, "--split", "calibration"), predict.EXIT_DONE)
        pd.testing.assert_frame_equal(pd.read_csv(whole / "images.csv"),
                                      pd.read_csv(split / "images.csv"))
        pd.testing.assert_frame_equal(pd.read_csv(whole / "m1.csv"),
                                      pd.read_csv(split / "m1.csv"))

    def test_locked_rows_are_refused_before_anything_is_read(self):
        out = self.tmp / "out_locked"
        with mock.patch.object(predict, "load_rgb", side_effect=AssertionError("read!")):
            self.assertEqual(self.run_pass(out, "--split", "test"), predict.EXIT_REFUSED)
        self.assertFalse((out / "shards").exists())

    def test_the_OOD_reference_is_embeddings_only(self):
        out = self.tmp / "out_ref"
        self.assertEqual(self.run_pass(out, "--split", "calibration", "--embeddings-only",
                                       "--sample", "3", "--sample-seed", "0"),
                         predict.EXIT_DONE)
        images, m1 = pd.read_csv(out / "images.csv"), pd.read_csv(out / "m1.csv")
        self.assertEqual(len(images), 3)
        self.assertNotIn("faith_status", images.columns)
        self.assertNotIn("e_lesion", m1.columns)
        self.assertEqual(np.load(out / "emb" / f"{self.ckpts[0].parent.name}.npy").shape,
                         (3, 512))

    def test_a_directory_is_never_resumed_with_a_different_run(self):
        out = self.tmp / "out_mixed"
        self.assertEqual(self.run_pass(out, "--split", "calibration", "--embeddings-only"),
                         predict.EXIT_DONE)
        self.assertEqual(self.run_pass(out, "--split", "calibration", "--embeddings-only",
                                       "--sample", "3"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
