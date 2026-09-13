"""Properties the Phase 3 code must not lose.

Run with:  python -m unittest discover -s tests -v

Everything here is a property that was verified by hand while building M1 and is
cheap to check again. The metric definitions in particular are load-bearing: every
number in the thesis is computed by them, and a silent change to one would be very
hard to notice later.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verify_dr.data.dataset import make_sampler, repath_to_cache  # noqa: E402
from verify_dr.evaluation.metrics import (  # noqa: E402
    auroc, expected_calibration_error, grading_metrics, macro_f1,
    quadratic_weighted_kappa,
)
from verify_dr.models.grading import (  # noqa: E402
    OrdinalHead, cumulative_to_grade, cumulative_to_probs,
)
from verify_dr.models.losses import GradingLoss, ordinal_targets  # noqa: E402


class TestOrdinalHead(unittest.TestCase):
    def test_thresholds_are_monotone_for_any_parameters(self):
        """The CORAL guarantee. It must hold by construction, not by training --
        an independent K-1-logit head violates it regularly, and the class
        probabilities derived from it then go negative."""
        head = OrdinalHead(16)
        for _ in range(200):
            with torch.no_grad():
                head.beta.normal_(0, 5)
                head.deltas.normal_(0, 5)
            thresholds = head.thresholds()
            gaps = thresholds[:-1] - thresholds[1:]
            self.assertTrue(bool((gaps > 0).all()), f"non-monotone: {thresholds}")

    def test_probabilities_are_a_distribution(self):
        head = OrdinalHead(16)
        probs = cumulative_to_probs(head(torch.randn(64, 16) * 10))
        self.assertTrue(np.allclose(probs.sum(1).detach().numpy(), 1.0, atol=1e-5))
        self.assertGreaterEqual(float(probs.min().detach()), 0.0)

    def test_grade_is_the_number_of_thresholds_passed(self):
        logits = torch.tensor([[9.0, 9.0, -9.0, -9.0], [-9.0] * 4, [9.0] * 4])
        self.assertEqual(cumulative_to_grade(logits).tolist(), [2, 0, 4])

    def test_head_can_memorise(self):
        """If the head plus loss cannot overfit its own capacity, the wiring is wrong.

        The feature width matches the sample count deliberately. A CORAL head
        projects to a *single* scalar and then slices it with K-1 thresholds, so
        its capacity is bounded by the feature width, not by K: OrdinalHead(8)
        cannot order 64 arbitrary labels however long it trains, and failing that
        would say nothing about the wiring. The real model feeds it a 512-wide
        neck.
        """
        torch.manual_seed(0)
        head, features = OrdinalHead(64), torch.randn(64, 64)
        grades = torch.randint(0, 5, (64,))
        criterion = GradingLoss(head="ordinal")
        optimiser = torch.optim.Adam(head.parameters(), lr=0.1)
        for _ in range(400):
            optimiser.zero_grad()
            logits = head(features)
            out = {"logits": logits, "rdr": logits[:, 1], "vtdr": logits[:, 2]}
            criterion(out, grades)["loss"].backward()
            optimiser.step()
        accuracy = (cumulative_to_grade(head(features)) == grades).float().mean()
        self.assertGreater(float(accuracy), 0.75)


class TestLosses(unittest.TestCase):
    def test_ordinal_targets_are_a_staircase(self):
        targets = ordinal_targets(torch.arange(5))
        self.assertEqual(targets.tolist(), [
            [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1],
        ])


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0] * 12 + [1] * 8 + [2] * 8 + [3] * 6 + [4] * 14)
        self.probs = np.full((len(self.y), 5), 0.2)

    def test_qwk_known_answers(self):
        self.assertAlmostEqual(quadratic_weighted_kappa(self.y, self.y), 1.0)
        perfect_disagreement = np.array([0, 4, 0, 4])
        self.assertAlmostEqual(
            quadratic_weighted_kappa(perfect_disagreement, 4 - perfect_disagreement), -1.0)

    def test_qwk_barely_punishes_skipping_grade_1(self):
        """The reason B1 is not decided on QWK. A model that never predicts grade 1
        still scores ~0.97, because grade 1 is one step from grade 0 and the
        quadratic weight on that error is the smallest non-zero one there is."""
        skips_grade_1 = self.y.copy()
        skips_grade_1[skips_grade_1 == 1] = 0
        self.assertGreater(quadratic_weighted_kappa(self.y, skips_grade_1), 0.95)

    def test_grade_1_recall_alone_cannot_decide_b1(self):
        """Recall is 1.000 both when the class is learned and when the model
        predicts it for everything. F1 and distinct_predictions separate them."""
        degenerate = grading_metrics(self.y, np.ones_like(self.y), self.probs)
        learned = grading_metrics(self.y, self.y.copy(), self.probs)

        self.assertEqual(degenerate["per_class_recall"]["1"], 1.0)
        self.assertEqual(learned["per_class_recall"]["1"], 1.0)

        self.assertLess(degenerate["per_class_f1"]["1"], 0.3)
        self.assertEqual(learned["per_class_f1"]["1"], 1.0)
        self.assertEqual(degenerate["distinct_predictions"], 1)
        self.assertEqual(learned["distinct_predictions"], 5)

    def test_macro_f1_punishes_a_skipped_class(self):
        """Accuracy does not. This is why the register reports macro-F1."""
        y = np.array([0] * 90 + [1] * 10)
        pred = np.zeros_like(y)
        self.assertAlmostEqual(float((pred == y).mean()), 0.90)
        self.assertLess(macro_f1(y, pred), 0.50)

    def test_auroc_known_answers(self):
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels), 1.0)
        self.assertAlmostEqual(auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels), 0.0)
        self.assertAlmostEqual(auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels), 0.5)

    def test_ece_known_answers(self):
        correct = np.array([1.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(expected_calibration_error(np.full(4, 0.5), correct), 0.0)
        self.assertAlmostEqual(expected_calibration_error(np.full(4, 0.9), correct), 0.4)

    def test_ece_uses_the_predicted_grade_not_the_argmax(self):
        """Under the ordinal head the grade is the threshold count, which need not be
        the argmax. Using probs.max() would measure the calibration of a prediction
        the model never made."""
        y_true = np.array([0, 0])
        y_pred = np.array([1, 1])                       # threshold count says 1
        probs = np.array([[0.4, 0.35, 0.1, 0.1, 0.05]] * 2)   # argmax says 0
        out = grading_metrics(y_true, y_pred, probs)
        self.assertAlmostEqual(out["ece"], 0.35, places=6)    # not 0.40
        self.assertEqual(out["argmax_agreement"], 0.0)


class TestDataset(unittest.TestCase):
    def test_repath_keeps_the_dataset_images_file_tail(self):
        frame = pd.DataFrame({"image_path": ["/old/mount/eyepacs/images/1_left.jpg"]})
        out = repath_to_cache(frame, Path("/kaggle/input/new"))
        self.assertEqual(out["image_path"][0],
                         "/kaggle/input/new/eyepacs/images/1_left.jpg")

    def test_repath_resolves_each_dataset_to_the_root_that_holds_it(self):
        """The cache is legitimately split: IDRiD's masks ship as their own
        published dataset, so a manifest spanning datasets must not be forced
        onto a single root."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root_a, root_b = Path(tmp) / "a", Path(tmp) / "b"
            (root_a / "eyepacs" / "images").mkdir(parents=True)
            (root_b / "idrid" / "images").mkdir(parents=True)
            (root_a / "eyepacs" / "images" / "1_left.jpg").write_bytes(b"x")
            (root_b / "idrid" / "images" / "IDRiD_01.jpg").write_bytes(b"x")

            frame = pd.DataFrame({"image_path": [
                "/gone/eyepacs/images/1_left.jpg",
                "/gone/idrid/images/IDRiD_01.jpg",
            ]})
            out = repath_to_cache(frame, [root_a, root_b])
            self.assertEqual(out["image_path"][0],
                             str(root_a / "eyepacs/images/1_left.jpg"))
            self.assertEqual(out["image_path"][1],
                             str(root_b / "idrid/images/IDRiD_01.jpg"))

    def test_repath_falls_back_to_the_first_root_when_nothing_matches(self):
        """So the failure surfaces in load_manifest's existence check with a
        readable path, rather than silently here."""
        out = repath_to_cache(
            pd.DataFrame({"image_path": ["/gone/ddr/images/x.jpg"]}),
            [Path("/no/such/a"), Path("/no/such/b")])
        self.assertEqual(out["image_path"][0], "/no/such/a/ddr/images/x.jpg")

    def test_sampler_never_draws_an_absent_grade(self):
        grades = np.array([0, 0, 0, 2, 2, 4])           # no grade 1 or 3
        sampler = make_sampler(grades, "class_balanced", seed=0)
        drawn = {int(grades[i]) for i in sampler}
        self.assertFalse(drawn & {1, 3})

    def test_natural_sampler_is_plain_shuffling(self):
        self.assertIsNone(make_sampler(np.array([0, 1, 2]), "natural"))

    def test_unknown_sampler_is_rejected(self):
        with self.assertRaises(ValueError):
            make_sampler(np.array([0, 1]), "whatever")


if __name__ == "__main__":
    unittest.main()
