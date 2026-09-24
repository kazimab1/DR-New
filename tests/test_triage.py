"""Scoring for Phase 7 (src/verify_dr/triage: ood, selective, signals, claims).

The load-bearing properties, each against an answer known independently:
the tie rule against brute-force enumeration of orderings, Ledoit-Wolf against
scikit-learn, the fast AUC against the plain curve, the faithfulness columns against
the per-image rule `faithfulness.faithful`, and the claim rule against each way it
can fail.
"""

import itertools
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verify_dr.triage import claims as C  # noqa: E402
from verify_dr.triage import faithfulness as F  # noqa: E402
from verify_dr.triage import ood as O  # noqa: E402
from verify_dr.triage import selective as S  # noqa: E402
from verify_dr.triage import signals as G  # noqa: E402


# ------------------------------------------------------------------ OOD


class Mahalanobis(unittest.TestCase):

    def test_ledoit_wolf_matches_scikit_learn(self):
        try:
            from sklearn.covariance import ledoit_wolf as reference
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(0)
        for n, p in ((50, 80), (400, 30), (2000, 64)):
            x = rng.normal(size=(n, p)) @ rng.normal(size=(p, p)) * 0.3
            x -= x.mean(axis=0)
            ours, shrink = O.ledoit_wolf(x)
            theirs, their_shrink = reference(x, assume_centered=True)
            self.assertAlmostEqual(shrink, their_shrink, places=10)
            np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-12)

    @staticmethod
    def reference_sample(n=3000, d=16, seed=1):
        rng = np.random.default_rng(seed)
        y = rng.choice(5, size=n, p=[0.6, 0.1, 0.15, 0.1, 0.05])
        centres = rng.normal(0, 3, size=(5, d))
        return centres[y] + rng.normal(size=(n, d)), y, centres

    def test_distance_is_the_nearest_class_in_the_shared_metric(self):
        emb, y, _ = self.reference_sample()
        g = O.fit_class_gaussian(emb, y)
        probe = np.random.default_rng(2).normal(0, 4, size=(50, emb.shape[1]))
        brute = np.min([np.sqrt(np.einsum("ij,jk,ik->i", probe - m, g.precision, probe - m))
                        for m in g.means], axis=0)
        np.testing.assert_allclose(O.min_mahalanobis(probe, g), brute, rtol=1e-9)
        np.testing.assert_allclose(g.precision @ np.linalg.inv(g.precision),
                                   np.eye(len(g.precision)), atol=1e-8)

    def test_shifted_images_score_higher_and_5pct_of_calibration_is_flagged(self):
        emb, y, centres = self.reference_sample()
        g = O.fit_class_gaussian(emb, y)
        cal, _, _ = self.reference_sample(seed=3)
        # the same world: resample with the reference's centres
        rng = np.random.default_rng(4)
        yc = rng.choice(5, size=4000, p=[0.6, 0.1, 0.15, 0.1, 0.05])
        cal = centres[yc] + rng.normal(size=(4000, emb.shape[1]))
        scale = O.fit_scale(O.min_mahalanobis(cal, g))
        z_cal = O.z_score(O.min_mahalanobis(cal, g), scale)
        self.assertAlmostEqual(z_cal.mean(), 0.0, places=10)
        self.assertAlmostEqual(z_cal.std(ddof=1), 1.0, places=10)
        self.assertAlmostEqual((z_cal >= scale.tau).mean(), 0.05, delta=0.002)
        shifted = centres[yc[:500]] + rng.normal(size=(500, emb.shape[1])) + 3.0
        z_shift = O.z_score(O.min_mahalanobis(shifted, g), scale)
        self.assertGreater((z_shift >= scale.tau).mean(), 0.9)

    def test_a_grade_with_no_reference_images_is_refused(self):
        emb, y, _ = self.reference_sample()
        with self.assertRaises(ValueError):
            O.fit_class_gaussian(emb[y != 4], y[y != 4])

    def test_statistics_round_trip_and_a_wrong_digest_is_refused(self):
        emb, y, _ = self.reference_sample()
        g = O.fit_class_gaussian(emb, y)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ood" / "m.npz"
            digest = O.save_gaussian(g, path)
            again = O.save_gaussian(g, Path(d) / "again.npz")
            self.assertEqual(digest, again, "the same statistics must give the same digest")
            back = O.load_gaussian(path, digest)
            np.testing.assert_array_equal(back.precision, g.precision)
            with self.assertRaises(ValueError):
                O.load_gaussian(path, "0" * 64)


# ------------------------------------------------------------------ selective


def brute_force_auc(score, correct):
    """Average AUC over every ordering consistent with the score -- the definition."""
    score, correct = np.asarray(score), np.asarray(correct, dtype=float)
    blocks = [np.flatnonzero(score == s) for s in np.unique(score)]
    total, count = 0.0, 0
    for perms in itertools.product(*[itertools.permutations(b) for b in blocks]):
        order = np.concatenate([np.array(p) for p in perms])
        acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
        total += acc.mean()
        count += 1
    return total / count


class CoverageAccuracy(unittest.TestCase):

    def test_ties_are_resolved_by_expectation_exactly(self):
        rng = np.random.default_rng(5)
        for _ in range(25):
            n = int(rng.integers(3, 8))
            score = rng.integers(0, 3, n).astype(float)
            correct = rng.integers(0, 2, n)
            rank = S.dense_rank(score)
            self.assertAlmostEqual(S.auc(rank, correct), brute_force_auc(score, correct), places=12)

    def test_the_fast_auc_equals_the_plain_curve(self):
        rng = np.random.default_rng(6)
        score = rng.integers(0, 40, 5000).astype(float)
        correct = rng.random(5000) < 0.7
        rank = S.dense_rank(score)
        self.assertAlmostEqual(S.auc(rank, correct), S.accuracy_curve(rank, correct).mean(),
                               places=12)

    def test_the_none_arm_scores_exactly_the_overall_accuracy(self):
        correct = np.random.default_rng(7).random(1234) < 0.8
        rank = S.dense_rank(np.zeros(1234))
        self.assertAlmostEqual(S.auc(rank, correct), correct.mean(), places=12)
        self.assertAlmostEqual(S.accuracy_at(rank, correct, 0.8), correct.mean(), places=12)

    def test_a_known_curve(self):
        rank = S.dense_rank(np.array([0.0, 1.0, 2.0, 3.0]))
        self.assertAlmostEqual(S.auc(rank, np.array([1, 1, 0, 0])),
                               (1 + 1 + 2 / 3 + 2 / 4) / 4, places=12)

    def test_lexicographic_keys_never_merge_distinct_cases(self):
        level = np.array([0, 0, 1, 1, 2])
        conf = np.array([0.3, 0.3, 1e-17, 0.0, 0.5])
        rank = S.dense_rank(level, conf)
        self.assertEqual(rank.tolist(), [0, 0, 2, 1, 3])

    def test_coverage_points_and_acceptance_probabilities(self):
        self.assertEqual(S.k_for(0.8, 17615), 14092)
        self.assertEqual(S.k_for(0.8, 3662), 2930)
        self.assertEqual(S.k_for(0.9, 10), 9)
        rank = S.dense_rank(np.array([0, 1, 1, 1, 2.0]))
        p = S.acceptance_probability(rank, 3)
        np.testing.assert_allclose(p, [1, 2 / 3, 2 / 3, 2 / 3, 0])
        self.assertAlmostEqual(p.sum(), 3.0)

    def test_referral_counts_are_expected_counts(self):
        # Five cases, the last two tied; at 80% (k = 4) half of the tied pair is accepted.
        rank = S.dense_rank(np.array([0, 1, 2, 3, 3.0]))
        yhat = np.array([2, 0, 2, 0, 2])
        y = np.array([2, 0, 0, 2, 2])
        r = S.referral_at(rank, yhat, y, 0.8)
        # accepted: 0 (TP), 1 (TN), 2 (FP), and half each of 3 (FN) and 4 (TP)
        self.assertAlmostEqual(r.sensitivity, 1.5 / 2.0)
        self.assertAlmostEqual(r.specificity, 1.0 / 2.0)
        self.assertAlmostEqual(r.referable_deferred, 1.0 / 3.0)


# ------------------------------------------------------------------ signals


class Signals(unittest.TestCase):

    def test_d_evidence_is_zero_where_M3_cannot_speak(self):
        yhat = np.array([0, 0, 1, 2, 3, 4])
        e = np.array([0, 2, 0, 2, 0, 1])
        self.assertEqual(G.d_evidence(yhat, e, np.full(6, 2)).tolist(), [0, 2, 1, 0, 0, 0])

    def test_faith_outcome_matches_the_per_image_rule(self):
        rng = np.random.default_rng(8)
        n, k = 400, F.CONTROLS
        e_orig = rng.normal(1.5, 0.5, n)
        e_lesion = e_orig - rng.normal(0.1, 0.2, n)
        ctrl = e_orig[:, None] - rng.normal(0.0, 0.15, (n, k))
        ctrl[rng.random((n, k)) < 0.1] = np.nan
        ctrl[:20, :6] = np.nan                        # some rows below 15 controls
        e_lesion[380:] = np.nan                       # some rows with no lesion
        ctrl[380:] = np.nan
        out = G.faith_outcome(e_orig, e_lesion, ctrl)
        for i in range(n):
            if not np.isfinite(e_lesion[i]):
                self.assertEqual(out[i], G.NO_LESION)
                continue
            verdict = F.faithful(e_orig[i] - e_lesion[i], e_orig[i] - ctrl[i])
            expected = {None: G.UNDETERMINED, True: G.FAITHFUL, False: G.UNFAITHFUL}[verdict]
            self.assertEqual(out[i], expected, f"row {i}")
        self.assertTrue({G.FAITHFUL, G.UNFAITHFUL, G.UNDETERMINED, G.NO_LESION} <= set(out))
        self.assertEqual(G.d_faith(out).tolist(), (out == G.UNFAITHFUL).astype(int).tolist())

    def test_a_tie_with_the_best_control_is_not_faithful(self):
        ctrl = np.full((1, F.CONTROLS), 1.0)
        ctrl[0, 0] = 0.8                              # the best control moves E by 0.2
        out = G.faith_outcome(np.array([1.0]), np.array([0.8]), ctrl)
        self.assertEqual(out[0], G.UNFAITHFUL)

    def test_every_defer_clause_and_the_adjacent_band(self):
        base = dict(d_ev=0, d_fa=0, z=0.0, e=1, conf=0.1)
        cases = [({}, 0), ({"d_ev": 1}, 1), ({"d_ev": 2}, 2), ({"d_fa": 1}, 2),
                 ({"z": 2.0}, 2), ({"e": 0, "conf": 0.5}, 2), ({"e": 0, "conf": 0.1}, 0),
                 ({"e": 1, "conf": 0.5}, 0), ({"d_ev": 1, "d_fa": 1}, 2)]
        for change, level in cases:
            c = {**base, **change}
            got = G.combined_level(np.array([c["d_ev"]]), np.array([c["d_fa"]]),
                                   np.array([c["z"]]), 1.645, np.array([c["e"]]),
                                   np.array([c["conf"]]), 0.4)[0]
            self.assertEqual(got, level, f"{change}")

    def test_adjacent_grade_set_accuracy_is_within_one(self):
        table = G.action_table(np.array([1, 1, 1, 0, 2]), np.array([1, 1, 1, 2, 3]),
                               np.array([0, 2, 3, 2, 1]))
        self.assertAlmostEqual(table["ADJACENT_GRADE_SET"]["accuracy"], 2 / 3)
        self.assertEqual(table["ACCEPT"]["accuracy"], 1.0)
        self.assertEqual(table["DEFER"]["accuracy"], 0.0)
        self.assertAlmostEqual(sum(v["share"] for v in table.values()), 1.0)


# ------------------------------------------------------------------ claims


class ClaimRule(unittest.TestCase):

    def test_resamples_are_identical_on_every_call(self):
        a = [idx.copy() for idx in C.resample_indices(50, count=5)]
        b = list(C.resample_indices(50, count=5))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_supported_only_when_all_three_hold(self):
        good = C.claim([0.03, 0.04, 0.035], [(0.01, 0.05), (0.02, 0.06), (0.015, 0.05)])
        self.assertTrue(good["supported"])
        # (a) one seed the wrong way
        self.assertFalse(C.claim([0.03, -0.01, 0.035], [(0.01, 0.05), (-0.03, 0.01),
                                                         (0.015, 0.05)])["direction"])
        # (b) one interval touching zero
        b = C.claim([0.03, 0.04, 0.035], [(0.01, 0.05), (-0.001, 0.06), (0.015, 0.05)])
        self.assertFalse(b["precision"])
        self.assertFalse(b["supported"])
        # (c) a positive effect smaller than its own seed spread
        c = C.claim([0.01, 0.05, 0.012], [(0.001, 0.02), (0.03, 0.07), (0.002, 0.02)])
        self.assertTrue(c["direction"] and c["precision"])
        self.assertFalse(c["stability"])
        self.assertFalse(c["supported"])

    def test_must_hold_on_both_externals(self):
        yes, no = {"supported": True}, {"supported": False}
        self.assertTrue(C.on_every_dataset({"aptos": yes, "messidor2": yes}))
        self.assertFalse(C.on_every_dataset({"aptos": yes, "messidor2": no}))
        self.assertFalse(C.on_every_dataset({}))

    def test_interval_is_the_central_95_percent(self):
        lo, hi = C.interval(np.arange(1001, dtype=float))
        self.assertAlmostEqual(lo, 25.0)
        self.assertAlmostEqual(hi, 975.0)
        self.assertTrue(math.isnan(C.interval([np.nan])[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
