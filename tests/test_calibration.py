"""M4a calibration (src/verify_dr/calibration, ANALYSIS_PLAN.md s4).

Each stage is checked against a case whose right answer is known by construction:
a temperature that was applied on purpose, a prior that was shifted on purpose.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verify_dr.calibration import (  # noqa: E402
    BCTS, bcts_probs, coral_probs, em_prior, fit_bcts, fit_bias, fit_temperature, grade_prior,
    nll, predicted_confidence, prior_correct, stage01,
)
from verify_dr.models.grading import cumulative_to_probs  # noqa: E402

THRESHOLDS = np.array([1.2, 0.2, -1.0, -2.2])       # CORAL biases: decreasing


def coral_logits(n, seed, spread=2.0):
    """Cumulative logits of a CORAL model: one latent score, four decreasing biases."""
    eta = np.random.default_rng(seed).normal(0.0, spread, n)
    return eta[:, None] + THRESHOLDS[None, :]


def sample(p, seed):
    """One grade per row from each row's distribution."""
    u = np.random.default_rng(seed).random(len(p))
    return (u[:, None] > np.cumsum(p, axis=1)).sum(axis=1).clip(0, p.shape[1] - 1)


class Probabilities(unittest.TestCase):

    def test_numpy_probabilities_equal_the_training_code(self):
        z = coral_logits(500, 0)
        np.testing.assert_allclose(coral_probs(z), cumulative_to_probs(torch.tensor(z)).numpy(),
                                   atol=1e-12)
        # The clamp path too: logits no CORAL head can emit, still identical.
        z = np.random.default_rng(1).normal(0, 3, (500, 4))
        np.testing.assert_allclose(coral_probs(z), cumulative_to_probs(torch.tensor(z)).numpy(),
                                   atol=1e-12)

    def test_prior_correction_multiplies_by_the_ratio(self):
        uniform = np.full((1, 5), 0.2)
        target = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
        np.testing.assert_allclose(prior_correct(uniform, target)[0], target)
        p = np.array([[0.1, 0.2, 0.3, 0.2, 0.2]])
        expected = p * target / 0.2
        np.testing.assert_allclose(prior_correct(p, target), expected / expected.sum())
        # And back again: correcting to a prior and then away from it is the identity.
        np.testing.assert_allclose(prior_correct(prior_correct(p, target), [0.2] * 5, target), p)

    def test_confidence_is_the_predicted_grade_not_the_maximum(self):
        p = np.array([[0.1, 0.5, 0.4, 0.0, 0.0]])
        self.assertEqual(predicted_confidence(p, np.array([2]))[0], 0.4)

    def test_grade_prior_counts(self):
        np.testing.assert_allclose(grade_prior([0, 0, 0, 1, 4]), [0.6, 0.2, 0.0, 0.0, 0.2])


class Temperature(unittest.TestCase):

    def test_recovers_a_temperature_applied_on_purpose(self):
        z = coral_logits(40000, 2)
        y = sample(coral_probs(z), 3)
        for t_true in (0.6, 1.0, 2.5):
            t = fit_temperature(z * t_true, y)
            self.assertAlmostEqual(t / t_true, 1.0, delta=0.04, msg=f"T={t_true}")

    def test_with_the_prior_correction_it_recovers_T_and_without_it_does_not(self):
        """The D11 situation: a model trained on a uniform prior, deployed at EyePACS's.

        The data come from p ~ coral(z) * pi. Fitting T with the correction recovers
        the temperature; fitting T alone must absorb a prior mismatch it cannot
        represent, and misses.
        """
        pi = np.array([0.73, 0.07, 0.15, 0.03, 0.02])
        z = coral_logits(40000, 4)
        y = sample(prior_correct(coral_probs(z), pi), 5)
        t_true = 1.8
        with_prior = fit_temperature(z * t_true, y, pi)
        self.assertAlmostEqual(with_prior / t_true, 1.0, delta=0.05)
        self.assertLess(nll(stage01(z * t_true, with_prior, pi), y),
                        nll(coral_probs(z * t_true, fit_temperature(z * t_true, y)), y))

    def test_the_search_stays_inside_its_bounds(self):
        z = coral_logits(2000, 6)
        y = sample(coral_probs(z), 7)
        self.assertLessEqual(fit_temperature(z * 1000.0, y), 20.0)
        self.assertGreaterEqual(fit_temperature(z * 1e-4, y), 0.05)


class EM(unittest.TestCase):
    """A discrete world where the right prior is known exactly."""

    @staticmethod
    def world(pi_src, pi_tgt, n, seed):
        rng = np.random.default_rng(seed)
        likelihood = rng.dirichlet(np.ones(12) * 0.6, size=5)       # [class, feature]
        y = rng.choice(5, size=n, p=pi_tgt)
        x = np.array([rng.choice(12, p=likelihood[c]) for c in y])
        joint = likelihood[:, x].T * pi_src                         # [n, class]
        return joint / joint.sum(axis=1, keepdims=True), y

    def test_recovers_a_shifted_prior_from_unlabelled_posteriors(self):
        pi_src = np.array([0.73, 0.07, 0.15, 0.03, 0.02])
        pi_tgt = np.array([0.49, 0.10, 0.27, 0.05, 0.09])            # APTOS-like
        p_src, _ = self.world(pi_src, pi_tgt, 60000, 8)
        result = em_prior(p_src, pi_src)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.prior, pi_tgt, atol=0.03)
        self.assertAlmostEqual(result.prior.sum(), 1.0, places=12)

    def test_no_shift_means_the_prior_stays_put(self):
        pi_src = np.array([0.73, 0.07, 0.15, 0.03, 0.02])
        p_src, _ = self.world(pi_src, pi_src, 60000, 9)
        np.testing.assert_allclose(em_prior(p_src, pi_src).prior, pi_src, atol=0.03)

    def test_stops_at_the_iteration_cap(self):
        pi_src = np.array([0.73, 0.07, 0.15, 0.03, 0.02])
        p_src, _ = self.world(pi_src, np.full(5, 0.2), 5000, 10)
        result = em_prior(p_src, pi_src, tol=0.0, max_iter=7)
        self.assertEqual((result.iterations, result.converged), (7, False))


class BiasCorrectedTemperature(unittest.TestCase):
    """D13 (ANALYSIS_PLAN.md addendum A.2): one temperature plus a bias per grade."""

    BIAS = np.array([0.0, -1.2, 0.4, -2.0, -2.5])

    @classmethod
    def setUpClass(cls):
        z = coral_logits(10000, 21)
        q = coral_probs(z) * np.exp(cls.BIAS)
        cls.y = sample(q / q.sum(axis=1, keepdims=True), 22)
        cls.z = z * 1.7                                    # over-confident on purpose
        cls.fit = fit_bcts(cls.z, cls.y)

    def test_recovers_the_temperature_and_biases_it_was_given(self):
        self.assertAlmostEqual(self.fit.temperature / 1.7, 1.0, delta=0.05)
        np.testing.assert_allclose(self.fit.bias, self.BIAS, atol=0.15)
        self.assertEqual(self.fit.bias[0], 0.0)

    def test_the_mean_posterior_is_the_grade_mix_exactly(self):
        q = bcts_probs(self.z, self.fit)
        np.testing.assert_allclose(q.mean(axis=0), grade_prior(self.y), atol=1e-9)

    def test_em_does_not_drift_where_nothing_shifted(self):
        """EM's premise, which stages 0+1 broke on the real calibration split."""
        q = bcts_probs(self.z, self.fit)
        result = em_prior(q, grade_prior(self.y))
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.prior, grade_prior(self.y), atol=1e-6)

    def test_where_a_fixed_prior_correction_breaks_the_premise_bcts_does_not(self):
        """The rehearsal's failure in miniature: the model's implicit prior is not the
        uniform one stage 0 assumes, so stage 0+1 posteriors average to the wrong grade
        mix and EM, run with no shift at all, walks away from the truth."""
        pi = grade_prior(self.y)
        p01 = stage01(self.z, fit_temperature(self.z, self.y, pi), pi)
        drift_01 = np.abs(em_prior(p01, pi).prior - pi).max()
        q = bcts_probs(self.z, self.fit)
        drift_bcts = np.abs(em_prior(q, pi).prior - pi).max()
        self.assertGreater(drift_01, 0.05)
        self.assertLess(drift_bcts, 1e-6)

    def test_bias_fit_refuses_a_grade_with_no_images(self):
        z = coral_logits(200, 23)
        y = np.zeros(200, dtype=int)
        with self.assertRaises(ValueError):
            fit_bias(coral_probs(z), y)

    def test_saturated_probabilities_do_not_break_the_bias_fit(self):
        """At the search's extreme temperatures some grades' probabilities are exactly 0
        for every image, the Hessian is singular, and an undamped solve raises. Found by
        running the fit on the fixture; the damped fallback must take over instead."""
        z = coral_logits(400, 25) * 50.0
        p = coral_probs(z, 0.05)
        y = np.argmax(p, axis=1)
        y[:5] = np.arange(5)                              # every grade present
        bias = fit_bias(p, y)
        self.assertTrue(np.all(np.isfinite(bias)))

    def test_zero_biases_reduce_to_temperature_scaling(self):
        z = coral_logits(50, 24)
        np.testing.assert_allclose(bcts_probs(z, BCTS(0.8, np.zeros(5))), coral_probs(z, 0.8),
                                   atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
