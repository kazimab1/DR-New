"""Plan step 2 end to end: fit_params.py then analyse.py on a synthetic internal pass.

The fixture writes pass directories in exactly the format scripts/predict.py writes
(images.csv, m1.csv, emb/<model>.npy, run.json) for six models, two OOD reference
samples, a calibration split, a val split and a locked test split, plus manifests
carrying the grades. The world is built so every signal has something to find: M1
is right about 75% of the time, M3 over-calls lesions the way the real pass showed,
and faithfulness has faithful, unfaithful, undetermined and lesion-free images.
"""

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyse  # noqa: E402
import fit_params  # noqa: E402
from verify_dr.calibration import coral_probs  # noqa: E402
from verify_dr.triage import faithfulness as F  # noqa: E402
from verify_dr.triage import ood as O  # noqa: E402
from verify_dr.triage.params import load_params, write_params  # noqa: E402

MODELS = [f"H1_{v}_s{s}" for v in ("eyepacs_full", "eyepacs_ddr_full") for s in (42, 43, 44)]
PREVALENCE = [0.70, 0.08, 0.15, 0.04, 0.03]
DIM = 24
REFERENCE = 600
THRESHOLDS = np.array([1.0, 0.0, -1.0, -2.0])
CONSTANTS = {"evidence_threshold": 0.5, "min_lesion_px": 4, "controls": F.CONTROLS,
             "min_controls_ok": F.MIN_CONTROLS_OK, "dilate_px": F.DILATE_PX,
             "inpaint_radius": F.INPAINT_RADIUS, "max_attempts": F.MAX_ATTEMPTS,
             "fov_threshold": F.FOV_THRESHOLD, "image_size": 512}


def sha(name):
    return (name.encode().hex() * 8)[:64]


class World:
    """Grades, M1 outputs, M3 evidence and faithfulness for one split."""

    def __init__(self, split, n, seed, dataset="EyePACS", prevalence=PREVALENCE):
        rng = np.random.default_rng(seed)
        self.split, self.n, self.dataset = split, n, dataset
        self.y = rng.choice(5, size=n, p=prevalence)
        self.paths = [f"/cache/{dataset.lower()}/{split}/{seed}_{i}_left.png" for i in range(n)]
        self.ids = [f"{dataset}::{Path(p).stem}" for p in self.paths]
        # M3: finds lesions in half the grade-0 images, more often the sicker the eye.
        u = rng.random(n)
        e = np.where(self.y == 0, np.where(u < 0.5, 0, np.where(u < 0.6, 1, 2)),
                     np.where(self.y == 1, np.where(u < 0.3, 0, np.where(u < 0.7, 1, 2)),
                              np.where(u < 0.1, 0, 2)))
        self.e = e
        self.rng = rng
        self.centres = np.random.default_rng(99).normal(0, 1.5, (5, DIM))

    def model(self, name):
        rng = np.random.default_rng(zlib.crc32(f"{name}|{self.split}|{self.n}".encode()))
        eta = 1.6 * (self.y - 0.8) + rng.normal(0, 0.9, self.n)
        z = 2.2 * (eta[:, None] + THRESHOLDS[None, :])        # over-confident on purpose
        yhat = (z > 0).sum(axis=1)
        e_orig = (1 / (1 + np.exp(-z))).sum(axis=1)
        has = self.e > 0
        drop = np.where(rng.random(self.n) < 0.6, rng.uniform(0.2, 0.6, self.n),
                        rng.uniform(-0.05, 0.05, self.n))
        e_lesion = np.where(has, e_orig - drop, np.nan)
        ctrl = e_orig[:, None] - rng.normal(0, 0.08, (self.n, F.CONTROLS))
        ctrl[rng.random((self.n, F.CONTROLS)) < 0.05] = np.nan
        ctrl[:7, :8] = np.nan                                  # a few undetermined
        ctrl[~has] = np.nan
        emb = self.centres[self.y] + rng.normal(0, 1, (self.n, DIM))
        emb[: self.n // 20] += 4.0                             # a few far from training
        frame = pd.DataFrame({"image_id": self.ids, "model": name})
        for j in range(4):
            frame[f"z{j}"] = z[:, j]
        frame["yhat"] = yhat
        frame["rdr_logit"] = z[:, 1]
        frame["vtdr_logit"] = z[:, 2]
        frame["e_orig"] = e_orig
        frame["e_lesion"] = e_lesion
        for j in range(F.CONTROLS):
            frame[f"e_c{j:02d}"] = ctrl[:, j]
        return frame, emb.astype(np.float16)

    def images(self):
        ok = np.where(self.e > 0, F.CONTROLS, 0)
        ok[:7] = np.where(self.e[:7] > 0, F.CONTROLS - 8, 0)
        rules = np.where(self.e == 0, "R1", np.where(self.e == 1, "R2", "R3"))
        return pd.DataFrame({
            "image_id": self.ids, "dataset": self.dataset, "image_path": self.paths,
            "split": self.split, "evidence_grade": self.e, "max_excludable_grade": 2,
            "rule": rules, "lesion_px": np.where(self.e > 0, 40, 0),
            "region_px": np.where(self.e > 0, 90, 0), "controls_ok": ok,
            "faith_status": np.where(self.e == 0, "none",
                                     np.where(ok >= F.MIN_CONTROLS_OK, "determined",
                                              "undetermined"))})

    def write_pass(self, out, models, embeddings_only=False, locked=False, sample=0):
        out.mkdir(parents=True)
        (out / "emb").mkdir()
        blocks = []
        for name in models:
            frame, emb = self.model(name)
            if embeddings_only:
                frame = frame.drop(columns=["e_lesion"] + [f"e_c{j:02d}" for j in range(F.CONTROLS)])
            blocks.append(frame)
            np.save(out / "emb" / f"{name}.npy", emb)
        pd.concat(blocks, ignore_index=True).to_csv(out / "m1.csv", index=False)
        images = self.images()
        if embeddings_only:
            images = images[["image_id", "dataset", "image_path", "split"]]
        images.to_csv(out / "images.csv", index=False)
        run = {"split": self.split, "rows": self.n, "sample": sample, "sample_seed": 0,
               "embeddings_only": embeddings_only, "locked": locked, "labels_written": False,
               "grading_checkpoints": {m: {"sha256": sha(m)} for m in models},
               "evidence_checkpoint": None if embeddings_only else {"sha256": sha("C2")},
               "constants": CONSTANTS}
        (out / "run.json").write_text(json.dumps(run))

    def manifest_rows(self):
        return pd.DataFrame({"image_path": self.paths, "grade": self.y,
                             "dataset": self.dataset, "split": self.split})


class FitAndRehearse(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="phase7_"))
        cls.patch = mock.patch.object(O, "REFERENCE_SIZE", REFERENCE)
        cls.patch.start()
        worlds = {"train": World("train", REFERENCE, 1), "calibration": World("calibration", 900, 2),
                  "val": World("val", 1200, 3), "test": World("test", 700, 4)}
        ddr_train = World("train", REFERENCE, 5, dataset="DDR", prevalence=[0.5, 0.05, 0.36, 0.02, 0.07])
        pred = cls.tmp / "predictions"
        full = [m for m in MODELS if "_ddr_" not in m]
        ddr = [m for m in MODELS if "_ddr_" in m]
        worlds["train"].write_pass(pred / "reference_eyepacs_full", full, True, sample=REFERENCE)
        mixed = World("train", REFERENCE, 6)
        mixed.y = np.where(np.arange(REFERENCE) % 5 == 0, ddr_train.y, mixed.y)
        mixed.paths = [p if i % 5 else p.replace("/eyepacs/", "/ddr/") for i, p in enumerate(mixed.paths)]
        mixed.ids = [f"{'DDR' if i % 5 == 0 else 'EyePACS'}::{Path(p).stem}"
                     for i, p in enumerate(mixed.paths)]
        cls.mixed = mixed
        mixed.write_pass(pred / "reference_eyepacs_ddr_full", ddr, True, sample=REFERENCE)
        for split in ("calibration", "val"):
            worlds[split].write_pass(pred / split, MODELS)
        worlds["test"].write_pass(cls.tmp / "locked" / "test", MODELS, locked=True)

        manifest = pd.concat([w.manifest_rows() for w in worlds.values()], ignore_index=True)
        cls.full_manifest = cls.tmp / "eyepacs_full.csv"
        manifest.to_csv(cls.full_manifest, index=False)
        m = mixed.manifest_rows()
        m["dataset"] = ["DDR" if i % 5 == 0 else "EyePACS" for i in range(REFERENCE)]
        cls.ddr_manifest = cls.tmp / "eyepacs_ddr_full.csv"
        pd.concat([m, manifest[manifest["split"] != "train"]]).to_csv(cls.ddr_manifest, index=False)
        cls.worlds, cls.pred = worlds, pred

        cls.fitted = cls.tmp / "fitted"
        code = fit_params.main(["--internal", str(pred), "--manifest-full", str(cls.full_manifest),
                                "--manifest-ddr", str(cls.ddr_manifest), "--out", str(cls.fitted)])
        assert code == 0
        cls.params = load_params(cls.fitted / "fitted_params.json")

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def rehearse(self, out, *extra, role="external"):
        return analyse.main(["dataset", "--pass-dir", str(self.pred / "val"),
                             "--labels", str(self.full_manifest), "--split", "val",
                             "--fitted", str(self.fitted / "fitted_params.json"),
                             "--ood-dir", str(self.fitted), "--name", "val", "--role", role,
                             "--out", str(out), "--rehearsal", "--resamples", "40", *extra])

    # ---------------------------------------------------------------- the fit

    def test_every_model_gets_every_parameter_and_the_digest_verifies(self):
        self.assertEqual(sorted(self.params["models"]), sorted(MODELS))
        for name, m in self.params["models"].items():
            self.assertTrue(0.05 <= m["temperature"] <= 20)
            self.assertIn(m["r"], [0.0, 0.5, 1.5, 2.5])
            self.assertAlmostEqual(m["ood"]["calibration_flagged"], 0.05, delta=0.01)
            self.assertEqual(m["ood"]["reference_rows"], REFERENCE)
            O.load_gaussian(self.fitted / m["ood"]["file"], m["ood"]["digest"])
            # the fixture's M1 is over-confident by 2.2x: calibration must soften it
            self.assertGreater(m["temperature"], 1.0)
            self.assertLess(m["calibration_fit"]["nll"]["stage01"], m["calibration_fit"]["nll"]["raw"])
        counts = np.bincount(self.worlds["train"].y, minlength=5)
        self.assertEqual(self.params["pi_src_counts"], counts.tolist())

    def test_r_is_the_best_calibration_auc_with_ties_to_the_smaller(self):
        for m in self.params["models"].values():
            aucs = {float(k): v for k, v in m["r_auc"].items()}
            best = max(aucs.values())
            self.assertEqual(m["r"], min(r for r, a in aucs.items() if a >= best - 1e-12))

    def test_a_changed_value_is_caught(self):
        tampered = copy.deepcopy(self.params)
        tampered["models"][MODELS[0]]["temperature"] *= 1.01
        path = self.tmp / "tampered.json"
        path.write_text(json.dumps(tampered))
        with self.assertRaises(ValueError):
            load_params(path)
        # ...but re-indenting, as pasting through a chat does, is not a change.
        path.write_text(json.dumps(self.params, indent=4))
        self.assertEqual(load_params(path)["digest"], self.params["digest"])

    # ---------------------------------------------------------------- rehearsal

    def test_the_rehearsal_writes_every_table(self):
        out = self.tmp / "analysis_val"
        self.assertEqual(self.rehearse(out), 0)
        res = json.loads((out / "results.json").read_text())
        self.assertTrue(res["rehearsal"])
        self.assertFalse(res["unblinded"])
        curves = np.load(out / "curves.npz")
        for name, r in res["models"].items():
            arms = r["arms"]
            # calibration never changes y-hat: at full coverage every arm scores the same
            for arm in analyse.ARMS:
                self.assertAlmostEqual(float(curves[f"{name}__{arm}"][-1]), r["accuracy"], places=5)
                lo, hi = arms[arm]["auc_interval"]
                self.assertLessEqual(lo, hi)
            self.assertAlmostEqual(arms["none"]["auc"], r["accuracy"], places=12)
            self.assertEqual(arms["none"]["distinct_scores"], 1)
            self.assertAlmostEqual(sum(v["share"] for v in r["F5"].values()), 1.0)
            self.assertIn("em", r["calibration"])
            self.assertIn("D4_ece_by_sample_size", r["calibration"])
            counts = r["faithfulness"]["counts"]
            self.assertEqual(counts["no_lesion"] + counts["undetermined"] + counts["determined"],
                             res["n"])
            self.assertGreater(counts["undetermined"], 0)
        claims = res["claims_on_this_dataset"]
        self.assertEqual(set(claims), {"eyepacs_full", "eyepacs_ddr_full", "H3_qwk_gain"})
        self.assertEqual(len(claims["eyepacs_full"]["auc_gain"]["per_seed"]), 3)
        ev = res["evidence"]
        self.assertEqual(np.array(ev["confusion_true_by_evidence"]).sum(), res["n"])

        verdict_dir = self.tmp / "verdicts"
        self.assertEqual(analyse.main(["verdicts", "--in-domain", str(out / "results.json"),
                                       "--external", str(out / "results.json"),
                                       "--out", str(verdict_dir)]), 0)
        verdicts = json.loads((verdict_dir / "verdicts.json").read_text())
        self.assertTrue(verdicts["rehearsal"])
        self.assertEqual(set(verdicts), {"rehearsal", "in_domain", "external", "primary",
                                         "replication", "H3"})

    def test_the_rehearsal_is_deterministic(self):
        a, b = self.tmp / "det_a", self.tmp / "det_b"
        self.assertEqual(self.rehearse(a, role="in_domain"), 0)
        self.assertEqual(self.rehearse(b, role="in_domain"), 0)
        ra, rb = (json.loads((d / "results.json").read_text()) for d in (a, b))
        for r in (ra, rb):
            r.pop("created_at")
            r.pop("pass")
        self.assertEqual(ra, rb)

    # ---------------------------------------------------------------- guards

    def run_locked(self, *extra, fitted=None):
        with mock.patch.object(analyse, "labels_for",
                               side_effect=AssertionError("a locked label was read")):
            return analyse.main(["dataset", "--pass-dir", str(self.tmp / "locked" / "test"),
                                 "--labels", str(self.full_manifest), "--split", "test",
                                 "--fitted", str(fitted or self.fitted / "fitted_params.json"),
                                 "--ood-dir", str(self.fitted), "--name", "eyepacs_test",
                                 "--role", "in_domain", "--out", str(self.tmp / "locked_out"),
                                 *extra])

    def test_locked_data_is_refused_before_any_label_is_read(self):
        self.assertEqual(self.run_locked(), analyse.EXIT_REFUSED)

    def test_unblinding_needs_committed_parameters(self):
        # fitted_params.json in a temp dir is not tracked by any repository
        self.assertEqual(self.run_locked("--unblind"), analyse.EXIT_REFUSED)
        # a real repository, committed file: the guard lets it through to the label join
        repo = self.tmp / "repo"
        repo.mkdir()
        shutil.copy(self.fitted / "fitted_params.json", repo / "fitted_params.json")
        git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
        subprocess.run(git[:3] + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "fitted_params.json"], check=True)
        subprocess.run(git + ["commit", "-qm", "params"], check=True)
        with self.assertRaises(AssertionError):          # it got as far as reading labels
            self.run_locked("--unblind", fitted=repo / "fitted_params.json")
        # ...and an edit after the commit is refused again
        text = (repo / "fitted_params.json").read_text()
        (repo / "fitted_params.json").write_text(text.replace('"plan"', '"plan" ', 1))
        self.assertEqual(self.run_locked("--unblind", fitted=repo / "fitted_params.json"),
                         analyse.EXIT_REFUSED)

    def test_a_pass_from_different_checkpoints_is_refused(self):
        other = self.tmp / "other_val"
        shutil.copytree(self.pred / "val", other)
        run = json.loads((other / "run.json").read_text())
        run["grading_checkpoints"][MODELS[0]]["sha256"] = "f" * 64
        (other / "run.json").write_text(json.dumps(run))
        code = analyse.main(["dataset", "--pass-dir", str(other), "--labels",
                             str(self.full_manifest), "--split", "val", "--fitted",
                             str(self.fitted / "fitted_params.json"), "--ood-dir",
                             str(self.fitted), "--name", "val", "--role", "in_domain",
                             "--out", str(self.tmp / "other_out"), "--rehearsal",
                             "--resamples", "5"])
        self.assertEqual(code, 1)

    def test_fewer_resamples_only_in_a_rehearsal(self):
        code = analyse.main(["dataset", "--pass-dir", str(self.pred / "val"), "--labels",
                             str(self.full_manifest), "--split", "val", "--fitted",
                             str(self.fitted / "fitted_params.json"), "--ood-dir",
                             str(self.fitted), "--name", "val", "--role", "in_domain",
                             "--out", str(self.tmp / "few"), "--resamples", "10"])
        self.assertEqual(code, analyse.EXIT_REFUSED)

    def test_fit_refuses_a_reference_drawn_differently(self):
        bad = self.tmp / "bad_internal"
        shutil.copytree(self.pred, bad)
        run = json.loads((bad / "reference_eyepacs_full" / "run.json").read_text())
        run["sample_seed"] = 1
        (bad / "reference_eyepacs_full" / "run.json").write_text(json.dumps(run))
        with self.assertRaises(SystemExit):
            fit_params.main(["--internal", str(bad), "--manifest-full", str(self.full_manifest),
                             "--manifest-ddr", str(self.ddr_manifest), "--out",
                             str(self.tmp / "bad_fit")])


class Params(unittest.TestCase):

    def test_non_finite_values_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                write_params({"x": float("nan")}, Path(d) / "p.json")

    def test_probabilities_used_downstream_are_the_training_ones(self):
        z = np.array([[2.0, 1.0, -1.0, -3.0]])
        self.assertAlmostEqual(coral_probs(z).sum(), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
