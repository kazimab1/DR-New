"""Phase 8 (scripts/phase8.py): the exploratory error analysis after the unblinding.

Built on test_fit_analyse's synthetic world: fit_params.py and analyse.py run first,
exactly as in the real chain, and Phase 8 must reproduce analyse.py's published AUCs
before it prints anything (P8.0).
"""

import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import analyse  # noqa: E402
import fit_params  # noqa: E402
import phase8  # noqa: E402
from test_fit_analyse import MODELS, REFERENCE, World  # noqa: E402
from verify_dr.triage import ood as O  # noqa: E402
from verify_dr.triage import selective as S  # noqa: E402


def quiet(fn, *args, **kwargs):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = fn(*args, **kwargs)
    return code, buf.getvalue()


class Phase8(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="phase8_"))
        cls.patch = mock.patch.object(O, "REFERENCE_SIZE", REFERENCE)
        cls.patch.start()
        worlds = {"train": World("train", REFERENCE, 1), "calibration": World("calibration", 900, 2),
                  "val": World("val", 1200, 3), "test": World("test", 700, 4)}
        pred = cls.tmp / "predictions"
        full = [m for m in MODELS if "_ddr_" not in m]
        ddr = [m for m in MODELS if "_ddr_" in m]
        worlds["train"].write_pass(pred / "reference_eyepacs_full", full, True, sample=REFERENCE)
        mixed = World("train", REFERENCE, 6)
        mixed.paths = [p if i % 5 else p.replace("/eyepacs/", "/ddr/")
                       for i, p in enumerate(mixed.paths)]
        mixed.ids = [f"{'DDR' if i % 5 == 0 else 'EyePACS'}::{Path(p).stem}"
                     for i, p in enumerate(mixed.paths)]
        mixed.write_pass(pred / "reference_eyepacs_ddr_full", ddr, True, sample=REFERENCE)
        for split in ("calibration", "val"):
            worlds[split].write_pass(pred / split, MODELS)
        worlds["test"].write_pass(cls.tmp / "locked" / "eyepacs_test", MODELS, locked=True)
        manifest = pd.concat([w.manifest_rows() for w in worlds.values()], ignore_index=True)
        cls.manifest = cls.tmp / "eyepacs_full.csv"
        manifest.to_csv(cls.manifest, index=False)
        m = mixed.manifest_rows()
        m["dataset"] = ["DDR" if i % 5 == 0 else "EyePACS" for i in range(REFERENCE)]
        ddr_manifest = cls.tmp / "eyepacs_ddr_full.csv"
        pd.concat([m, manifest[manifest["split"] != "train"]]).to_csv(ddr_manifest, index=False)
        cls.pred, cls.worlds = pred, worlds

        cls.fitted = cls.tmp / "fitted"
        assert fit_params.main(["--internal", str(pred), "--manifest-full", str(cls.manifest),
                                "--manifest-ddr", str(ddr_manifest), "--out", str(cls.fitted)]) == 0

        # the rehearsal on val: an external role, so D4 and EM are exercised too
        cls.val_results = cls.tmp / "analysis" / "val"
        code, _ = quiet(analyse.main, [
            "dataset", "--pass-dir", str(pred / "val"), "--labels", str(cls.manifest),
            "--split", "val", "--fitted", str(cls.fitted / "fitted_params.json"),
            "--ood-dir", str(cls.fitted), "--name", "val", "--role", "external",
            "--out", str(cls.val_results), "--rehearsal", "--resamples", "30"])
        assert code == 0

        # a real unblinding of the locked split: committed parameters, full resamples
        cls.repo = cls.tmp / "repo"
        cls.repo.mkdir()
        shutil.copy(cls.fitted / "fitted_params.json", cls.repo / "fitted_params.json")
        git = ["git", "-C", str(cls.repo), "-c", "user.email=t@t", "-c", "user.name=t"]
        subprocess.run(git[:3] + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "fitted_params.json"], check=True)
        subprocess.run(git + ["commit", "-qm", "params"], check=True)
        cls.test_results = cls.tmp / "analysis" / "eyepacs_test"
        code, _ = quiet(analyse.main, [
            "dataset", "--pass-dir", str(cls.tmp / "locked" / "eyepacs_test"),
            "--labels", str(cls.manifest), "--split", "test",
            "--fitted", str(cls.repo / "fitted_params.json"), "--ood-dir", str(cls.fitted),
            "--name", "eyepacs_test", "--role", "in_domain", "--unblind",
            "--out", str(cls.test_results)])
        assert code == 0

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def errors(self, pass_dir, results, split, out, fitted=None):
        return quiet(phase8.main, [
            "errors", "--pass-dir", str(pass_dir), "--labels", str(self.manifest),
            "--split", split, "--fitted", str(fitted or self.repo / "fitted_params.json"),
            "--ood-dir", str(self.fitted), "--results", str(results), "--out", str(out)])

    # ---------------------------------------------------------------- P8.0

    def test_the_gate_passes_on_the_real_chain_and_every_table_is_written(self):
        code, text = self.errors(self.pred / "val", self.val_results / "results.json", "val",
                                 self.tmp / "p8_val")
        self.assertEqual(code, 0, text)
        self.assertIn("P8.0 reproduction gate passed: 6 models x 2 analyses x 5 arms", text)
        self.assertIn("REHEARSAL -- not a result.", text)
        out = json.loads((self.tmp / "p8_val" / "phase8.json").read_text())
        self.assertTrue(out["exploratory"])
        for analysis in ("registered", "amended"):
            s = out["summary_primary"][analysis]
            for cov in ("@80", "@90"):
                o = s["overlap"][cov]["exact_grade"]
                parts = sum(o[k]["mean"] for k in ("confidence_only", "disagreement_only",
                                                   "both", "neither"))
                self.assertAlmostEqual(parts, 1.0, places=9)
            for m in out["per_model"][analysis].values():
                t = m["taxonomy"]["silent_errors"]
                self.assertAlmostEqual(t["m3_abstains_yhat_ge3"] + t["m3_agrees_with_wrong_grade"]
                                       + t["visible"], 1.0, places=9)

    def test_the_locked_unblinding_is_accepted_and_candidates_follow_the_rule(self):
        code, text = self.errors(self.tmp / "locked" / "eyepacs_test",
                                 self.test_results / "results.json", "test", self.tmp / "p8_test")
        self.assertEqual(code, 0, text)
        self.assertNotIn("REHEARSAL", text)
        cands = pd.read_csv(self.tmp / "p8_test" / "candidates.csv")
        self.assertTrue(set(cands["category"]) <= set(phase8.CATEGORIES))
        for category, group in cands.groupby("category"):
            self.assertLessEqual(len(group), phase8.CANDIDATES_PER_CATEGORY)
            conf = group["m1_confidence"].to_numpy()
            self.assertTrue(np.all(np.diff(conf) <= 1e-12), "most confident first")
        a = cands[cands["category"].str.startswith("A_")]
        self.assertTrue((a["m1_grade"] == 0).all() and (a["true_grade"] >= 1).all())
        self.assertTrue((a["n_microaneurysm"] > 0).all())
        b = cands[cands["category"].str.startswith("B_")]
        self.assertTrue(((b["true_grade"] == 0) & (b["evidence_amended"] == 2)).all())
        out = json.loads((self.tmp / "p8_test" / "phase8.json").read_text())
        self.assertEqual(set(out["figure_candidates_qualifying"]), set(phase8.CATEGORIES))

    def test_no_qualifying_candidate_is_a_finding_not_a_failure(self):
        empty = self.tmp / "empty_candidates.csv"
        pd.DataFrame(columns=["category", "image_id", "dataset", "image_path"]).to_csv(empty, index=False)
        args = SimpleNamespace(candidates=[empty], cache_root=None, evidence_checkpoint=None,
                               per_category=8, out=self.tmp / "figs_empty")
        code, text = quiet(phase8.run_figures, args, segmenter=object(), size=64)
        self.assertEqual(code, 0)
        self.assertIn("no figure candidates qualified", text)

    def test_a_changed_auc_fails_the_gate(self):
        res = json.loads((self.val_results / "results.json").read_text())
        res["amended"]["models"][MODELS[0]]["arms"]["disagreement"]["auc"] += 1e-9
        bad = self.tmp / "bad_results.json"
        bad.write_text(json.dumps(res))
        code, _ = self.errors(self.pred / "val", bad, "val", self.tmp / "p8_bad")
        self.assertEqual(code, phase8.EXIT_REFUSED)

    def test_results_of_another_pass_are_refused(self):
        code, _ = self.errors(self.pred / "calibration", self.val_results / "results.json",
                              "calibration", self.tmp / "p8_other")
        self.assertEqual(code, phase8.EXIT_REFUSED)

    def test_a_locked_pass_needs_the_unblinding_on_record(self):
        res = json.loads((self.test_results / "results.json").read_text())
        res["unblinded"] = False
        bad = self.tmp / "not_unblinded.json"
        bad.write_text(json.dumps(res))
        code, _ = self.errors(self.tmp / "locked" / "eyepacs_test", bad, "test",
                              self.tmp / "p8_nb")
        self.assertEqual(code, phase8.EXIT_REFUSED)

    # ---------------------------------------------------------------- P8.1

    def test_secondary_transcribes_every_outcome(self):
        code, text = quiet(phase8.main, ["secondary", "--results",
                                         str(self.val_results / "results.json"),
                                         str(self.test_results / "results.json"),
                                         "--out", str(self.tmp / "p8_sec")])
        self.assertEqual(code, 0)
        for marker in ("acc@80", "F3 (eyepacs_full)", "F4 (eyepacs_full)", "F5 (eyepacs_full)",
                       "D4 (eyepacs_full)", "E1 median", "E3 grade 4", "--- AMENDED ---"):
            self.assertIn(marker, text)
        tables = json.loads((self.tmp / "p8_sec" / "secondary.json").read_text())
        res = json.loads((self.val_results / "results.json").read_text())
        mean80 = np.mean([res["models"][f"H1_eyepacs_full_s{s}"]["arms"]["confidence"]["acc@80"]
                          for s in (42, 43, 44)])
        got = tables[0]["analyses"]["registered"]["eyepacs_full"]["arms"]["confidence"]["acc@80"]
        self.assertAlmostEqual(got["mean"], mean80, places=12)
        self.assertNotIn("D4", tables[1]["analyses"]["registered"]["eyepacs_full"])  # in-domain

    # ---------------------------------------------------------------- the pieces

    def test_overlap_is_an_expectation_over_ties(self):
        err = np.array([1, 1, 0, 0, 0], dtype=float)
        tied = S.dense_rank(np.zeros(5))
        q = phase8.deferral_probability(tied, 0.8)          # k = 4 accepted of 5 tied
        np.testing.assert_allclose(q, 0.2)
        o = phase8.overlap(err, q, q)
        self.assertAlmostEqual(o["both"], 0.04)
        self.assertAlmostEqual(o["neither"], 0.64)
        self.assertAlmostEqual(o["confidence_only"] + o["disagreement_only"], 0.32)
        self.assertTrue(np.isnan(phase8.overlap(np.zeros(5), q, q)["both"]))

    def test_auroc_matches_sklearn_with_ties(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(0)
        score = rng.integers(0, 4, 400).astype(float)
        positive = rng.random(400) < 0.3 + 0.1 * score
        self.assertAlmostEqual(phase8.auroc(score, positive), roc_auc_score(positive, score),
                               places=12)
        self.assertTrue(np.isnan(phase8.auroc(score, np.zeros(400, bool))))

    def test_figures_draw_every_candidate(self):
        from PIL import Image

        cache = self.tmp / "cache"
        (cache / "eyepacs").mkdir(parents=True)
        rows = []
        for i in range(3):
            path = cache / "eyepacs" / f"img{i}.png"
            rgb = np.zeros((64, 64, 3), np.uint8)
            rgb[8:56, 8:56] = (180, 90, 40)
            Image.fromarray(rgb).save(path)
            rows.append({"category": list(phase8.CATEGORIES)[i % 2], "image_id": f"EyePACS::img{i}",
                         "dataset": "EyePACS", "image_path": str(path), "true_grade": 1,
                         "m1_grade": 0, "m1_confidence": 0.95, "evidence_registered": 2,
                         "evidence_amended": 1, "rule_amended": "R2",
                         **{f"n_{t}": 1 for t in phase8.LESION_NAMES},
                         **{f"area_{t}": 9 for t in phase8.LESION_NAMES}})
        csv = self.tmp / "cands.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)

        class Blob(torch.nn.Module):             # one 3x3 microaneurysm, nothing else
            def forward(self, x):
                out = torch.full((x.shape[0], 4, x.shape[2], x.shape[3]), -5.0)
                out[:, 0, 20:23, 30:33] = 5.0
                return out

        args = SimpleNamespace(candidates=[csv], cache_root=None, evidence_checkpoint=None,
                               per_category=8, out=self.tmp / "figs")
        code, text = quiet(phase8.run_figures, args, segmenter=Blob(), size=64)
        self.assertEqual(code, 0, text)
        sheets = sorted(p.name for p in (self.tmp / "figs").glob("[A-C]_*.png") if "__" not in p.name)
        self.assertEqual(len(sheets), 2)
        masks = phase8.lesion_masks(torch.sigmoid(Blob()(torch.zeros(1, 3, 64, 64)))[0].numpy(),
                                    0.5, 4)
        self.assertEqual(int(masks["microaneurysm"].sum()), 9)
        self.assertFalse(masks["haemorrhage"].any())

    def test_the_audit_reads_images_and_never_a_grade(self):
        from PIL import Image

        root = self.tmp / "audit_src"
        manifests = root / "manifests"
        manifests.mkdir(parents=True)
        specs = {"eyepacs_full.csv": ("EyePACS", "test", (160, 80, 40)),
                 "aptos_external.csv": ("APTOS", "test", (150, 70, 30)),
                 "messidor2_external.csv": ("Messidor2", "test", (128, 128, 128)),
                 "eyepacs_ddr_full.csv": ("DDR", "train", (170, 90, 50))}
        for file, (dataset, split, colour) in specs.items():
            paths = []
            for i in range(4):
                p = root / dataset / f"{i}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                rgb = np.zeros((48, 48, 3), np.uint8)
                rgb[6:42, 6:42] = colour
                Image.fromarray(rgb).save(p)
                paths.append(str(p))
            # a grade column that would crash any numeric use: it must never be read
            pd.DataFrame({"image_path": paths, "dataset": dataset, "split": split,
                          "grade": ["LOCKED"] * 4}).to_csv(manifests / file, index=False)
        raw = root / "raw"
        raw.mkdir()
        Image.fromarray(np.full((40, 60, 3), 128, np.uint8)).save(raw / "IM001.png")
        args = SimpleNamespace(manifest_dir=manifests, cache_root=None,
                               locked=self.tmp / "locked", raw_root=[raw], per_source=4,
                               raw_sample=3, sheet_per_row=4, seed=0, out=self.tmp / "audit")
        with mock.patch.object(pd, "read_csv", wraps=pd.read_csv) as spy:
            code, text = quiet(phase8.run_audit, args)
        self.assertEqual(code, 0, text)
        for call in spy.call_args_list:
            if "manifests" in str(call.args[0]):
                self.assertIn("usecols", call.kwargs)
        audit = json.loads((self.tmp / "audit" / "audit.json").read_text())
        self.assertEqual(set(audit["cached"]), {"EyePACS test", "APTOS", "Messidor-2", "DDR"})
        self.assertGreater(audit["cached"]["Messidor-2"]["near_gray_share"]["median"], 0.9)
        self.assertEqual(len(audit["raw_messidor2"]), 1)
        self.assertIn("eyepacs_test", audit["detection"])
        self.assertTrue((self.tmp / "audit" / "contact_sheet.png").exists())

        # A raw root inside the cache is the cache's own copy, not the source mirror:
        # the first real run measured exactly that. It must be skipped, not measured.
        inside = SimpleNamespace(**{**vars(args), "cache_root": [root], "raw_root": [root / "Messidor2"],
                                    "out": self.tmp / "audit_inside"})
        code, text = quiet(phase8.run_audit, inside)
        self.assertEqual(code, 0, text)
        self.assertIn("inside the cache", text)
        self.assertEqual(json.loads((self.tmp / "audit_inside" / "audit.json").read_text())
                         ["raw_messidor2"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
