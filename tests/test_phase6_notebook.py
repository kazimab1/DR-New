"""Exercise the REAL cell text from 06_final_training.ipynb against a fixture.

Deliberately not a re-implementation of the planning arithmetic. Phase 4 lost a
week to tests that restated the code's own wrong assumption and therefore agreed
with it; these exec the notebook's actual source.
"""
import json, shutil, tempfile, unittest
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "06_final_training.ipynb"
CELLS = [c for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]

def cell_with(token):
    hits = [c for c in CELLS if token in "".join(c["source"])]
    assert len(hits) == 1, f"{token}: expected 1 cell, found {len(hits)}"
    return "".join(hits[0]["source"])

PLAN_SRC, RUN_SRC = cell_with("VARIANTS ="), cell_with("def spent_h")
SUM_SRC = cell_with("per_class_f1 is keyed")


class FakeClock:
    """Stands in for the time module. Training is instant here; billing is not."""

    def __init__(self):
        self.now = 1_000_000.0

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Fixture(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="phase6_")); shutil.rmtree(self.tmp, ignore_errors=True)
        self.man = self.tmp / "manifests"; self.man.mkdir(parents=True)
        self.res = self.tmp / "results";   self.res.mkdir(parents=True)
        # 16 560 train rows -> 1.00 h at 10 epochs / 46 img/s
        for v in ("eyepacs_full", "eyepacs_ddr_full"):
            rows = ["image_path,grade,split"]
            rows += [f"/x/{i}.jpg,0,train" for i in range(16560)]
            rows += [f"/y/{i}.jpg,0,val" for i in range(10)]
            (self.man / f"{v}.csv").write_text("\n".join(rows))
        self.calls = []

    def env(self, budget, hours_per_run=1.0, session_limit=999.0):
        clock = FakeClock()

        def fake_run(cmd):
            self.calls.append(cmd)
            clock.advance(hours_per_run * 3600)

        g = {"MANIFEST_DIR": self.man, "RESULTS": self.res,
             "THROUGHPUT_IMG_S": 46.0, "GPU_HOURS_LEFT": budget,
             "SAFETY_MARGIN_H": 0.0, "SESSION_LIMIT_H": session_limit,
             "time": clock, "SESSION_START": clock.time(),
             "run": fake_run,
             "q": lambda x: str(x), "REPO_DIR": Path("/repo"),
             "CACHE_FLAGS": "/cache", "print": lambda *a, **k: None}
        exec(PLAN_SRC, g)
        return g

    def finish(self, name, epochs=10, **best):
        d = self.res / name; d.mkdir(parents=True, exist_ok=True)
        (d / "checkpoint.pt").write_text("x")
        (d / "history.json").write_text(json.dumps([{"epoch": i} for i in range(epochs)]))
        b = {"qwk": .68, "macro_f1": .4, "mae": .3, "distinct_predictions": 5,
             "per_class_f1": {str(i): .5 for i in range(5)},
             "per_class_recall": {str(i): .5 for i in range(5)}}
        b.update(best)
        (d / "metrics.json").write_text(json.dumps(
            {"best_val": b, "epochs_run": epochs, "minutes": 60.0}))

    def partial(self, name, epochs):
        d = self.res / name; d.mkdir(parents=True, exist_ok=True)
        (d / "checkpoint.pt").write_text("x")
        (d / "history.json").write_text(json.dumps([{"epoch": i} for i in range(epochs)]))

    # ---------------------------------------------------------------- tests
    def test_cost_matches_hand_arithmetic(self):
        g = self.env(99)
        self.assertAlmostEqual(g["rows"][0]["cost h"], 1.0, places=2)

    def test_budget_stops_before_a_run_it_cannot_finish(self):
        g = self.env(2.5)                      # room for exactly two 1.0 h runs
        exec(RUN_SRC, g)
        self.assertEqual(len(self.calls), 2, "should start exactly two runs")
        self.assertEqual(len(g["remaining"]), 4)
        self.assertTrue(g["remaining"][0].endswith("s44"))

    def test_completed_runs_are_skipped_not_retrained(self):
        self.finish("H1_eyepacs_full_s42")
        g = self.env(99)
        exec(RUN_SRC, g)
        self.assertEqual(len(self.calls), 5)
        self.assertNotIn("s42", " ".join(c for c in self.calls if "eyepacs_full_s" in c
                                          and "ddr" not in c))
        self.assertIn("H1_eyepacs_full_s42", g["completed"])

    def test_partial_run_is_priced_only_for_its_remaining_epochs(self):
        self.partial("H1_eyepacs_full_s42", epochs=7)
        g = self.env(99)
        row = g["rows"][0]
        self.assertIn("partial (7/10", row["state"])
        self.assertAlmostEqual(row["left h"], 0.30, places=2)

    def test_run_order_is_grouped_by_variant(self):
        g = self.env(99)
        self.assertEqual([r["run"] for r in g["rows"]], [
            "H1_eyepacs_full_s42", "H1_eyepacs_full_s43", "H1_eyepacs_full_s44",
            "H1_eyepacs_ddr_full_s42", "H1_eyepacs_ddr_full_s43",
            "H1_eyepacs_ddr_full_s44"])

    def test_frozen_recipe_is_stated_not_defaulted(self):
        """The script defaults to 12 epochs; Stage B selected 10."""
        g = self.env(99); exec(RUN_SRC, g)
        cmd = self.calls[0]
        for flag in ("--epochs 10", "--head ordinal_focal", "--image-size 512",
                     "--sampler stratified_exposure", "--focal-gamma 2.0",
                     "--batch-size 32", "--patience 3", "--resume", "--seed 42"):
            self.assertIn(flag, cmd, f"missing {flag}")

    def test_every_run_gets_its_own_seed(self):
        g = self.env(99); exec(RUN_SRC, g)
        seeds = [c.split("--seed ")[1].split()[0] for c in self.calls]
        self.assertEqual(seeds, ["42", "43", "44"] * 2)

    def test_missing_manifest_raises_rather_than_pricing_at_zero(self):
        (self.man / "eyepacs_ddr_full.csv").unlink()
        with self.assertRaises(RuntimeError) as cm:
            self.env(99)
        self.assertIn("eyepacs_ddr_full.csv", str(cm.exception))

    def test_summary_reads_grade1_f1_by_key(self):
        """per_class_f1 is keyed by string grade; indexing it by position is wrong."""
        self.finish("H1_eyepacs_full_s42",
                    per_class_f1={"0": .9, "1": .123, "2": .4, "3": .3, "4": .2})
        g = self.env(99)
        out = []
        g["print"] = lambda *a, **k: out.append(" ".join(str(x) for x in a))
        exec(SUM_SRC, g)
        self.assertIn("0.123", "\n".join(out))

    def test_a_slow_run_shrinks_what_follows_it(self):
        """The estimate is 46 img/s. If reality is half that, the guard must
        notice after run 1 rather than confidently starting two more."""
        g = self.env(4.5, hours_per_run=2.0)
        exec(RUN_SRC, g)
        self.assertEqual(len(self.calls), 2,
                         "4.5 h at 2 h actual per run fits two, not four")

    def test_time_before_section_8_is_billed(self):
        """Extraction and the carry-forward copy come out of the same quota."""
        g = self.env(2.5)
        g["time"].advance(1.2 * 3600)      # a long cache extraction
        exec(RUN_SRC, g)
        self.assertEqual(len(self.calls), 1,
                         "1.2 h already billed leaves room for one 1.0 h run")

    def test_the_session_wall_binds_even_with_quota_to_spare(self):
        """A full 30 h quota does not mean 30 h in one sitting. A session killed
        mid-run may never save its output, losing the whole session."""
        g = self.env(30.0, session_limit=2.5)
        exec(RUN_SRC, g)
        self.assertEqual(len(self.calls), 2,
                         "session wall of 2.5 h fits two 1.0 h runs, not six")

    def test_quota_still_binds_when_it_is_the_smaller_limit(self):
        g = self.env(2.5, session_limit=999.0)
        exec(RUN_SRC, g)
        self.assertEqual(len(self.calls), 2)

    def test_collapsed_run_is_flagged(self):
        self.finish("H1_eyepacs_full_s42", distinct_predictions=1)
        g = self.env(99)
        out = []
        g["print"] = lambda *a, **k: out.append(" ".join(str(x) for x in a))
        exec(SUM_SRC, g)
        self.assertIn("WARNING", "\n".join(out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
