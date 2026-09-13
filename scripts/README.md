# scripts/ — build utilities

Not yet implemented. Each script's contract is specified below so it can be written
(by you or by Claude) without re-deriving the design.

Run order: `build_cache.py` → `prepare_manifest.py` → `build_variants.py`.

---

## `build_cache.py` — Phase 1

**The most important script in the project.** Turns variable-size JPEGs into a fixed
512 px cache. Without it a training run takes ~4 h; with it, ~1.5 h.

```
--source-root   /kaggle/input/<dataset>
--dataset       EyePACS | DDR | IDRiD | APTOS | Messidor2
--output-root   /kaggle/working/cache512
--size          512
--quality       90
```

Per image: retinal-field crop (threshold low intensity, bounding box of the retinal
circle) → resize shortest side to 512 → centre-crop 512×512 → CLAHE on the LAB
L channel (clipLimit 2.0, tiles 8×8) → save JPEG q90.

Emits `cache_report.json`: per-dataset counts in and out, crop failures, mean output
size. **A0 checks this.** Run CPU-only — it consumes no GPU quota.

Masks, where present, go through the identical geometric transform with
nearest-neighbour interpolation and no CLAHE.

## `prepare_manifest.py` — Phase 2

Converts each source's native labelling into the manifest contract in
`docs/05_dataset_card.md` § Manifest contract.

```
--dataset EyePACS  --folder-root <cache>/eyepacs        # ImageFolder-style
--dataset DDR      --labels DR_grading.csv --image-dir <cache>/ddr
--dataset APTOS    --labels train.csv --id-col id_code --grade-col diagnosis
--dataset Messidor2 --labels messidor2_grades.csv       # join images + grades datasets
--dataset IDRiD    --labels <grading csv> --coords <od/fovea csv>
```

**The one that bites — EyePACS patient IDs:**

```python
m = re.match(r"^(\d+)_(left|right)$", Path(p).stem)
patient_id = f"EyePACS::{m.group(1)}"
eye = m.group(2)
```

Every dataset prefixes its patient IDs (`EyePACS::`, `DDR::`, …) so merged manifests
cannot collide. Where no patient ID exists, fall back to the image stem **and print a
warning** — the fallback treats each image as its own patient, which leaks for any
dataset with two eyes per person.

DDR grade-5 rows are routed to `quality_manifest.csv`, not the grading manifest.

## `build_variants.py` — Phase 2

Builds the three development variants plus the locked externals.

```
eyepacs_full           all unique EyePACS, official split
eyepacs_balanced_1000  up to 1,000 unique images per grade  (training ablation only)
eyepacs_ddr_full       EyePACS + DDR merged
aptos_external         LOCKED
messidor2_external     LOCKED
```

Split policy: use the **official** EyePACS competition split as the headline
(35,126 train / 53,576 test — already patient-disjoint). Carve validation (10%) and
calibration (5%) out of *train only*, patient-grouped.

**Must assert patient-disjointness across all splits and fail loudly.** Do not
disable this assertion — see `docs/02_research_protocol.md` Rule 3.

Balanced sampling is without replacement; shortfalls on rare grades are recorded in
`manifests/dataset_plan.json`, never fabricated. Validation, calibration and test rows
are never duplicated.

## `train_grading.py` / `train_evidence.py` / `train_geometry.py` — Phases 3–4

Per `docs/03_model_architecture.md` §§ M1, M2a, M2b. Each takes a config from
`configs/`, writes checkpoints and `metrics.json` to
`results/<stage>/<experiment_id>/`, and checkpoints **every epoch** so a killed Kaggle
session can resume rather than restart.

## `evaluate.py` — Phases 6–7

Runs a trained model over a manifest and emits the decision record in
`docs/03_model_architecture.md` § M4, plus aggregate metrics.

**Guard rail:** refuse to run against a manifest marked `locked: true` unless
`--unblind` is passed *and* `preregistration/frozen_config.yaml` exists. Cheap
insurance against accidentally burning the external sets.
