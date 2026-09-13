# scripts/ — build utilities

Not yet implemented. Each script's contract is specified below so it can be written
(by you or by Claude) without re-deriving the design.

Run order: `build_cache.py` → `prepare_manifest.py` → `build_variants.py`.

---

## `build_cache.py` — Phase 1 ✅ implemented

**The most important script in the project.** Turns variable-size JPEGs into a fixed
512 px cache. Without it a training run takes ~4 h; with it, ~1.5 h.

```
--source-root   /kaggle/input/<dataset>     scanned recursively
--dataset       EyePACS | DDR | IDRiD | APTOS | Messidor2
--output-root   /kaggle/working/cache512
--size          512
--quality       90
--fit           pad (default) | crop
--clahe / --no-clahe          --clahe-clip 2.0   --clahe-tiles 8
--mask          CHANNEL=DIR   repeatable; microaneurysm, haemorrhage,
                              hard_exudate, soft_exudate
--workers N   --limit N   --overwrite   --dry-run   --contact-sheet N   --seed
```

Per image: retinal-field crop → square fit → resize to `--size` → CLAHE on the LAB
L channel → JPEG at `--quality`. Output mirrors the source tree under
`<output-root>/<dataset>/images/`, so class-folder layouts and filenames (and
therefore EyePACS patient IDs) survive into Phase 2.

Emits `<output-root>/<dataset>/cache_report.json`: counts in and out, crop fallback
rate, mean output size, per-channel mask tallies, recorded failures. **A0 checks
this** — the gate is a fallback rate ≤ 0.005, and the script flags it in the console
when exceeded. `--contact-sheet N` also writes a grid of N random crops for A0's
visual audit. Run CPU-only; it consumes no GPU quota.

Masks take the *image's* geometry — computed from the image, never from the mask,
since masks are mostly black and would crop to nothing — with nearest-neighbour
interpolation, no CLAHE, written as lossless PNG under `<dataset>/masks/<channel>/`.

Resumable: existing non-empty outputs are skipped unless `--overwrite`. Writes are
atomic, so an interrupted run cannot leave a truncated file that the next run would
mistake for finished work. Per-image failures are recorded and skipped, never fatal.

### Two deliberate deviations from the original contract

**1. `--fit pad` is the default, not centre-crop.** Centre-cropping a wide retinal
bounding box discards the nasal and temporal periphery — exactly the regions M3 needs
for the quadrant-based 4-2-1 haemorrhage rule. Padding to square keeps the whole
field at the same aspect ratio. `--fit crop` reproduces the original behaviour if you
want to ablate it.

**2. The crop threshold is measured, not fixed.** The first implementation used
`max(7, mean × 0.1)`, which fundus JPEG compression noise defeats: the black surround
routinely reaches 15–20/255, so ~58% of surround pixels cleared the threshold, the
box expanded to the full frame, and the crop silently did nothing *while still
reporting success*. The threshold is now set above the surround level measured from
the corner patches, on a downsampled median-filtered copy. A box within 2% of the
full frame is reported as a fallback rather than counted as a crop, so the A0 rate
reflects reality. Side effect: ~3.8× faster, since thresholding happens at ≤512 px.

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
