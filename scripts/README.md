# scripts/ — build utilities

`build_cache.py` and `prepare_manifest.py` are implemented. The remaining contracts are
specified below so they can be written without re-deriving the design.

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

## `prepare_manifest.py` — Phase 2 ✅ implemented

Converts each source's native labelling into the manifest contract in
`docs/05_dataset_card.md` § Manifest contract, reading the 512 px cache.

```
--dataset {EyePACS,DDR,IDRiD,APTOS,Messidor2}   --cache-root <cache>   --output FILE
--folder-labels                 grade from a 0-4 class-folder component (EyePACS)
--labels FILE                   label file (.txt/.csv/.xlsx); repeatable, one per split
--id-col / --grade-col          override column guessing
--coords FILE                   IDRiD Part C centre table; repeatable (fovea and OD)
--coords-source-dir DIR         original images the coordinates refer to
--size / --fit / --tol-scale    must match build_cache.py
--quality-output FILE           where DDR grade-5 ungradable rows go
--dry-run
```

Emits `<output>.report.json` alongside the CSV: row and patient counts,
images-per-patient, grade distribution, ungradable count, patient-ID fallbacks,
unmatched labels in both directions, per-channel mask coverage.

### It does not assign `split`

Splitting is patient-grouped and belongs to `build_variants.py`, which owns the
disjointness assertion. This script records **`source_split`** — whatever split the
mirror itself shipped — for reference only. A mirror's split can put one patient's two
eyes on opposite sides (Q5 in `00_verify_inputs.ipynb` tests exactly this), so adopting
it would leak.

### Patient IDs — the one that bites

```python
m = re.match(r"^(\d+)_(left|right)$", Path(p).stem)   # EyePACS
patient_id = f"EyePACS::{m.group(1)}"
eye = m.group(2)
```

Messidor-2 pairs eyes per examination (`20051020_43808_0100_PP` → `Messidor2::20051020_43808`).
DDR, IDRiD and APTOS have no patient identifiers, so each image becomes its own
patient and the script **warns loudly** — that fallback leaks for any dataset with two
eyes per person. Every dataset prefixes its IDs (`EyePACS::`, `DDR::`, …) so merged
manifests cannot collide.

Sanity check: EyePACS and Messidor-2 should report ~2.0 images per patient. A value
near 1.0 means the pairing failed and the split will leak.

### IDRiD coordinates are re-projected, not copied

Part C publishes optic-disc and fovea centres in **original pixel coordinates**, which
the cache's crop, square fit and resize invalidate. Rather than storing a transform at
cache time, the geometry is recomputed here by calling `build_cache.retinal_bbox` and
`build_cache.map_point` — the same functions that produced the cache, so the two cannot
drift apart. `--size`, `--fit` and `--tol-scale` must therefore match the cache build.
Each point also gets a `*_in_frame` flag, since a crop can legitimately push a centre
outside the frame.

### Matching is by filename stem

Label files reference `.png` where the cache holds `.jpg` (Messidor-2 does exactly
this), so rows are joined on stem. Unmatched entries are reported in both directions
rather than silently dropped.

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
