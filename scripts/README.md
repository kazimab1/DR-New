# scripts/ — build utilities

All three Phase 1-2 build scripts are implemented. The remaining contracts are
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

Two mask-naming conventions are handled: DDR reuses the image stem
(`007-0004.jpg` → `label/MA/007-0004.tif`), IDRiD appends a channel suffix
(`IDRiD_55.jpg` → `1. Microaneurysms/IDRiD_55_MA.tif`). If the per-channel tallies
in `cache_report.json` show `written: 0, absent: N`, the mask directories were found
but no filename matched — check the naming before assuming the masks are missing.

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
--no-grades                     build from cached images with grade -1 (masks/geometry only)
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

### IDRiD Part A has no grades

IDRiD's two parts use **different id formats for different images**: Part A
(segmentation) is `IDRiD_01`–`IDRiD_81`, the Part B grading table lists
`IDRiD_001`–`IDRiD_516`. They do not join. Once the cache holds Part A — which is
what the mask top-up publishes — no grading row matches and the script exits 1 with a
diagnostic naming both id formats.

That is what `--no-grades` is for: a manifest built from the cached images with
`grade = -1`, carrying masks and projected coordinates. IDRiD is this project's mask
and geometry source for C1 and C2, not a grading dataset, so grades are not needed.
`build_variants.py` drops anything outside 0–4, so such a manifest can never leak into
a development variant.

### Matching is by filename stem

Label files reference `.png` where the cache holds `.jpg` (Messidor-2 does exactly
this), so rows are joined on stem. Unmatched entries are reported in both directions
rather than silently dropped.

## `build_variants.py` — Phase 2 ✅ implemented

Assembles the development variants and the locked externals from the per-dataset
manifests.

```
--eyepacs FILE  [--ddr FILE] [--aptos FILE] [--messidor2 FILE]
--output-dir DIR   --seed 42
--val-frac 0.10   --calibration-frac 0.05   --test-frac 0.20
--balanced-n 1000
--eyepacs-split {regroup,source}
--dry-run
```

Outputs `eyepacs_full.csv`, `eyepacs_balanced_<N>.csv`, `eyepacs_ddr_full.csv`,
`aptos_external.csv`, `messidor2_external.csv` and `dataset_plan.json`.

### The three development variants share their held-out rows

Validation, calibration and test rows are **byte-identical** across `eyepacs_full`,
the balanced variant and the merged variant. Only training rows differ. That is what
makes B6 (does balancing help?) and B7/H3 (does DDR improve transfer?) controlled
comparisons rather than unrelated experiments — the script asserts it and refuses to
write if it does not hold.

Consequently: **balancing touches train only** (Rule 4 — a balanced test set makes
specificity and PPV meaningless), and **DDR is added to train only**.

### Splitting

Patients, never images. A patient's stratification label is their **worst** grade
across both eyes, so rare grades stay represented everywhere. Patient-disjointness is
asserted for every variant and the script exits non-zero if it fails (Rule 3).

| Mode | Behaviour |
|---|---|
| `regroup` (default) | Ignore the mirror's split, build a fresh patient-grouped one. Always safe. |
| `source` | Honour `source_split`. Verified for patient-disjointness first; **refuses to run** if the mirror's split leaks. |

**Rule 6 consequence.** A regrouped EyePACS test set is *not* the official competition
split, so its QWK is not comparable to the Kaggle leaderboard. The script says so on
exit and records `leaderboard_comparable: false` in `dataset_plan.json`, so the thesis
can state which protocol produced each number. Use `--eyepacs-split source` only when
the mirror genuinely ships the official split.

### Other guarantees

Balanced sampling is without replacement; shortfalls on rare grades are recorded in
`dataset_plan.json` and never fabricated by duplication. Externals get `locked=True`
and `split=test` on every row, for `evaluate.py`'s guard rail (Rule 1). A manifest
whose `dataset` column disagrees with the flag it was passed under is rejected. An
images-per-patient ratio below 1.5 on EyePACS warns loudly, since that means patient
parsing failed upstream and the split would leak.

## `train_grading.py` — Phase 3 **(built)**

Trains M1 per `docs/03_model_architecture.md` § M1 and drives experiments B1–B7.
Reads a `build_variants.py` manifest, trains on `train`, selects on `val`, and never
reads `test`.

```
python scripts/train_grading.py \
    --manifest manifests/eyepacs_balanced_1000.csv \
    --experiment B1_res512 --image-size 512 --epochs 10 --resume
```

Writes to `results/<stage>/<experiment_id>/`: `config.json`, `history.json` (per
epoch), `best.pt` (best val QWK), `checkpoint.pt` (full training state), and
`metrics.json` on completion.

Three things worth knowing:

- **`--cache-root` repaths the manifest.** Manifests store absolute paths, and the
  cache is mounted somewhere different in every Kaggle session. Without it a Phase 2
  manifest fails on a Phase 3 mount; the error says so and names the flag.
- **`--resume` refuses an architecture change.** The backbone is globally pooled, so a
  768 px checkpoint loads into a 384 px run without error and silently produces a
  hybrid experiment. Resuming across a different backbone, head, image size or fusion
  setting is a hard failure; changes that only affect the trajectory (lr, sampler,
  epochs) print a `note:` line to be recorded as a deviation.
- **Checkpoints are written every epoch**, so a killed Kaggle session costs one epoch
  rather than the run.

## `train_evidence.py` / `train_geometry.py` — Phase 4

Per `docs/03_model_architecture.md` §§ M2a, M2b. Same contract as above: a config from
`configs/`, results under `results/<stage>/<experiment_id>/`, checkpoint every epoch.

## `evaluate.py` — Phases 6–7

Runs a trained model over a manifest and emits the decision record in
`docs/03_model_architecture.md` § M4, plus aggregate metrics.

**Guard rail:** refuse to run against a manifest marked `locked: true` unless
`--unblind` is passed *and* `preregistration/frozen_config.yaml` exists. Cheap
insurance against accidentally burning the external sets.
