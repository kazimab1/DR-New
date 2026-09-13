# VERIFY-DR — Dataset Card

Six Kaggle datasets. Every source is Kaggle-hosted so the whole project runs in a
Kaggle notebook.

---

## Roster

| Dataset | Kaggle slug | Role | Locked? |
|---|---|---|---|
| EyePACS | `tantai31124/eyepacs-original` (or the original competition) | Primary development | No |
| DDR | `samriddhibagchi/ddr-dataset-credits-to-authors` | Merged dev + **lesion masks** + ungradable class | No |
| IDRiD | a mirror carrying **Parts A + C** (see below) | Lesion masks + **OD/fovea coords** + grading | No |
| APTOS 2019 | `aptos2019-blindness-detection` (competition) | **External test #1** | **YES** |
| Messidor-2 images | `mariaherrerot/messidor2preprocess` | **External test #2** | **YES** |
| Messidor-2 grades | `google-brain/messidor2-dr-grades` | Adjudicated labels — join by image ID | **YES** |

---

## Approximate grade distributions

Verify these against your actual mirrors — counts differ between mirrors.

**EyePACS** (competition train, 35,126 images):

| Grade | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| n | ~25,810 | ~2,443 | ~5,292 | ~873 | ~708 |
| % | 73.5 | 7.0 | 15.1 | 2.5 | 2.0 |

Full set including the released test labels is ~88,702 images (grade 3 ≈ 2,087,
grade 4 ≈ 1,914).

**APTOS 2019** (train, 3,662 images):

| Grade | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| n | ~1,805 | ~370 | ~999 | ~193 | ~295 |
| % | 49.3 | 10.1 | 27.3 | 5.3 | 8.1 |

**The prevalence shift is the point.** Grade 0 moves 73% → 49% between EyePACS and
APTOS. That is why temperature scaling will not transfer (D2) and why prior-shift
correction is a real contribution (D3).

**DDR** (~13,673 images): includes ~1,151 **grade-5 ungradable** images. Do not drop
them — they are free supervision for the quality/REACQUIRE head (G2).

---

## Verification log — fill this in during Phase 0

Two questions are load-bearing. Answer both in writing before proceeding.

### Q1. Does the DDR mirror contain lesion segmentation masks?

```bash
find /kaggle/input/ddr-dataset-credits-to-authors -maxdepth 3 -type d | head -40
find /kaggle/input/ddr-dataset-credits-to-authors -iname "*segmentation*" -maxdepth 3 | head
```

You are looking for a `lesion_segmentation/` tree with per-lesion mask folders
(microaneurysm, haemorrhage, hard exudate, soft exudate), roughly 757 annotated
images.

> **If it is not there, stop.** The evidence pathway has no training data and there is
> no version of this thesis without it. Find another mirror first.
>
> Mirrors of DDR vary: some carry only `DR_grading/`, some carry the full release.
> `notebooks/00_verify_inputs.ipynb` Q1 discovers the layout rather than assuming it,
> so run that before committing to this mirror. The canonical DDR segmentation subset
> is 757 images split 383 train / 149 valid / 225 test.

**Answer:** _(fill in)_

### Q2. What preprocessing has `messidor2preprocess` already applied?

```bash
find /kaggle/input/messidor2preprocess -maxdepth 2 | head -20
python -c "from PIL import Image; im=Image.open('<a sample>'); print(im.size, im.mode)"
```

Check whether images are already contrast-normalised or Ben-Graham processed. If they
are, your measured "domain gap" is partly a preprocessing artefact — either match the
processing across all sources, or report the confound explicitly.

**Answer:** _(fill in)_

### Q3b. Observed source layouts

Recorded from a real Kaggle session, so Phase 2 knows what it is parsing:

| Source | Layout |
|---|---|
| DDR | `DDR-dataset/` nested inside the mount, then `DR_grading/{train,valid,test}` + `{train,valid,test}.txt`, `lesion_segmentation/{train,valid,test}/{image,label}`, `lesion_detection/` (XML boxes, unused) |
| EyePACS | `EYEPACS_original_filed/{train,val,test}/` — the mirror ships **its own split**, which Q5 tests for patient disjointness |
| IDRiD | `Imagenes/IDRiD_*.jpg` + `idrid_labels.csv` (Part B only) |
| Messidor-2 | images under `messidor-2/messidor-2/preprocess/`, grades in a separate dataset |

**DDR grading labels are `.txt`, not `.csv`** — lines of `<image> <grade>`. The manifest
builder must handle that, not assume a CSV.

### Q3. Do the Messidor-2 grades join cleanly to the images?

Images and adjudicated grades are **two separate Kaggle datasets**. Confirm the ID
join covers every image and record how many fail to match.

**Answer:** _(fill in)_

### Q4. Does IDRiD include the OD/fovea coordinate CSVs?

C1 — and therefore all quadrant reasoning — depends on them.

**IDRiD is published in three parts, and a grading-only mirror has neither piece
this project needs:**

| Part | Contents | Needed for |
|---|---|---|
| A — Segmentation | MA/HE/EX/SE pixel masks + optic-disc masks | C2 (extra mask training data) |
| B — Grading | 516 images + severity CSV | nothing critical |
| **C — Localization** | **optic-disc + fovea centre coordinates** | **C1, and therefore all of M3** |

A mirror titled "IDRiD Diabetic Retinopathy – Grading" is **Part B only**. Without
Part C there is no coordinate frame, so no quadrant assignment, so the haemorrhage arm
of the 4-2-1 rule cannot be evaluated and M3 falls back to count-only rules.

That is survivable — declare it in the limitations — but it costs the most defensible
part of the reasoner. Prefer adding a Parts A + C mirror.

**Answer:** _(fill in)_

---

## Manifest contract

Every manifest, development or external:

| Column | Required | Notes |
|---|---|---|
| `image_path` | Yes | Path into the 512 px cache |
| `grade` | Yes | 0–4 (DDR grade 5 goes to the quality manifest, not here) |
| `dataset` | Yes | `EyePACS` / `DDR` / `IDRiD` / `APTOS` / `Messidor2` |
| `patient_id` | Yes | Dataset-prefixed, e.g. `EyePACS::16` |
| `eye` | Where known | `left` / `right` — required for eye-pair fusion (B5) |
| `split` | Yes | `train` / `val` / `calibration` / `test` |
| `*_mask` | Optional | Four lesion channels, where annotated |

### Patient IDs — the one that bites

EyePACS filenames are `<patient>_<left|right>.jpeg`. Parse them:

```python
m = re.match(r"^(\d+)_(left|right)$", Path(p).stem)
patient_id = f"EyePACS::{m.group(1)}"
eye = m.group(2)
```

The original blueprint used an image-stem proxy, which makes every image its own
patient and puts a patient's two eyes on opposite sides of the split. DR is bilateral
and correlated, so that inflates every number you report. The split builder asserts
patient-disjointness — do not disable it.

---

## Licence and ethics

All six are publicly released research datasets, used here for non-commercial academic
research. All are de-identified at source; no re-identification is attempted. Cite the
original dataset papers in the thesis, not just the Kaggle mirrors.

Research software, not a clinical device.
