# VERIFY-DR — Model Architecture

> **As built (2026-09-22) — read this first.** This document is the *design*, kept as the
> record of intent. What runs at the unblinding differs in these places; the diagram is in
> the report's architecture section (`docs/report.html`), and the analysis choices in
> `preregistration/ANALYSIS_PLAN.md`.
>
> - **M0:** the cache pads to square rather than centre-cropping (A0). **The quality head
>   was never built**, so nothing is routed to REACQUIRE (D10).
> - **M2a:** trained on DDR-seg only; IDRiD held out as C3's unseen domain (D2).
> - **M2b:** 0.686 DD against a 0.5 gate — **never run**, so there are no quadrants (D1).
> - **M3:** R1–R3* only, evidence grades 0–2, abstains above 2; R4 and R5 cannot fire (D1).
> - **M4a:** a parameter-free sampling-prior correction precedes temperature scaling,
>   because the frozen sampler trains on a uniform class prior (D11).
> - **M4b/M4c:** faithfulness is a 19-draw randomisation test rather than a margin;
>   `d_evidence` is 0 when M1's grade is above 2; `d_conf` uses p(ŷ), not max p; the two
>   gate weights reduce to one ratio r.

Five modules. The two prediction pathways are **trained on different supervision and
never share weights or gradients** — that independence is the whole point, and it is
what lets one pathway meaningfully dissent from the other.

---

## 1. System overview

```
                         fundus image (variable size)
                                     |
                    ┌────────────────▼────────────────┐
                    │  M0  PREPROCESSING & QUALITY    │
                    │  retinal-field crop → 512×512   │
                    │  CLAHE on L channel             │
                    │  quality head → gradable?       │
                    └────────┬───────────────┬────────┘
                             │               │
              ┌──────────────▼──┐         ┌──▼──────────────────┐
              │ M1  GRADING     │         │ M2  EVIDENCE        │
              │ PATHWAY         │         │ PATHWAY             │
              │                 │         │                     │
              │ EfficientNet-B0 │         │ ResNet18-UNet       │
              │ ordinal head    │   NO    │ 4 lesion channels   │
              │ + rDR / VTDR    │ SHARED  │ + OD/fovea regressor│
              │                 │ WEIGHTS │                     │
              │ trained on      │         │ trained on          │
              │ IMAGE GRADES    │         │ PIXEL MASKS         │
              │ (EyePACS/DDR)   │         │ (DDR-seg, IDRiD)    │
              └────────┬────────┘         └──────────┬──────────┘
                       │                             │
                 grade + probs              lesion counts + geometry
                       │                             │
                       │                  ┌──────────▼──────────┐
                       │                  │ M3  ICDR REASONER   │
                       │                  │ deterministic rules │
                       │                  │ → evidence_grade    │
                       │                  │ → unobservable[]    │
                       │                  └──────────┬──────────┘
                       │                             │
                    ┌──▼─────────────────────────────▼──┐
                    │  M4  VERIFICATION & TRIAGE        │
                    │  • counterfactual faithfulness    │
                    │  • calibration + prior shift      │
                    │  • DISAGREEMENT GATE  ◄── novel   │
                    │  → ACCEPT / ADJACENT / DEFER /    │
                    │    REACQUIRE                      │
                    └───────────────┬───────────────────┘
                                    │
                          structured decision record
```

---

## M0 — Preprocessing and quality

**Offline (once, cached).** Threshold the image at low intensity to find the retinal
circle, crop to its bounding box, resize shortest side to 512, centre-crop 512×512,
save as JPEG q90. Apply CLAHE (clipLimit 2.0, 8×8 tiles) to the L channel of LAB.

**Online (quality head).** A 3-layer CNN on the 512×512 input, plus hand-computed
Laplacian variance, mean brightness and RMS contrast.

| | |
|---|---|
| Output | `quality_score ∈ [0,1]`, `gradable ∈ {0,1}` |
| Supervision | **DDR grade 5** (n≈1,151 ungradable) as positives, random gradable images as negatives |
| Loss | BCE |
| Note | The original blueprint left the quality head unsupervised. DDR's ungradable class is free labels for exactly this. |

Anything with `gradable = 0` short-circuits to `REACQUIRE` and never reaches M1/M2.

---

## M1 — Grading pathway

Predicts ICDR grade from the image alone. This is the "conventional" model that
everything else exists to audit.

```
512×512×3
  → EfficientNet-B0 (ImageNet init)        [B, 1280, 16, 16]
  → global average pool                    [B, 1280]
  → eye-pair fusion (optional, B5)         [B, 2560]
  → dropout(0.3) → FC 512 → ReLU           [B, 512]
  ├─ ordinal head    → 4 logits            [B, 4]
  ├─ rDR head        → 1 logit             [B, 1]
  └─ VTDR head       → 1 logit             [B, 1]
```

### Ordinal head

Four binary thresholds, CORAL-style: unit `k` predicts `P(grade > k)`. A shared weight
vector with four learned biases **guarantees monotonicity** — you cannot get
`P(grade>2) > P(grade>1)`, which an independent 4-logit head would happily produce.

```
P(grade > k) = σ(wᵀh + b_k),   b_0 ≥ b_1 ≥ b_2 ≥ b_3   (enforced by construction)

grade = Σ_k 1[P(grade > k) > 0.5]

P(grade = 0) = 1 − P(>0)
P(grade = k) = P(>k−1) − P(>k)      for k = 1,2,3
P(grade = 4) = P(>3)
```

That class-probability vector is what M4 calibrates.

### Auxiliary heads

- **rDR** — referable DR, grade ≥ 2. The clinically operative threshold.
- **VTDR** — vision-threatening DR, grade ≥ 3.

They are redundant with the ordinal head by construction, but training them explicitly
sharpens the decision boundaries that actually matter for screening.

### Loss

```
L_M1 = L_ordinal + 0.3·BCE(rDR) + 0.3·BCE(VTDR)

L_ordinal = Σ_k focal(σ(wᵀh + b_k), 1[y > k]),   γ = 2.0
```

Focal weighting handles the ~36:1 grade-0 to grade-4 imbalance without discarding
data. B3 tests plain CE and non-focal ordinal against this.

### Eye-pair fusion (B5)

EyePACS images are `<patient>_left` and `<patient>_right`. Diabetic retinopathy is
bilateral and correlated — the fellow eye is genuine evidence. When both eyes of a
patient are present, concatenate their pooled features before the FC layer. When only
one is available, duplicate it. Cheap, and historically one of the larger single
gains on this dataset.

### Training

| | |
|---|---|
| Optimiser | AdamW, lr 3e-4, weight decay 1e-4 |
| Schedule | Cosine, 2-epoch linear warmup |
| Precision | AMP (fp16) |
| Batch | 32 at 512 px on a T4 |
| Epochs | 12, early stop on validation QWK, patience 3 |
| Augmentation | Random flip, rotation ±15°, brightness/contrast ±0.2, mild scale jitter. **No colour-channel shuffling** — lesion colour is diagnostic. |

---

## M2 — Evidence pathway

Completely separate network. Different architecture, different data, different
supervision, no shared parameters. If it shared an encoder with M1 the two pathways
would fail together and disagreement would carry no information.

### M2a — Lesion segmenter

```
512×512×3
  → ResNet18 encoder (ImageNet init)
      skips at 256², 128², 64², 32²
  → UNet decoder with those skips
  → 1×1 conv → 4 channels → sigmoid        [B, 4, 512, 512]
```

Four channels, and only four, because these are the only lesion types with pixel
annotations on Kaggle:

| Channel | Lesion | ICDR relevance |
|---|---|---|
| 0 | Microaneurysm | Presence ⇒ at least mild NPDR |
| 1 | Haemorrhage | Counts per quadrant drive the 4-2-1 rule |
| 2 | Hard exudate | Supports moderate NPDR; DME marker |
| 3 | Soft exudate (cotton wool spot) | Supports moderate–severe NPDR |

> **Why not eight?** The original blueprint specified 8 channels including venous
> beading, IRMA and neovascularisation. No Kaggle-available dataset labels those at
> pixel level (FGADR does, but requires a signed agreement). Rather than pretend, M3
> declares them unobservable — see below.

**Training data:** DDR lesion-segmentation subset (~757 images) + IDRiD segmentation
(81 images). Small, so augmentation is heavy and synchronised between image and mask.

**Loss:** `0.5·Dice + 0.5·BCE`, per channel, averaged. Dice alone is unstable when a
channel is empty — many images have no soft exudates at all.

### M2b — Optic disc / fovea regressor

```
512×512×3
  → ResNet18 (shared encoder with M2a, separate head)
  → GAP → FC 128 → FC 4
  → [od_x, od_y, fovea_x, fovea_y]   normalised to [0,1]
```

**Supervision:** IDRiD ground-truth centre coordinates (516 images).
**Loss:** smooth L1.
**Acceptance gate (C1):** mean error < 0.5 disc diameters, else quadrant assignment is
too noisy to reason over and M3 must fall back to a count-only rule.

This module is what makes partial 4-2-1 reasoning possible. Without disc and fovea
positions you have no coordinate frame, and "haemorrhages in 3 quadrants" is
undefined.

### Structured facts emitted

```python
{
  "microaneurysm":  {"count": int, "quadrants": [str]},
  "haemorrhage":    {"count": int, "quadrants": [str], "per_quadrant": {...}},
  "hard_exudate":   {"count": int, "area_frac": float},
  "soft_exudate":   {"count": int, "quadrants": [str]},
  "geometry":       {"od_center": [x,y], "fovea_center": [x,y], "disc_diameter_px": float}
}
```

Counts come from connected-component analysis on the thresholded masks (threshold
0.5, minimum component area 3 px to suppress speckle).

**Quadrant frame:** origin at the optic disc centre; the disc→fovea vector defines the
temporal axis. Superior/inferior/nasal/temporal follow from that, mirrored for left
versus right eyes.

---

## M3 — ICDR reasoner

Deterministic Python. **No learned parameters.** Every output traceable to a rule that
fired, and every rule quotable in the thesis.

### Rules

```
R1  no lesions of any type                      → evidence_grade 0
R2  MA present, no haemorrhage/exudate          → evidence_grade 1  (mild NPDR)
R3  MA + haemorrhage or exudate,
    4-2-1 not met                               → evidence_grade 2  (moderate NPDR)
R4  4-2-1 severe criterion met:
      ≥20 intraretinal haemorrhages
      in each of 4 quadrants                    → evidence_grade 3  (severe NPDR)
R5  neovascularisation or vitreous
    haemorrhage detected                        → evidence_grade 4  (PDR)
```

### Declared unobservable

R4 in full ICDR also fires on *venous beading in ≥2 quadrants* or *IRMA in ≥1
quadrant*. R5 requires detecting neovascularisation. **None of those three are
observable with Kaggle data.** So the reasoner emits them explicitly:

```json
{
  "evidence_grade": 2,
  "rules_fired": ["R3: MA + haemorrhage present, 4-2-1 haemorrhage criterion not met"],
  "unobservable": ["venous_beading", "IRMA", "neovascularisation"],
  "max_excludable_grade": 3,
  "abstain_upward": true
}
```

`max_excludable_grade = 3` means: *given what I can see, I can rule out nothing above
grade 3.* The reasoner never claims a case is not severe NPDR or PDR on evidence it
does not have.

> This is the honest core of the project. Existing systems have exactly these blind
> spots — single 45° macula-centred fields cannot support real 4-quadrant assessment
> either — and emit a confident grade anyway. Naming the blind spot converts a dataset
> limitation into a safety property.

### Two further limits, stated plainly

- EyePACS and APTOS are largely single-field macula-centred. True ETDRS 4-quadrant
  assessment needs 7 fields. Your quadrants are an approximation over one field and
  the thesis must say so.
- Therefore R4 is a **partial** 4-2-1: the haemorrhage arm only.

---

## M4 — Verification and triage

Where the contribution lives.

### M4a — Calibration

```
Stage 1  temperature scaling      T fitted on the INTERNAL calibration split only
Stage 2  prior-shift correction   Saerens–Decock EM on UNLABELLED target images
```

Stage 2 exists because grade-0 prevalence is ~73% in EyePACS and ~49% in APTOS. A
temperature fitted on one prior does not transfer to the other, and positive
predictive value depends entirely on getting this right. EM estimates the target prior
from unlabelled target images — no target labels are used, so the external set stays
locked.

D3 also reports an oracle variant using the true target prior, as an upper bound on
what the correction could achieve.

### M4b — Counterfactual faithfulness

```
1. Take the lesion regions M2 detected and M3 cited.
2. Inpaint them out (OpenCV Telea, radius 5).
3. Re-run M1 on the modified image.
4. Δ_lesion = grade_original − grade_inpainted
5. CONTROL: inpaint random regions of equal total area → Δ_random
6. faithful := Δ_lesion > Δ_random + margin
```

**Step 5 is not optional.** Without it you have measured that the model reacts to
inpainting, not that it uses the lesions. This control is the difference between a
faithfulness test and a demonstration of nothing.

### M4c — The disagreement gate

The novel part. Four signals, one decision:

```python
d_evidence = |grade_M1 − evidence_grade_M3|      # independent second opinion
d_conf     = 1 − max(p_calibrated)               # the conventional baseline
d_ood      = z_score(embedding, train_manifold)  # distribution shift
d_faith    = 0 if faithful else 1                # explanation is causal?

disagreement_score = w1·d_evidence + w2·d_faith   # the arm under test
```

Decision policy:

| Condition | Action |
|---|---|
| `gradable = 0` | **REACQUIRE** |
| `d_evidence = 0` and `faithful` and `d_ood < τ_ood` | **ACCEPT** — return the grade |
| `d_evidence = 1` | **ADJACENT_GRADE_SET** — return `{g−1, g, g+1}` |
| `d_evidence ≥ 2`, or not `faithful`, or `d_ood ≥ τ_ood` | **DEFER** — human reads it |
| `evidence_grade` low but `unobservable` non-empty and `d_conf` high | **DEFER** |

**F1/F2 compare five arms** — no gate, confidence only, OOD only, disagreement only,
combined — on coverage–accuracy curves. Confidence-only is the Leibig-style baseline
this project exists to beat. If it wins, H1 is falsified and you report that.

---

## Why the independence matters

Confidence-based deferral asks the model to grade its own homework. A miscalibrated
network is *confidently wrong* — and no function of its own logits can detect that,
because the error and the confidence come from the same corrupted computation.

M2 is trained on pixel-level lesion annotations from different hospitals in a
different country. When it disagrees with M1, that disagreement carries information
M1's softmax cannot contain. Two networks trained on the same labels would agree on
the same mistakes; these two need not.

That is the bet. F2 tests whether it pays off under dataset shift.

---

## Parameter budget

| Module | Parameters | Trained on |
|---|---|---|
| M0 quality | ~0.3 M | DDR grade-5 vs gradable |
| M1 grading | ~5.3 M (EfficientNet-B0) | EyePACS ±DDR image grades |
| M2a segmenter | ~14 M (ResNet18-UNet) | DDR-seg + IDRiD-seg pixel masks |
| M2b OD/fovea | ~11 M (shared encoder) | IDRiD coordinates |
| M3 reasoner | **0** | — deterministic |
| M4 triage | 1 (temperature) + 2 gate weights | Internal calibration split |

Small by design. The contribution is the arrangement, not the scale — and every
component has to fit in Kaggle's 30 GPU-hours per week.
