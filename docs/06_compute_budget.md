# VERIFY-DR — Compute Budget

**Constraint:** Kaggle gives 30 GPU-hours/week, sessions capped at ~12 hours.
**Plan:** ~100 GPU-hours total ≈ 3–4 weeks of quota, spread over 16–20 weeks.

Compute is not the bottleneck. Writing and analysis are.

---

## Budget

| Stage | Work | GPU-h |
|---|---|---|
| 0 | 512 px pre-resize cache | **0** (6–10 CPU-h) |
| 1 | B1 resolution ablation, `balanced_1000`, 1 seed | 5 |
| 2 | B2–B4 recipe search, `eyepacs_full` @512, 1 seed | 10 |
| 3 | C1 OD/fovea regressor (IDRiD, 516 images) | 1 |
| 4 | C2–C4 evidence U-Net (~840 images), 4 runs | 4 |
| 5 | Frozen recipe × 3 seeds × 2 variants | 12 |
| 6 | B5 eye-pair fusion head | 3 |
| 7 | D/E/F — calibration, faithfulness, triage (inference only) | 5 |
| 8 | Contingency ×2 | ~40 |
| | **Total** | **~80–100** |

---

## Why the cache changes everything

The original blueprint read raw JPEGs from `/kaggle/input` every epoch. Fundus images
are large and variable-sized, so **JPEG decode, not the GPU, is the bottleneck** —
roughly 25–40 min/epoch on full EyePACS, with the GPU mostly idle.

Pre-resized to 512 px, the same epoch takes 7–9 minutes. One run drops from ~4 hours
to ~1.5.

Applied across the blueprint's original grid (4 variants × 3 suites × 3 seeds), the
difference is the difference between a feasible thesis and roughly a year of
wall-clock. Build the cache first; everything else depends on it.

---

## Rules of thumb

- **Full EyePACS @512, B0, batch 32, AMP, T4:** ~7–9 min/epoch *from cache*.
  12 epochs ≈ 1.5–1.8 h.
- **`balanced_1000` (~4,300 images):** ~1 min/epoch. Use it for anything exploratory.
- **Evidence U-Net (~840 images):** ~1 h for a full run. Effectively free.
- **768 px:** ~2.2× the cost of 512. Budget for it in B1 only.

---

## Working within session limits

1. **Checkpoint every epoch.** Sessions die. Resume rather than restart — never
   restart a run to "get a clean number".
2. **One experiment per notebook run.** Simplifies resume and stops one crash taking
   out a batch of results.
3. **Write results to `results/<stage>/<experiment_id>/` immediately**, not at the end.
4. **Run the cache build on CPU-only.** It consumes no GPU quota.
5. **Save checkpoints as a Kaggle dataset**, not just to `/kaggle/working` — working
   directories do not survive.

---

## If you fall behind

Cut in this order. Everything above the line is load-bearing for the thesis.

| Priority | Item | Why |
|---|---|---|
| Keep | F1, F2 | The thesis |
| Keep | E1, E2 | Faithfulness needs its control |
| Keep | C2, C4 | Evidence path must exist and be characterised |
| Keep | H1 | Your one comparable number |
| Keep | D2, D3 | Contribution #2 |
| — | — | — |
| Cut first | B6 dataset-balancing ablation | Interesting, not load-bearing |
| Cut | D4 EM sample-size sweep | Nice-to-have practicality result |
| Cut | B2 backbone comparison | Just use EfficientNet-B0 |
| Cut | 3 seeds → 2 | Report the reduced spread honestly |
| Cut | B7 / `eyepacs_ddr_full` | Drops H3; state it as future work |

Never cut E2, and never cut F2.
