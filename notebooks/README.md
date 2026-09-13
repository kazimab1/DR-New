# notebooks/

Kaggle notebooks, one per phase. Each attaches the datasets it needs, pulls this repo,
and calls into `scripts/`.

| Notebook | Phase | GPU? | Purpose |
|---|---|---|---|
| `00_verify_inputs.ipynb` | 0 | No | Attach all six datasets; answer the four verification questions in `docs/05_dataset_card.md` |
| `01_build_cache.ipynb` | 1 | **No** | Build the 512 px cache; save as a private Kaggle dataset. CPU-only — no quota used. |
| `02_manifests.ipynb` | 2 | No | Build manifests and splits; assert patient-disjointness |
| `03_grading_sweeps.ipynb` | 3 | Yes | B1–B6 |
| `04_evidence.ipynb` | 4 | Yes | C1–C4 |
| `05_final_training.ipynb` | 6 | Yes | Frozen recipe × 3 seeds × 2 variants |
| `06_unblinding.ipynb` | 6 | Yes | **The single external evaluation.** Run once. |
| `07_calibration_triage.ipynb` | 7 | Yes | D, E, F |
| `08_error_analysis.ipynb` | 8 | No | G1–G3; produces figure 1 |

## Kaggle housekeeping

- **Checkpoint every epoch.** Sessions die at ~12 h. Resume; never restart to get a
  "clean" number.
- **Save checkpoints as a Kaggle dataset** — `/kaggle/working` does not survive.
- **One experiment per run.** A crash then costs one result, not a batch.
- **Write to `results/<stage>/<id>/` immediately**, not at the end.

## Before running `06_unblinding.ipynb`

1. `preregistration/PREREGISTRATION.md` is filled in and committed
2. The freeze commit hash is recorded inside it
3. Your supervisor has acknowledged it in writing

This notebook consumes the locked external sets. There is no second attempt.
