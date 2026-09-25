# notebooks/

Kaggle notebooks, one per phase. Each attaches the datasets it needs, clones this
repo, and calls into `scripts/`.

**Keep notebooks thin.** Code lives in `scripts/` and `src/`, under version control.
Notebook cells aren't diffable, and six weeks in you won't know which variant produced
which number. A notebook should pull the repo, call a script, and show the output.

## Status

| Notebook | Phase | GPU? | Status |
|---|---|---|---|
| `00_verify_inputs.ipynb` | 0 | No | Done — answers the verification questions |
| `01_build_cache.ipynb` | 1 | **No** | Done — the 512 px cache, A0 |
| `01b_idrid_masks.ipynb` | 1 | **No** | Done — IDRiD Part A masks top-up |
| `01c_idrid_grading.ipynb` | 1 | **No** | Done — IDRiD Part B, for C1 |
| `02_manifests.ipynb` | 2 | No | Done — manifests, patient-grouped splits, variants |
| `03_grading_sweeps.ipynb` | 3 | Yes | Done — B1–B3 |
| `03b_stage_b_batch.ipynb` | 3 | Yes | Done — B4 and B5 in one session |
| `04_phase4.ipynb` | 4 | Yes | Done — C1–C4 |
| `06_final_training.ipynb` | 6a | Yes | Done — the six final runs |
| `06_check_runs.ipynb` | 6a | No | Done — run status, the official-split check (D8) |
| `07_internal_pass.ipynb` | 7, step 1 | Yes | Done 2026-09-24 — `verify-dr-internal` |
| `08_fit_and_rehearse.ipynb` | 7, step 2 | **No** | Run once 2026-09-24; **re-run** for the amended analysis (D12, D13), the one round before the lock |
| `09_locked_pass.ipynb` | 6b, step 4 | Yes | Ready — refuses to start before step 3 (parameters committed, digest recorded) and on any checkpoint that is not the fitted one |
| `10_unblinding.ipynb` | 6b, step 5 | **No** | Ready — the one label join; both analyses, registered first |

Each remaining notebook gets written **in the same commit as the script it drives** —
a notebook and its script are tested together or neither works.

## Settings (right-hand panel)

| Setting | Value | Why |
|---|---|---|
| **Accelerator** | None for phases 0/1/2/8; **GPU T4 ×2** for 3/4/6/7 | T4 has tensor cores, so AMP pays off. P100 doesn't. |
| **Persistence** | **Files only** on the cache build | `build_cache.py` resumes, so `/kaggle/working` surviving lets a 10-hour build span several sittings with no rework. |
| **Environment** | **Pin to original environment** | Kaggle updates its base image; over 20 weeks that will silently change a dependency. |
| **Internet** | On only for `git clone` / pip | cv2, numpy, pandas and torch are preinstalled. |

**CPU-only sessions do not consume the 30 GPU-h/week quota.** The cache build is
6–10 hours — run it with the accelerator on None and it costs nothing. Run it with a
GPU attached and you have burned a third of a week on JPEG decoding.

## When to create a new notebook

Whenever **any** of these is true:

- The accelerator changes (CPU ↔ GPU)
- It is a different phase
- A run would approach the ~12 h session cap
- It is the unblinding — `06_unblinding.ipynb` runs **once** and must be separately
  auditable

Within a phase, one experiment per *run*, not per notebook. A crash then costs one
result rather than a batch.

## Inputs per notebook

| Notebook | Inputs |
|---|---|
| `00_verify_inputs` | All six — the one time you want them together |
| `01_build_cache` | EyePACS, DDR, IDRiD (+ APTOS, Messidor-2 on a later pass) |
| `02_manifests` | `verify-dr-cache-512` **and the raw datasets** — see below |
| | A cache published as a `.zip` is extracted automatically on first use |
| `03`–`08` | `verify-dr-cache-512` + `verify-dr-manifests`, plus checkpoints and results |

**Phase 2 still needs the raw mounts.** Labels live there, not in the cache: DDR's
`train/valid/test.txt`, IDRiD's Part B grading CSV and Part C coordinate tables,
APTOS's `train.csv`, Messidor-2's grades. IDRiD coordinate re-projection also reads the
original Part A images, because the published centres are in original pixel space and
the geometry has to be recomputed against them.

**From Phase 3 onward the cache and manifests are enough** and the raw datasets can be
detached. Publish each via *Save Version* → Output tab → **New Dataset**, named
`verify-dr-cache-512` and `verify-dr-manifests`.

## Notebooks do not share `/kaggle/working`

Each notebook gets its own working directory, so files one notebook writes are not
visible to the next. `00_verify_inputs.ipynb` writes `verification_log.json` as a
convenience, but **you do not need to download and re-upload it** —
`01_build_cache.ipynb` re-discovers every path itself and uses the log only as an
optional accelerator when it happens to be present.

To carry real artefacts between notebooks (the cache, checkpoints), publish them as a
Kaggle dataset and add that as an input.

## Getting the repo in

The clone cell handles both cases. For a private repo, store a GitHub PAT under
**Add-ons → Secrets** as `GH_TOKEN`; for a public repo it clones anonymously and the
token lookup is skipped.

## Re-run the clone cell after any repo change

The clone cell prints the commit it checked out:

```
repo ready at /kaggle/working/repo
checked out: 42034e0  Handle IDRiD Part A having no grades; ...
```

Updating a notebook cell does **not** update the scripts on disk. Running one cell in
isolation after a fix leaves `/kaggle/working/repo` at the old commit, and the failure
then surfaces far from its cause — usually as `exit 2`, argparse rejecting a flag the
old script does not have. `02_manifests.ipynb` checks for the flags it needs and says
so plainly if the checkout is behind.

When in doubt, run from the clone cell down.

## Running long jobs

**Save Version → Save & Run All (Commit)**, not interactive. Interactive sessions die
when your browser idles or disconnects; batch commits run unattended to the session
cap.

- Checkpoint every epoch. Resume, never restart to get a "clean" number.
- Save checkpoints as a Kaggle dataset — `/kaggle/working` does not survive.
- Write results to `results/<stage>/<id>/` immediately, not at the end.

## Before running `06_unblinding.ipynb`

1. `preregistration/PREREGISTRATION.md` is filled in and committed
2. The freeze commit hash is recorded inside it
3. Your supervisor has acknowledged it in writing

That notebook consumes the locked external sets. There is no second attempt.
