# VERIFY-DR

**Evidence-verified diabetic retinopathy grading with disagreement-gated referral.**

MSc thesis project. Trains an image-level DR grader and an independently-supervised
lesion-evidence pathway, then uses *disagreement between them* — rather than the
grader's own confidence — to decide which automated gradings can be trusted.

---

## The research question

> Can disagreement between an image-level DR grader and an independently-supervised
> lesion-evidence pathway be used to decide *which automated gradings to trust* — and
> does that decision rule transfer across datasets better than gating on the model's
> own confidence?

Three pre-registered hypotheses:

| | Hypothesis | Falsified if |
|---|---|---|
| **H1** | Disagreement-gating dominates confidence-gating on the coverage–accuracy curve | Confidence-gating matches or beats it |
| **H2** | Prior-shift correction restores external calibration that temperature scaling loses | External ECE stays flat after correction |
| **H3** | Adding DDR to training improves external transfer | External QWK unchanged or worse |

All three may come out negative. Pre-registration is what makes a negative result
publishable rather than a failure.

---

## Where to start

**Read [`docs/00_START_HERE.md`](docs/00_START_HERE.md).** It is the start-to-finish
guide and it tells you what to do in what order.

| Document | What it is |
|---|---|
| [`docs/00_START_HERE.md`](docs/00_START_HERE.md) | The full walkthrough, phase by phase |
| [`docs/01_project_assessment.docx`](docs/01_project_assessment.docx) | Supervisor-facing assessment: what changed from the original blueprint and why |
| [`docs/02_research_protocol.md`](docs/02_research_protocol.md) | Research question, hypotheses, and the rules that keep the study honest |
| [`docs/03_model_architecture.md`](docs/03_model_architecture.md) | Full architecture: every module, tensor shape, and loss |
| [`docs/04_experiment_register.md`](docs/04_experiment_register.md) | All 26 experiments with a status column — your running log |
| [`docs/05_dataset_card.md`](docs/05_dataset_card.md) | The six Kaggle datasets, their roles, and how to verify each |
| [`docs/06_compute_budget.md`](docs/06_compute_budget.md) | GPU-hour plan that fits Kaggle's 30 h/week quota |
| [`docs/07_thesis_outline.md`](docs/07_thesis_outline.md) | Chapter map, with which experiment feeds which section |
| [`docs/08_glossary.md`](docs/08_glossary.md) | DR, ICDR and metrics terminology |
| [`docs/presentation/`](docs/presentation/) | 5-minute project presentation |
| [`preregistration/`](preregistration/) | The freeze. Fill this in before touching external data. |

---

## The one rule that matters most

**APTOS and Messidor-2 are locked.** They are never used for training, validation,
early stopping, calibration fitting, hyperparameter choice, or model selection. You
run them **once**, after `preregistration/PREREGISTRATION.md` is filled in and
committed.

Breaking this rule invalidates the entire generalisation claim, and it cannot be
undone by re-running anything.

---

## Status

Project scaffolding only. No experiments have been run.
