# configs/

One YAML per experiment. `base.yaml` holds shared settings; stage configs override
only what they vary, so a diff between two configs shows exactly what an experiment
tested.

At the Phase 5 freeze, the selected composition is copied to
`preregistration/frozen_config.yaml` and does not change again.

| File | Experiments |
|---|---|
| `base.yaml` | Shared defaults |
| `stage_b_grading.yaml` | B1–B7 sweep definitions |
| `stage_c_evidence.yaml` | C1–C4 |
| `stage_df_triage.yaml` | D1–D4, E1–E3, F1–F5 |
