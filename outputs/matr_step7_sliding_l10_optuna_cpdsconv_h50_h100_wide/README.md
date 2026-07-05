# MATR Step 7 Fusion Optuna Tuning

This directory was produced by `scripts/tune_matr_step7_fusion_optuna.py`.

- Tuned models: cpmlp_cpdsconv_fusion
- Target scale: 100.0
- Test metrics used: false
- Search trials per model: 50
- Confirm top-k per model: 3
- Search reference models: persistence
- Confirm reference models: persistence, cpmlp
- Objective: validation `avg_MAE_mean` from the selected fusion model row.

Each Optuna trial calls `scripts/run_matr_step7_validation_selection.py` with
`persistence`, `cpmlp`, and one fusion model. Baselines are references; the
trial score is computed from the tuned fusion model only.
