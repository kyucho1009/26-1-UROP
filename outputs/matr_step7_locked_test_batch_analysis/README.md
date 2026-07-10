# MATR Locked Test Inference-Only Batch Analysis

This folder reuses the previously trained global-batch checkpoints and the
original battery-level test split. No training, validation selection, Optuna
tuning, or batch-specific fine-tuning was performed in this run.

- Selected model: `cpmlp_cpdsconv_fusion`
- Models evaluated: ['persistence', 'cpmlp', 'cpmlp_cpdsconv_fusion']
- Lookback cycles: 10
- Horizons: [10, 50, 100]
- Seeds: [42, 43, 44]
- Dataset batch filter: none (all batches loaded)
- Checkpoint root: `outputs\matr_step7_locked_test_cpdsconv_l10_h10_h50_h100_wide_best\checkpoints`
- Source split manifest root: `outputs\matr_step7_locked_test_cpdsconv_l10_h10_h50_h100_wide_best`
- Target scale: loaded from each checkpoint

Important files:

- `test_predictions.csv`: window-level predictions, including `batch_id`
- `test_summary_by_model_horizon.csv`: global test metrics reproduced from checkpoints
- `test_results_by_seed_batch_model_horizon.csv`: window-weighted batch metrics per seed
- `test_summary_by_batch_model_horizon.csv`: seed-aggregated window-weighted batch metrics
- `test_metrics_by_cell.csv`: metrics for each held-out battery cell
- `test_summary_macro_cell_by_batch_model_horizon.csv`: seed-aggregated cell-macro batch metrics
