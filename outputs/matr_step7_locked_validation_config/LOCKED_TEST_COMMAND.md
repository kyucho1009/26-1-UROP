# Locked Test Evaluation Command

Run this only after the validation-selected configuration is fixed.

Linux/GPU:

```bash
bash outputs/matr_step7_locked_validation_config/run_locked_test.sh
```

PowerShell/GPU:

```powershell
.\outputs\matr_step7_locked_validation_config\run_locked_test.ps1
```

Direct command:

```bash
python scripts/run_matr_locked_test_evaluation.py \
  --data-root MATR \
  --config-path outputs/matr_step7_locked_validation_config/final_validation_config.json \
  --output-dir outputs/matr_locked_test_evaluation_final \
  --device cuda
```

Expected output files:

```text
outputs/matr_locked_test_evaluation_final/
  locked_test_config.json
  dataset_manifest.json
  split_manifest_seed42.json
  split_manifest_seed43.json
  split_manifest_seed44.json
  validation_results_raw.csv
  validation_results_by_seed.csv
  validation_summary_by_model_horizon.csv
  locked_validation_summary.csv
  test_results_raw.csv
  test_results_by_seed.csv
  test_summary_by_model_horizon.csv
  locked_test_summary.csv
  selected_model_test_summary.json
  test_predictions.csv
  checkpoints/
  README.md
```

Do not use the generated test metrics to change model architecture or
hyperparameters.
