# MATR Step 7 Locked Validation Configuration

This folder freezes the validation-selected configuration after Optuna tuning.

- Dataset: MATR
- Test metrics used: false
- Selected model: `cpmlp_cpdsconv_fusion`
- Source trial: Optuna confirm `trial 15`, candidate `1`
- Target scale: `100`
- Selection rule: validation-only metric values, lowest average validation MAE with RMSE, MAPE, MAE stability, and skill as tie-breakers

Locked validation evidence:

| Metric | Value |
| --- | ---: |
| avg_MAE_mean | 0.0017806132 |
| avg_RMSE_mean | 0.0056678547 |
| avg_MAPE_percent_mean | 0.1931147271 |
| std_MAE_mean | 0.0014779451 |
| Skill MAE vs persistence | 0.2643207201 |
| MAE improvement percent vs CPMLP | 7.2923750541 |

Reproduce the locked validation run:

```bash
bash outputs/matr_step7_locked_validation_config/run_locked_validation.sh
```

PowerShell:

```powershell
.\outputs\matr_step7_locked_validation_config\run_locked_validation.ps1
```

Run the locked test evaluation after the configuration is frozen:

```bash
bash outputs/matr_step7_locked_validation_config/run_locked_test.sh
```

PowerShell:

```powershell
.\outputs\matr_step7_locked_validation_config\run_locked_test.ps1
```

Do not change model or hyperparameters after this point based on test metrics.
