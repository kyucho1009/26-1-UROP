# MATR Step 7 Validation Selection

This directory was produced by `scripts/run_matr_step7_validation_selection.py`.

- Dataset used: MATR only
- Selection stage: step7_validation_only
- Test metrics used for selection: false
- Lookback cycles: 10
- Sample mode: sliding-window
- Horizons: [50, 100]
- Features: ['current', 'voltage', 'dV']
- Target: delta_soh = SOH_input_end - SOH_(input_end+h)
- Model selection: validation-only metric values; lowest average MAE across horizons,
  then lower average RMSE, lower average MAPE, lower MAE standard deviation, and
  higher average skill versus persistence
- Selected model: cpmlp_cpdsconv_fusion

The test split is saved only in split manifests. No test predictions or test
metrics are computed in this Step 7 pipeline.
