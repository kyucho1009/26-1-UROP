# MATR Locked Test Evaluation

This folder evaluates the validation-locked model configuration on the held-out
test split. Test metrics are reported only after model and hyperparameters were
locked.

- Selected model: `cpmlp_cpdsconv_fusion`
- Test metrics used for selection: false
- Lookback cycles: 5
- Horizons: [50, 100]
- Seeds: [42, 43, 44]
- Target scale: 100.0
- Features: ['current', 'voltage', 'dV']

Important files:

- `test_results_raw.csv`: test metrics per model, seed, and horizon
- `test_summary_by_model_horizon.csv`: seed-aggregated test metrics per horizon
- `locked_test_summary.csv`: final cross-horizon test summary
- `selected_model_test_summary.json`: selected model test result summary
