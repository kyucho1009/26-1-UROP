#!/usr/bin/env bash
set -euo pipefail

python scripts/run_matr_locked_test_evaluation.py \
  --data-root MATR \
  --config-path outputs/matr_step7_locked_validation_config/final_validation_config.json \
  --output-dir outputs/matr_locked_test_evaluation_final \
  --device cuda
