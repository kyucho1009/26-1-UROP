$ErrorActionPreference = "Stop"

python scripts/run_matr_step7_validation_selection.py `
  --data-root MATR `
  --output-dir outputs/matr_step7_locked_validation_run `
  --lookback 20 `
  --horizons 10 50 100 `
  --seeds 42 43 44 `
  --models persistence cpmlp cpmlp_cpdsconv_fusion `
  --device cuda `
  --target-scale 100 `
  --epochs 100 `
  --patience 12 `
  --lr 0.0004076706 `
  --weight-decay 0.0000030463 `
  --batch-size 16 `
  --mlp-embed-dim 64 `
  --gru-embed-dim 64 `
  --model-hidden 256 `
  --gru-hidden 64 `
  --dsconv-channels 64 `
  --dropout 0.35 `
  --clip-grad-norm 1.0
