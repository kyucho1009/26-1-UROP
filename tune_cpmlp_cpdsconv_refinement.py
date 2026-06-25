from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_command(args: argparse.Namespace) -> list[str]:
    tuner = Path(args.tuner).resolve()
    config_file = Path(args.config_file).resolve()
    command = [
        sys.executable,
        str(tuner),
        "--script",
        str(Path(args.compare_script).resolve()),
        "--data-dir",
        str(Path(args.data_dir).resolve()),
        "--output-root",
        str(Path(args.output_root).resolve()),
        "--models",
        "cpmlp_cpdsconv_fusion",
        "--config-file",
        str(config_file),
        "--fixed-len",
        str(args.fixed_len),
        "--early-cycle",
        str(args.early_cycle),
        "--horizon",
        str(args.horizon),
        "--feature-mode",
        args.feature_mode,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--device",
        args.device,
        "--seeds",
        args.seeds,
        "--split-seed",
        str(args.split_seed),
        "--split-mode",
        args.split_mode,
        "--split-gap",
        str(args.split_gap),
    ]
    if args.limit_trials > 0:
        command.extend(["--limit-trials", str(args.limit_trials)])
    if args.resume:
        command.append("--resume")
    if args.rerun_missing_checkpoints:
        command.append("--rerun-missing-checkpoints")
    if args.dry_run:
        command.append("--dry-run")
    if args.continue_on_error:
        command.append("--continue-on-error")
    return command


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Refine CPMLP+CPDSConv around the best huber050 setting with "
            "multiple seeds."
        )
    )
    parser.add_argument("--tuner", default=script_dir / "tune_delta_hybrids.py")
    parser.add_argument("--compare-script", default=script_dir / "compare_soh_models.py")
    parser.add_argument("--data-dir", default=script_dir / "raw_samples")
    parser.add_argument(
        "--config-file",
        default=script_dir / "cpmlp_cpdsconv_huber_lr_refinement_configs.json",
    )
    parser.add_argument(
        "--output-root",
        default=script_dir / "tuning_outputs_cpmlp_cpdsconv_huber_lr_refinement",
    )
    parser.add_argument("--fixed-len", type=int, default=60)
    parser.add_argument("--early-cycle", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--feature-mode", default="practical")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=["battery", "same-domain-eval", "chronological-within-file", "condition-gap-within-file"],
        default="condition-gap-within-file",
    )
    parser.add_argument("--split-gap", type=int, default=5)
    parser.add_argument("--limit-trials", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-missing-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print("Running CPMLP+CPDSConv huber/lr refinement command:")
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
