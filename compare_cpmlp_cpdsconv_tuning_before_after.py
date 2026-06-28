from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_NAME = "cpmlp_cpdsconv_fusion"
BASELINE_MODEL = "persistence"

COMPARE_ARGS = {
    "fixed_len": "--fixed-len",
    "early_cycle": "--early-cycle",
    "horizon": "--horizon",
    "epochs": "--epochs",
    "batch_size": "--batch-size",
    "lr": "--lr",
    "weight_decay": "--weight-decay",
    "mlp_embed_dim": "--mlp-embed-dim",
    "gru_embed_dim": "--gru-embed-dim",
    "model_hidden": "--model-hidden",
    "gru_hidden": "--gru-hidden",
    "dsconv_channels": "--dsconv-channels",
    "dropout": "--dropout",
    "patience": "--patience",
    "min_delta": "--min-delta",
    "huber_delta": "--huber-delta",
    "clip_grad_norm": "--clip-grad-norm",
    "lr_scheduler_patience": "--lr-scheduler-patience",
    "lr_scheduler_factor": "--lr-scheduler-factor",
    "target_scale": "--target-scale",
}

CONFIG_COLUMNS = [
    "config_name",
    "early_cycle",
    "fixed_len",
    "split_gap",
    "lr",
    "huber_delta",
    "target_scale",
    "zero_output_init",
    "mlp_embed_dim",
    "model_hidden",
    "dsconv_channels",
    "dropout",
]


def parse_horizons(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_configs(path: Path) -> dict[int, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected horizon-keyed config JSON: {path}")
    configs = {}
    for key, value in raw.items():
        horizon = int(key)
        if not isinstance(value, dict):
            raise ValueError(f"horizon {horizon} config must be an object")
        config = dict(value)
        config["horizon"] = int(config.get("horizon", horizon))
        if config["horizon"] != horizon:
            raise ValueError(f"horizon key {horizon} does not match config horizon {config['horizon']}")
        configs[horizon] = config
    return configs


def ps_quote(value: str) -> str:
    if not value:
        return "''"
    if re.search(r"\s|'", value):
        return "'" + value.replace("'", "''") + "'"
    return value


def command_to_text(command: list[str]) -> str:
    return " ".join(ps_quote(str(part)) for part in command)


def build_command(args: argparse.Namespace, config: dict[str, Any], output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(args.script),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(output_dir),
        "--models",
        f"{BASELINE_MODEL},{MODEL_NAME}",
        "--target-mode",
        str(config.get("target_mode", "delta")),
        "--feature-mode",
        str(config.get("feature_mode", args.feature_mode)),
        "--split-mode",
        str(config.get("split_mode", args.split_mode)),
        "--seed",
        str(config.get("seed", args.seed)),
        "--split-seed",
        str(config.get("split_seed", args.split_seed)),
        "--device",
        args.device,
    ]
    for key, flag in COMPARE_ARGS.items():
        if key in config:
            command.extend([flag, str(config[key])])

    split_mode = str(config.get("split_mode", args.split_mode))
    if split_mode == "condition-gap-within-file":
        command.extend(["--split-gap", str(config.get("split_gap", args.split_gap))])
    if args.eval_domain and split_mode == "same-domain-eval":
        command.extend(["--eval-domain", args.eval_domain])
    if args.max_files > 0:
        command.extend(["--max-files", str(args.max_files)])
    if args.include_regex:
        command.extend(["--include-regex", args.include_regex])
    if args.exclude_regex:
        command.extend(["--exclude-regex", args.exclude_regex])
    if bool(config.get("zero_output_init", False)):
        command.append("--zero-output-init")
    return command


def read_metrics_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["model"]: row for row in rows if row.get("model")}


def valid_metrics(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_metrics_csv(path)
    except Exception:
        return False
    return MODEL_NAME in rows and BASELINE_MODEL in rows


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_delta(before: float | None, after: float | None) -> tuple[float | None, float | None]:
    if before is None or after is None:
        return None, None
    delta = before - after
    percent = None if before == 0 else delta / before * 100.0
    return delta, percent


def read_phase_summary(phase: str, horizon: int, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    rows = read_metrics_csv(output_dir / "model_comparison_metrics.csv")
    model = rows[MODEL_NAME]
    persistence = rows[BASELINE_MODEL]
    result: dict[str, Any] = {
        "phase": phase,
        "horizon": horizon,
        "output_dir": str(output_dir),
        "model_test_RMSE": to_float(model.get("RMSE")),
        "model_test_MAE": to_float(model.get("MAE")),
        "model_val_RMSE": to_float(model.get("val_RMSE")),
        "model_val_MAE": to_float(model.get("val_MAE")),
        "persistence_test_RMSE": to_float(persistence.get("RMSE")),
        "persistence_test_MAE": to_float(persistence.get("MAE")),
        "persistence_val_RMSE": to_float(persistence.get("val_RMSE")),
        "persistence_val_MAE": to_float(persistence.get("val_MAE")),
    }
    for key in CONFIG_COLUMNS:
        if key in config:
            result[key] = config[key]
    rmse_gain, rmse_gain_pct = metric_delta(result["persistence_test_RMSE"], result["model_test_RMSE"])
    mae_gain, mae_gain_pct = metric_delta(result["persistence_test_MAE"], result["model_test_MAE"])
    result["RMSE_improvement_vs_persistence"] = rmse_gain
    result["RMSE_improvement_percent_vs_persistence"] = rmse_gain_pct
    result["MAE_improvement_vs_persistence"] = mae_gain
    result["MAE_improvement_percent_vs_persistence"] = mae_gain_pct
    return result


def run_phase(
    args: argparse.Namespace,
    phase: str,
    horizon: int,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any] | None:
    metrics_path = output_dir / "model_comparison_metrics.csv"
    command = build_command(args, config, output_dir)
    if args.dry_run:
        print(command_to_text(command), flush=True)
        return None
    if valid_metrics(metrics_path) and not args.force:
        print(f"[skip] {phase} horizon={horizon}: {metrics_path}", flush=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run] {phase} horizon={horizon}", flush=True)
        print(command_to_text(command), flush=True)
        subprocess.run(command, check=True)
    return read_phase_summary(phase, horizon, config, output_dir)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_root: Path, rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    lines = [
        "# CPMLP+CPDSConv Tuning Before/After Comparison",
        "",
        "This compares pre-tuning and final tuned configurations under the gap20 split.",
        "Because early_cycle and fixed_len are part of the tuned configuration, before/after sample windows may differ.",
        "",
    ]
    if comparisons:
        lines.extend(["## Test Metric Change", "", markdown_table(comparisons), ""])
    if rows:
        lines.extend(["## Raw Phase Results", "", markdown_table(rows), ""])
    (output_root / "tuning_before_after_report.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def run(args: argparse.Namespace) -> None:
    args.before_config = Path(args.before_config).resolve()
    args.after_config = Path(args.after_config).resolve()
    args.script = Path(args.script).resolve()
    args.data_dir = Path(args.data_dir).resolve()
    args.output_root = Path(args.output_root).resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    before_configs = load_configs(args.before_config)
    after_configs = load_configs(args.after_config)
    horizons = args.horizons or sorted(set(before_configs) & set(after_configs))
    if not horizons:
        raise ValueError("no shared horizons between before and after configs")

    phase_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        if horizon not in before_configs or horizon not in after_configs:
            raise ValueError(f"missing horizon {horizon} in before or after config")
        before_dir = args.output_root / f"horizon_{horizon}" / "before"
        after_dir = args.output_root / f"horizon_{horizon}" / "after"
        before = run_phase(args, "before", horizon, before_configs[horizon], before_dir)
        after = run_phase(args, "after", horizon, after_configs[horizon], after_dir)
        if args.dry_run:
            continue
        if before and after:
            phase_rows.extend([before, after])
            rmse_gain, rmse_gain_pct = metric_delta(before["model_test_RMSE"], after["model_test_RMSE"])
            mae_gain, mae_gain_pct = metric_delta(before["model_test_MAE"], after["model_test_MAE"])
            comparison_rows.append(
                {
                    "horizon": horizon,
                    "before_config": before.get("config_name"),
                    "after_config": after.get("config_name"),
                    "before_test_RMSE": before["model_test_RMSE"],
                    "after_test_RMSE": after["model_test_RMSE"],
                    "RMSE_gain_after_tuning": rmse_gain,
                    "RMSE_gain_percent_after_tuning": rmse_gain_pct,
                    "before_test_MAE": before["model_test_MAE"],
                    "after_test_MAE": after["model_test_MAE"],
                    "MAE_gain_after_tuning": mae_gain,
                    "MAE_gain_percent_after_tuning": mae_gain_pct,
                    "after_RMSE_improvement_percent_vs_persistence": after[
                        "RMSE_improvement_percent_vs_persistence"
                    ],
                    "after_MAE_improvement_percent_vs_persistence": after[
                        "MAE_improvement_percent_vs_persistence"
                    ],
                }
            )

    if not args.dry_run:
        write_csv(args.output_root / "tuning_before_after_raw.csv", phase_rows)
        write_csv(args.output_root / "tuning_before_after_summary.csv", comparison_rows)
        (args.output_root / "tuning_before_after_summary.json").write_text(
            json.dumps(comparison_rows, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        write_report(args.output_root, phase_rows, comparison_rows)
        print(f"summary: {args.output_root / 'tuning_before_after_summary.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-config", default="pre_tuning_cpmlp_cpdsconv_configs_by_horizon.json")
    parser.add_argument("--after-config", default="final_tuned_cpmlp_cpdsconv_configs_by_horizon.json")
    parser.add_argument("--script", default=Path(__file__).with_name("compare_soh_models.py"))
    parser.add_argument("--data-dir", default="raw_samples")
    parser.add_argument("--output-root", default="tuning_before_after_cpmlp_cpdsconv_gap20")
    parser.add_argument("--horizons", type=parse_horizons, default=None)
    parser.add_argument("--feature-mode", default="practical")
    parser.add_argument(
        "--split-mode",
        choices=["battery", "same-domain-eval", "chronological-within-file", "condition-gap-within-file"],
        default="condition-gap-within-file",
    )
    parser.add_argument("--split-gap", type=int, default=20)
    parser.add_argument("--eval-domain", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--include-regex", default="")
    parser.add_argument("--exclude-regex", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
