from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

pd = None


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
    "lr",
    "huber_delta",
    "target_scale",
    "zero_output_init",
    "mlp_embed_dim",
    "model_hidden",
    "dsconv_channels",
    "dropout",
    "weight_decay",
    "batch_size",
    "epochs",
    "patience",
]


def require_pandas():
    global pd
    if pd is None:
        try:
            import pandas as _pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pandas is required to read locked test metrics and write summaries. "
                "Install it with: python -m pip install pandas"
            ) from exc
        pd = _pd
    return pd


def parse_horizons(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def slugify(value: str) -> str:
    value = value.replace(".", "p")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return re.sub(r"_+", "_", value).strip("_").lower() or "item"


def ps_quote(value: str) -> str:
    if not value:
        return "''"
    if re.search(r"\s|'", value):
        return "'" + value.replace("'", "''") + "'"
    return value


def command_to_text(command: list[str]) -> str:
    return " ".join(ps_quote(str(part)) for part in command)


def load_configs(path: Path) -> dict[int, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected horizon-keyed JSON object: {path}")
    configs: dict[int, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"horizon {key} config must be an object")
        horizon = int(key)
        config = dict(value)
        config["horizon"] = int(config.get("horizon", horizon))
        if config["horizon"] != horizon:
            raise ValueError(f"horizon key {horizon} does not match config horizon {config['horizon']}")
        if config.get("model", MODEL_NAME) != MODEL_NAME:
            raise ValueError(f"unsupported model for horizon {horizon}: {config.get('model')}")
        configs[horizon] = config
    return configs


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
        str(args.seed),
        "--split-seed",
        str(args.split_seed),
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


def valid_metrics_file(path: Path) -> bool:
    pandas = require_pandas()
    if not path.exists():
        return False
    try:
        metrics = pandas.read_csv(path)
    except Exception:
        return False
    if "model" not in metrics.columns:
        return False
    models = set(metrics["model"].astype(str))
    return MODEL_NAME in models and BASELINE_MODEL in models


def finite_or_none(value: Any) -> float | None:
    pandas = require_pandas()
    if value is None or pandas.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def improvement(baseline: float | None, model: float | None) -> tuple[float | None, float | None]:
    if baseline is None or model is None:
        return None, None
    delta = baseline - model
    pct = None if baseline == 0 else delta / baseline * 100.0
    return delta, pct


def read_horizon_summary(output_dir: Path, horizon: int, config: dict[str, Any]) -> dict[str, Any]:
    pandas = require_pandas()
    metrics_path = output_dir / "model_comparison_metrics.csv"
    metrics = pandas.read_csv(metrics_path)
    model_row = metrics.loc[metrics["model"].astype(str) == MODEL_NAME].iloc[0].to_dict()
    baseline_row = metrics.loc[metrics["model"].astype(str) == BASELINE_MODEL].iloc[0].to_dict()

    model_rmse = finite_or_none(model_row.get("RMSE"))
    model_mae = finite_or_none(model_row.get("MAE"))
    persistence_rmse = finite_or_none(baseline_row.get("RMSE"))
    persistence_mae = finite_or_none(baseline_row.get("MAE"))
    rmse_gain, rmse_gain_pct = improvement(persistence_rmse, model_rmse)
    mae_gain, mae_gain_pct = improvement(persistence_mae, model_mae)

    row: dict[str, Any] = {
        "horizon": horizon,
        "model": MODEL_NAME,
        "output_dir": str(output_dir),
        "model_test_RMSE": model_rmse,
        "model_test_MAE": model_mae,
        "persistence_test_RMSE": persistence_rmse,
        "persistence_test_MAE": persistence_mae,
        "RMSE_improvement_vs_persistence": rmse_gain,
        "RMSE_improvement_percent_vs_persistence": rmse_gain_pct,
        "MAE_improvement_vs_persistence": mae_gain,
        "MAE_improvement_percent_vs_persistence": mae_gain_pct,
        "model_val_RMSE": finite_or_none(model_row.get("val_RMSE")),
        "model_val_MAE": finite_or_none(model_row.get("val_MAE")),
        "persistence_val_RMSE": finite_or_none(baseline_row.get("val_RMSE")),
        "persistence_val_MAE": finite_or_none(baseline_row.get("val_MAE")),
    }
    for key in CONFIG_COLUMNS:
        if key in config:
            row[key] = config[key]
    return row


def write_commands(output_root: Path, commands: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "locked_test_commands.json").write_text(
        json.dumps(commands, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    lines = [
        "# Locked test commands generated from the gap20 final Optuna config.",
        "# Do not edit hyperparameters after reviewing these test results.",
    ]
    lines.extend(item["command_text"] for item in commands)
    (output_root / "locked_test_commands.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(output_root: Path, summary: pd.DataFrame) -> None:
    preview_cols = [
        "horizon",
        "early_cycle",
        "fixed_len",
        "model_test_RMSE",
        "model_test_MAE",
        "persistence_test_RMSE",
        "persistence_test_MAE",
        "RMSE_improvement_vs_persistence",
        "RMSE_improvement_percent_vs_persistence",
        "MAE_improvement_vs_persistence",
        "MAE_improvement_percent_vs_persistence",
    ]
    preview_cols = [column for column in preview_cols if column in summary.columns]
    lines = [
        "# Locked Test Summary",
        "",
        "These results are from the final fixed configuration. Do not use them for further hyperparameter selection.",
        "",
        "Compare each horizon only against its own persistence baseline.",
        "",
    ]
    if preview_cols:
        lines.extend(["## Summary", "", dataframe_to_markdown(summary[preview_cols]), ""])
    (output_root / "locked_test_summary.md").write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(frame: Any) -> str:
    columns = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        rows.append([format_markdown_cell(row[column]) for column in frame.columns])
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def format_markdown_cell(value: Any) -> str:
    pandas = require_pandas()
    if value is None or pandas.isna(value):
        return ""
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def run(args: argparse.Namespace) -> None:
    args.config = Path(args.config).resolve()
    args.script = Path(args.script).resolve()
    args.data_dir = Path(args.data_dir).resolve()
    args.output_root = Path(args.output_root).resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    configs = load_configs(args.config)
    selected_horizons = args.horizons or sorted(configs)
    missing = sorted(set(selected_horizons) - set(configs))
    if missing:
        raise ValueError(f"config file does not contain horizons: {missing}")

    commands: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for horizon in selected_horizons:
        config = configs[horizon]
        output_dir = args.output_root / f"horizon_{horizon}"
        metrics_path = output_dir / "model_comparison_metrics.csv"
        command = build_command(args, config, output_dir)
        commands.append(
            {
                "horizon": horizon,
                "output_dir": str(output_dir),
                "command": command,
                "command_text": command_to_text(command),
            }
        )

        if args.dry_run:
            print(command_to_text(command), flush=True)
            continue

        if valid_metrics_file(metrics_path):
            if args.force:
                print(f"[rerun] horizon={horizon} existing metrics will be overwritten: {metrics_path}", flush=True)
            else:
                print(f"[skip] horizon={horizon} existing metrics: {metrics_path}", flush=True)
                summary_rows.append(read_horizon_summary(output_dir, horizon, config))
                continue
        elif metrics_path.exists() and not args.force:
            raise RuntimeError(
                f"incomplete metrics file exists for horizon={horizon}: {metrics_path}. "
                "Inspect it, delete the horizon output directory, or rerun with --force."
            )

        print(f"[locked-test] horizon={horizon}", flush=True)
        print(command_to_text(command), flush=True)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            if args.continue_on_error:
                print(f"[failed] horizon={horizon}", flush=True)
                continue
            raise
        summary_rows.append(read_horizon_summary(output_dir, horizon, config))

    write_commands(args.output_root, commands)
    if args.dry_run:
        return

    if summary_rows:
        pandas = require_pandas()
        summary = pandas.DataFrame(summary_rows).sort_values("horizon")
        summary.to_csv(args.output_root / "locked_test_summary.csv", index=False)
        (args.output_root / "locked_test_summary.json").write_text(
            json.dumps(summary.to_dict(orient="records"), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        write_report(args.output_root, summary)
        print(f"summary: {args.output_root / 'locked_test_summary.csv'}", flush=True)
        print(f"report:  {args.output_root / 'locked_test_summary.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="optuna_cpmlp_cpdsconv_tuning_gap20/final_optuna_configs_by_horizon.json")
    parser.add_argument("--script", default=Path(__file__).with_name("compare_soh_models.py"))
    parser.add_argument("--data-dir", default="raw_samples")
    parser.add_argument("--output-root", default="locked_test_cpmlp_cpdsconv_gap20")
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
    parser.add_argument("--force", action="store_true", help="Rerun locked tests even when valid metrics already exist.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
