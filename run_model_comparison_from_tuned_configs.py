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


TARGET_MODEL = "cpmlp_cpdsconv_fusion"
BASELINE_MODEL = "persistence"
REFERENCE_MODEL = "cpmlp"
DEFAULT_MODELS = "persistence,cpmlp,cpdsconv,cpmlp_cpdsconv_fusion"

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


def parse_models(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def build_command(args: argparse.Namespace, config: dict[str, Any], output_dir: Path, models: list[str]) -> list[str]:
    command = [
        sys.executable,
        str(args.script),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(output_dir),
        "--models",
        ",".join(models),
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


def read_metrics_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def valid_metrics(path: Path, models: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_metrics_csv(path)
    except Exception:
        return False
    found = {row.get("model") for row in rows}
    return set(models).issubset(found)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def improvement(reference: float | None, model: float | None) -> tuple[float | None, float | None]:
    if reference is None or model is None:
        return None, None
    delta = reference - model
    percent = None if reference == 0 else delta / reference * 100.0
    return delta, percent


def summarize_horizon(output_dir: Path, horizon: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_metrics_csv(output_dir / "model_comparison_metrics.csv")
    by_model = {row["model"]: row for row in rows}
    persistence = by_model.get(BASELINE_MODEL, {})
    cpmlp = by_model.get(REFERENCE_MODEL, {})
    persistence_rmse = to_float(persistence.get("RMSE"))
    persistence_mae = to_float(persistence.get("MAE"))
    cpmlp_rmse = to_float(cpmlp.get("RMSE"))
    cpmlp_mae = to_float(cpmlp.get("MAE"))

    summary = []
    for row in rows:
        model = row["model"]
        test_rmse = to_float(row.get("RMSE"))
        test_mae = to_float(row.get("MAE"))
        val_rmse = to_float(row.get("val_RMSE"))
        val_mae = to_float(row.get("val_MAE"))
        rmse_vs_persistence, rmse_pct_vs_persistence = improvement(persistence_rmse, test_rmse)
        mae_vs_persistence, mae_pct_vs_persistence = improvement(persistence_mae, test_mae)
        rmse_vs_cpmlp, rmse_pct_vs_cpmlp = improvement(cpmlp_rmse, test_rmse)
        mae_vs_cpmlp, mae_pct_vs_cpmlp = improvement(cpmlp_mae, test_mae)
        item: dict[str, Any] = {
            "horizon": horizon,
            "model": model,
            "test_RMSE": test_rmse,
            "test_MAE": test_mae,
            "val_RMSE": val_rmse,
            "val_MAE": val_mae,
            "RMSE_improvement_vs_persistence": rmse_vs_persistence,
            "RMSE_improvement_percent_vs_persistence": rmse_pct_vs_persistence,
            "MAE_improvement_vs_persistence": mae_vs_persistence,
            "MAE_improvement_percent_vs_persistence": mae_pct_vs_persistence,
            "RMSE_improvement_vs_cpmlp": rmse_vs_cpmlp,
            "RMSE_improvement_percent_vs_cpmlp": rmse_pct_vs_cpmlp,
            "MAE_improvement_vs_cpmlp": mae_vs_cpmlp,
            "MAE_improvement_percent_vs_cpmlp": mae_pct_vs_cpmlp,
            "output_dir": str(output_dir),
        }
        for key in CONFIG_COLUMNS:
            if key in config:
                item[key] = config[key]
        summary.append(item)
    return summary


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


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
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


def write_report(output_root: Path, rows: list[dict[str, Any]], target_model: str) -> None:
    preview_keys = [
        "horizon",
        "model",
        "test_RMSE",
        "test_MAE",
        "RMSE_improvement_percent_vs_persistence",
        "MAE_improvement_percent_vs_persistence",
        "RMSE_improvement_percent_vs_cpmlp",
        "MAE_improvement_percent_vs_cpmlp",
    ]
    preview = [{key: row.get(key) for key in preview_keys} for row in rows]
    target_rows = [row for row in rows if row.get("model") == target_model]
    lines = [
        "# Model Comparison From Tuned Gap20 Configs",
        "",
        "All models are evaluated under each horizon's final tuned problem setting.",
        "Compare models within the same horizon only.",
        "",
    ]
    if target_rows:
        lines.extend([f"## {target_model} Summary", "", markdown_table(target_rows), ""])
    if preview:
        lines.extend(["## All Models", "", markdown_table(preview), ""])
    (output_root / "model_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.config = Path(args.config).resolve()
    args.script = Path(args.script).resolve()
    args.data_dir = Path(args.data_dir).resolve()
    args.output_root = Path(args.output_root).resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    configs = load_configs(args.config)
    models = parse_models(args.models)
    horizons = args.horizons or sorted(configs)
    all_rows: list[dict[str, Any]] = []
    commands = []
    for horizon in horizons:
        if horizon not in configs:
            raise ValueError(f"missing horizon {horizon} in {args.config}")
        config = configs[horizon]
        output_dir = args.output_root / f"horizon_{horizon}"
        metrics_path = output_dir / "model_comparison_metrics.csv"
        command = build_command(args, config, output_dir, models)
        commands.append({"horizon": horizon, "output_dir": str(output_dir), "command_text": command_to_text(command)})
        if args.dry_run:
            print(command_to_text(command), flush=True)
            continue
        if valid_metrics(metrics_path, models) and not args.force:
            print(f"[skip] horizon={horizon}: {metrics_path}", flush=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run] horizon={horizon} models={','.join(models)}", flush=True)
            print(command_to_text(command), flush=True)
            subprocess.run(command, check=True)
        all_rows.extend(summarize_horizon(output_dir, horizon, config))

    (args.output_root / "model_comparison_commands.json").write_text(
        json.dumps(commands, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if not args.dry_run:
        write_csv(args.output_root / "model_comparison_summary.csv", all_rows)
        (args.output_root / "model_comparison_summary.json").write_text(
            json.dumps(all_rows, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        write_report(args.output_root, all_rows, TARGET_MODEL)
        print(f"summary: {args.output_root / 'model_comparison_summary.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="final_tuned_cpmlp_cpdsconv_configs_by_horizon.json")
    parser.add_argument("--script", default=Path(__file__).with_name("compare_soh_models.py"))
    parser.add_argument("--data-dir", default="raw_samples")
    parser.add_argument("--output-root", default="model_comparison_from_tuned_gap20")
    parser.add_argument("--models", default=DEFAULT_MODELS)
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
