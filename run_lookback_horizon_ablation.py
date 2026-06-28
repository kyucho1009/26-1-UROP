from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

pd = None


DEFAULT_SCENARIOS = [
    {
        "name": "short_online_5_to_10",
        "lookback_cycles": 5,
        "horizon": 10,
        "description": "Near-term online SOH update from a short recent history.",
    },
    {
        "name": "short_online_10_to_10",
        "lookback_cycles": 10,
        "horizon": 10,
        "description": "Near-term online SOH update with a richer recent history.",
    },
    {
        "name": "medium_10_to_50",
        "lookback_cycles": 10,
        "horizon": 50,
        "description": "Medium-term degradation forecasting with limited recent history.",
    },
    {
        "name": "medium_20_to_50",
        "lookback_cycles": 20,
        "horizon": 50,
        "description": "Medium-term forecasting with enough recent cycles to estimate trend.",
    },
    {
        "name": "long_20_to_100",
        "lookback_cycles": 20,
        "horizon": 100,
        "description": "Longer-horizon prognosis from a moderate recent history.",
    },
]


DEFAULT_MODEL_CONFIGS: dict[str, dict[str, Any]] = {}


ARG_NAMES = {
    "lr": "--lr",
    "weight_decay": "--weight-decay",
    "mlp_embed_dim": "--mlp-embed-dim",
    "gru_embed_dim": "--gru-embed-dim",
    "model_hidden": "--model-hidden",
    "gru_hidden": "--gru-hidden",
    "dsconv_channels": "--dsconv-channels",
    "dropout": "--dropout",
    "huber_delta": "--huber-delta",
    "clip_grad_norm": "--clip-grad-norm",
    "lr_scheduler_patience": "--lr-scheduler-patience",
    "lr_scheduler_factor": "--lr-scheduler-factor",
    "target_scale": "--target-scale",
}


def require_pandas():
    global pd
    if pd is None:
        try:
            import pandas as _pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pandas is required to aggregate completed ablation results. "
                "Install pandas locally or run this script in Colab."
            ) from exc
        pd = _pd
    return pd


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return re.sub(r"_+", "_", value).strip("_").lower() or "scenario"


def to_abs_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(value)))


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_scenario(raw: dict[str, Any], index: int) -> dict[str, Any]:
    lookback = raw.get("lookback_cycles", raw.get("early_cycle", raw.get("lookback")))
    horizon = raw.get("horizon")
    if lookback is None or horizon is None:
        raise ValueError(f"scenario #{index} must include lookback_cycles and horizon: {raw}")
    scenario = {
        "name": str(raw.get("name") or f"lookback_{lookback}_horizon_{horizon}"),
        "lookback_cycles": int(lookback),
        "horizon": int(horizon),
        "description": str(raw.get("description", "")),
    }
    if scenario["lookback_cycles"] <= 0 or scenario["horizon"] <= 0:
        raise ValueError(f"lookback_cycles and horizon must be positive: {scenario}")
    return scenario


def load_scenarios(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.scenario_file:
        raw = load_json(args.scenario_file)
        if isinstance(raw, dict):
            raw = raw.get("scenarios", raw.get("items", []))
        if not isinstance(raw, list):
            raise ValueError("--scenario-file must contain a list or a dict with a scenarios list")
        scenarios = [normalize_scenario(item, idx + 1) for idx, item in enumerate(raw)]
    elif args.scenarios:
        scenarios = []
        for idx, item in enumerate(parse_csv_list(args.scenarios), start=1):
            parts = item.split(":")
            if len(parts) not in (2, 3):
                raise ValueError("--scenarios entries must look like lookback:horizon or lookback:horizon:name")
            lookback, horizon = int(parts[0]), int(parts[1])
            name = parts[2] if len(parts) == 3 else f"lookback_{lookback}_horizon_{horizon}"
            scenarios.append(normalize_scenario({"name": name, "lookback_cycles": lookback, "horizon": horizon}, idx))
    elif args.lookbacks or args.horizons:
        if not args.lookbacks or not args.horizons:
            raise ValueError("--lookbacks and --horizons must be provided together for grid search")
        lookbacks = parse_int_csv(args.lookbacks)
        horizons = parse_int_csv(args.horizons)
        scenarios = []
        idx = 1
        for lookback in lookbacks:
            for horizon in horizons:
                scenarios.append(
                    normalize_scenario(
                        {
                            "name": f"lookback_{lookback}_horizon_{horizon}",
                            "lookback_cycles": lookback,
                            "horizon": horizon,
                            "description": (
                                f"Grid search scenario: recent {lookback} cycles "
                                f"predict {horizon} cycles ahead."
                            ),
                        },
                        idx,
                    )
                )
                idx += 1
    else:
        scenarios = [normalize_scenario(item, idx + 1) for idx, item in enumerate(DEFAULT_SCENARIOS)]

    seen = set()
    unique = []
    for scenario in scenarios:
        key = (scenario["lookback_cycles"], scenario["horizon"], scenario["name"])
        if key not in seen:
            seen.add(key)
            unique.append(scenario)
    return unique


def load_model_configs(path: str | Path | None) -> dict[str, dict[str, Any]]:
    configs = {model: values.copy() for model, values in DEFAULT_MODEL_CONFIGS.items()}
    if not path:
        return configs
    raw = load_json(path)
    if isinstance(raw, dict) and "models" in raw:
        raw = raw["models"]
    if not isinstance(raw, dict):
        raise ValueError("--model-config-file must contain a dict keyed by model name")
    for model, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"model config for {model} must be a dict")
        configs.setdefault(model, {}).update(values)
    return configs


def valid_metrics_file(path: Path, model: str) -> bool:
    if not path.exists():
        return False
    pandas = require_pandas()
    try:
        metrics = pandas.read_csv(path)
    except Exception:
        return False
    return "model" in metrics.columns and model in set(metrics["model"].astype(str))


def build_command(
    args: argparse.Namespace,
    scenario: dict[str, Any],
    model: str,
    seed: int,
    split_seed: int,
    out_dir: Path,
    model_config: dict[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        str(args.script),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(out_dir),
        "--fixed-len",
        str(args.fixed_len),
        "--early-cycle",
        str(scenario["lookback_cycles"]),
        "--horizon",
        str(scenario["horizon"]),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--models",
        model,
        "--target-mode",
        args.target_mode,
        "--feature-mode",
        args.feature_mode,
        "--split-mode",
        args.split_mode,
        "--seed",
        str(seed),
        "--split-seed",
        str(split_seed),
        "--device",
        args.device,
        "--patience",
        str(model_config.get("patience", args.patience)),
        "--min-delta",
        str(model_config.get("min_delta", args.min_delta)),
    ]
    for key, arg_name in ARG_NAMES.items():
        if key in model_config and model_config[key] is not None:
            command.extend([arg_name, str(model_config[key])])
    if model_config.get("zero_output_init", False):
        command.append("--zero-output-init")
    if args.split_mode == "condition-gap-within-file":
        command.extend(["--split-gap", str(args.split_gap)])
    if args.eval_domain and args.split_mode == "same-domain-eval":
        command.extend(["--eval-domain", args.eval_domain])
    if args.skip_test_eval:
        command.append("--skip-test-eval")
    if args.max_files > 0:
        command.extend(["--max-files", str(args.max_files)])
    if args.include_regex:
        command.extend(["--include-regex", args.include_regex])
    if args.exclude_regex:
        command.extend(["--exclude-regex", args.exclude_regex])
    return command


def read_split_info(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "split_info.json"
    if not path.exists():
        return {}
    try:
        info = load_json(path)
    except Exception:
        return {}
    result: dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        shape = info.get(f"{split}_shape")
        if isinstance(shape, list) and shape:
            result[f"{split}_samples"] = shape[0]
            result[f"{split}_input_shape"] = "x".join(str(part) for part in shape[1:])
    return result


def read_metrics(
    out_dir: Path,
    scenario: dict[str, Any],
    model: str,
    seed: int,
    split_seed: int,
    model_config: dict[str, Any],
) -> pd.DataFrame:
    pandas = require_pandas()
    metrics = pandas.read_csv(out_dir / "model_comparison_metrics.csv")
    metrics = metrics.loc[metrics["model"].astype(str) == model].copy()
    metrics.insert(0, "scenario_name", scenario["name"])
    metrics.insert(1, "lookback_cycles", scenario["lookback_cycles"])
    metrics.insert(2, "horizon", scenario["horizon"])
    metrics.insert(3, "scenario_description", scenario.get("description", ""))
    metrics["seed"] = seed
    metrics["split_seed"] = split_seed
    metrics["trial_dir"] = str(out_dir)
    for key, value in model_config.items():
        if key not in metrics.columns and key in ARG_NAMES:
            metrics[key] = value
    for key in ["config_name", "patience", "min_delta"]:
        if key not in metrics.columns and key in model_config:
            metrics[key] = model_config[key]
    if "zero_output_init" not in metrics.columns:
        metrics["zero_output_init"] = bool(model_config.get("zero_output_init", False))
    split_info = read_split_info(out_dir)
    for key, value in split_info.items():
        metrics[key] = value
    return metrics


def aggregate_gaps(trials: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["scenario_name", "lookback_cycles", "horizon", "split_seed", "seed"]
    metric_cols = [
        column
        for column in ["val_RMSE", "val_MAE", "RMSE", "MAE", "val_MAPE_percent", "MAPE_percent"]
        if column in trials.columns
    ]
    persistence = trials.loc[trials["model"] == "persistence", key_cols + metric_cols].copy()
    if persistence.empty:
        return trials.copy()
    persistence = persistence.rename(columns={column: f"persistence_{column}" for column in metric_cols})
    merged = trials.merge(persistence, on=key_cols, how="left")
    for column in metric_cols:
        baseline_col = f"persistence_{column}"
        if baseline_col in merged.columns:
            merged[f"{column}_gap_vs_persistence"] = merged[column] - merged[baseline_col]
            merged[f"{column}_ratio_vs_persistence"] = merged[column] / merged[baseline_col].replace(0, pd.NA)
    return merged


def summarize(trials: pd.DataFrame, output_root: Path) -> None:
    trials = trials.sort_values(
        [column for column in ["scenario_name", "lookback_cycles", "horizon", "model", "split_seed", "seed"] if column in trials.columns]
    )
    trials.to_csv(output_root / "lookback_horizon_trials.csv", index=False)

    with_gaps = aggregate_gaps(trials)
    with_gaps.to_csv(output_root / "lookback_horizon_persistence_gaps.csv", index=False)

    metric_candidates = [
        "val_RMSE",
        "val_MAE",
        "RMSE",
        "MAE",
        "val_RMSE_gap_vs_persistence",
        "val_MAE_gap_vs_persistence",
        "RMSE_gap_vs_persistence",
        "MAE_gap_vs_persistence",
        "val_RMSE_ratio_vs_persistence",
        "RMSE_ratio_vs_persistence",
        "train_samples",
        "val_samples",
        "test_samples",
    ]
    metric_cols = [column for column in metric_candidates if column in with_gaps.columns]
    group_cols = ["scenario_name", "lookback_cycles", "horizon", "model"]
    grouped = (
        with_gaps.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]
    counts = with_gaps.groupby(group_cols, dropna=False).size().reset_index(name="n_trials")
    grouped = grouped.merge(counts, on=group_cols, how="left")
    sort_cols = [
        column
        for column in ["lookback_cycles", "horizon", "val_RMSE_mean", "val_MAE_mean", "RMSE_mean", "MAE_mean"]
        if column in grouped.columns
    ]
    grouped = grouped.sort_values(sort_cols, ascending=True)
    grouped.to_csv(output_root / "lookback_horizon_summary_by_model.csv", index=False)

    scenario_cols = [
        column
        for column in ["train_samples", "val_samples", "test_samples", "persistence_val_RMSE", "persistence_RMSE"]
        if column in with_gaps.columns
    ]
    scenario_overview = (
        with_gaps.groupby(["scenario_name", "lookback_cycles", "horizon"], dropna=False)[scenario_cols]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    scenario_overview.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in scenario_overview.columns
    ]
    scenario_overview.to_csv(output_root / "lookback_horizon_scenario_overview.csv", index=False)

    best_by_scenario = grouped.sort_values(
        [column for column in ["scenario_name", "val_RMSE_mean", "val_MAE_mean"] if column in grouped.columns]
    ).groupby("scenario_name", as_index=False, group_keys=False).head(1)
    best_by_scenario.to_csv(output_root / "lookback_horizon_best_model_by_scenario.csv", index=False)

    horizon_sort_cols = [
        column
        for column in ["horizon", "val_RMSE_mean", "val_MAE_mean", "RMSE_mean", "MAE_mean"]
        if column in grouped.columns
    ]
    if "horizon" in grouped.columns and "val_RMSE_mean" in grouped.columns:
        selection_by_horizon = grouped.sort_values(horizon_sort_cols, ascending=True).copy()
        selection_by_horizon["selection_rank_within_horizon"] = (
            selection_by_horizon.groupby("horizon", dropna=False).cumcount() + 1
        )
        selection_by_horizon["selection_basis"] = "val_RMSE_mean, then val_MAE_mean; test metrics are report-only"
        selection_by_horizon.to_csv(output_root / "lookback_horizon_selection_by_horizon.csv", index=False)
        best_by_horizon = selection_by_horizon.groupby("horizon", as_index=False, group_keys=False).head(1)
        best_by_horizon.to_csv(output_root / "lookback_horizon_best_model_by_horizon.csv", index=False)
    write_report(output_root, grouped, scenario_overview)


def write_report(output_root: Path, grouped: pd.DataFrame, scenario_overview: pd.DataFrame) -> None:
    lines = [
        "# Lookback-Horizon Ablation Report",
        "",
        "This run compares problem definitions, not just model hyperparameters.",
        "`lookback_cycles` is passed to the existing code as `--early-cycle`.",
        "`horizon` is the number of cycles after the last input cycle to predict.",
        "By default, no model-specific tuned hyperparameter config is applied; pass `--model-config-file` only for an explicit tuned comparison.",
        "Model selection/ranking should use validation metrics first. Test metrics are report-only after choices are fixed.",
        "",
        "Use the tables below to judge whether a scenario is meaningful, stable, and non-trivial relative to persistence.",
        "",
        "## Output Files",
        "",
        "- `lookback_horizon_trials.csv`: every model/seed/split/scenario trial.",
        "- `lookback_horizon_summary_by_model.csv`: mean/std/min/max grouped by scenario and model.",
        "- `lookback_horizon_persistence_gaps.csv`: trial-level gaps against persistence.",
        "- `lookback_horizon_scenario_overview.csv`: sample counts and persistence difficulty by scenario.",
        "- `lookback_horizon_best_model_by_scenario.csv`: diagnostic ranking only; do not treat this as automatic scenario selection.",
        "- `lookback_horizon_selection_by_horizon.csv`: validation-ranked model/lookback candidates within each horizon.",
        "- `lookback_horizon_best_model_by_horizon.csv`: top validation candidate within each horizon; inspect the early-cycle trade-off before treating it as final.",
        "",
    ]
    if not scenario_overview.empty:
        lines.extend(["## Scenario Overview", "", dataframe_to_markdown(scenario_overview), ""])
    preview_cols = [
        column
        for column in [
            "scenario_name",
            "lookback_cycles",
            "horizon",
            "model",
            "val_RMSE_mean",
            "val_MAE_mean",
            "RMSE_mean",
            "MAE_mean",
            "val_RMSE_gap_vs_persistence_mean",
            "n_trials",
        ]
        if column in grouped.columns
    ]
    if preview_cols:
        lines.extend(["## Summary Preview", "", dataframe_to_markdown(grouped[preview_cols]), ""])
    (output_root / "lookback_horizon_report.md").write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for _, row in frame.iterrows():
        body.append("| " + " | ".join(format_markdown_cell(row[column]) for column in frame.columns) + " |")
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
    args.script = to_abs_path(args.script)
    args.data_dir = to_abs_path(args.data_dir)
    output_root = to_abs_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(args)
    if args.limit_scenarios > 0:
        scenarios = scenarios[: args.limit_scenarios]
    if args.split_mode == "condition-gap-within-file" and scenarios:
        max_lookback = max(int(scenario["lookback_cycles"]) for scenario in scenarios)
        if args.split_gap < max_lookback:
            raise ValueError(
                f"--split-gap={args.split_gap} is smaller than the largest lookback/early_cycle ({max_lookback}). "
                "Use --split-gap at least as large as the largest lookback to reduce input-window overlap."
            )
    models = parse_csv_list(args.models)
    split_seeds = parse_int_csv(args.split_seeds)
    seeds = parse_int_csv(args.seeds)
    model_configs = load_model_configs(args.model_config_file)

    plan = []
    for scenario in scenarios:
        for split_seed in split_seeds:
            for seed in seeds:
                for model in models:
                    scenario_dir = f"{scenario['lookback_cycles']:02d}lb_{scenario['horizon']:03d}hz_{slugify(scenario['name'])}"
                    out_dir = output_root / scenario_dir / f"split_{split_seed}" / f"seed_{seed}" / slugify(model)
                    plan.append((scenario, split_seed, seed, model, out_dir))

    (output_root / "lookback_horizon_plan.json").write_text(
        json.dumps(
            [
                {
                    "scenario_name": scenario["name"],
                    "lookback_cycles": scenario["lookback_cycles"],
                    "horizon": scenario["horizon"],
                    "split_seed": split_seed,
                    "seed": seed,
                    "model": model,
                    "output_dir": str(out_dir),
                }
                for scenario, split_seed, seed, model, out_dir in plan
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"planned trials: {len(plan)}", flush=True)
    print(f"output root: {output_root}", flush=True)
    if args.dry_run:
        for idx, (scenario, split_seed, seed, model, out_dir) in enumerate(plan, start=1):
            config = model_configs.get(model, {})
            command = build_command(args, scenario, model, seed, split_seed, out_dir, config)
            print(f"\n[dry-run {idx}/{len(plan)}] {' '.join(command)}", flush=True)
        return

    pandas = require_pandas()
    rows: list[pd.DataFrame] = []
    completed = 0
    for idx, (scenario, split_seed, seed, model, out_dir) in enumerate(plan, start=1):
        metrics_path = out_dir / "model_comparison_metrics.csv"
        config = model_configs.get(model, {})
        if args.resume and valid_metrics_file(metrics_path, model):
            print(f"\n[skip {idx}/{len(plan)}] {metrics_path}", flush=True)
        else:
            command = build_command(args, scenario, model, seed, split_seed, out_dir, config)
            print("\n" + "=" * 100, flush=True)
            print(
                f"[trial {idx}/{len(plan)}] scenario={scenario['name']} "
                f"lookback={scenario['lookback_cycles']} horizon={scenario['horizon']} "
                f"model={model} split_seed={split_seed} seed={seed}",
                flush=True,
            )
            print(" ".join(command), flush=True)
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError:
                if args.continue_on_error:
                    print(f"[error] trial failed but continuing: {out_dir}", flush=True)
                    continue
                raise
        if valid_metrics_file(metrics_path, model):
            rows.append(read_metrics(out_dir, scenario, model, seed, split_seed, config))
            completed += 1

    if rows:
        summarize(pandas.concat(rows, ignore_index=True), output_root)
    print(f"\ncompleted trials with metrics: {completed}/{len(plan)}", flush=True)
    print(f"summary: {output_root / 'lookback_horizon_summary_by_model.csv'}", flush=True)
    print(f"report:  {output_root / 'lookback_horizon_report.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default=Path(__file__).with_name("compare_soh_models.py"))
    parser.add_argument("--data-dir", default="raw_samples")
    parser.add_argument("--output-root", default="lookback_horizon_ablation")
    parser.add_argument("--scenario-file", default="")
    parser.add_argument(
        "--scenarios",
        default="",
        help="Comma-separated lookback:horizon or lookback:horizon:name entries. Ignored when --scenario-file is set.",
    )
    parser.add_argument(
        "--lookbacks",
        default="",
        help="Comma-separated lookback values. Use with --horizons to build every lookback x horizon scenario.",
    )
    parser.add_argument(
        "--horizons",
        default="",
        help="Comma-separated horizon values. Use with --lookbacks to build every lookback x horizon scenario.",
    )
    parser.add_argument(
        "--models",
        default="persistence,cpmlp,cpgru,cpdsconv,cpmlp_cpgru_fusion,cpmlp_dsconv_fusion,cpmlp_cpdsconv_fusion",
    )
    parser.add_argument(
        "--model-config-file",
        default="",
        help="Optional explicit model-specific config. Leave empty for fair comparison from shared defaults.",
    )
    parser.add_argument("--fixed-len", type=int, default=60)
    parser.add_argument("--target-mode", choices=["absolute", "delta"], default="delta")
    parser.add_argument("--feature-mode", default="practical")
    parser.add_argument(
        "--split-mode",
        choices=["battery", "same-domain-eval", "chronological-within-file", "condition-gap-within-file"],
        default="condition-gap-within-file",
    )
    parser.add_argument("--split-gap", type=int, default=20)
    parser.add_argument("--eval-domain", default="")
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--split-seeds", default="42")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--include-regex", default="")
    parser.add_argument("--exclude-regex", default="")
    parser.add_argument("--limit-scenarios", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

