from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

pd = None


DEFAULT_MODELS = ",".join(
    [
        "persistence",
        "cpmlp",
        "cpmlp_gru_fusion",
        "cpgru",
        "cpmlp_cpgru_fusion",
        "cpmlp_dsconv_fusion",
        "cpmlp_dsconv_nogru",
        "cpdsconv",
        "cpmlp_cpdsconv_fusion",
        "cpmlp_gru_residual",
        "flatten_mlp",
        "curve_cnn",
        "gru_only",
        "gru_dsconv",
    ]
)
DEFAULT_LOOKBACKS = "5,10,20"
DEFAULT_HORIZONS = "10,50,100"
DEFAULT_SCENARIOS = ",".join(
    f"{lookback}:{horizon}:h{horizon}_lb{lookback}"
    for horizon in [10, 50, 100]
    for lookback in [5, 10, 20]
)
DEFAULT_LIION_EXCLUDE_REGEX = r"^(NA-ion|ZN-coin)"


def require_pandas():
    global pd
    if pd is None:
        try:
            import pandas as _pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pandas is required to aggregate final comparison outputs. "
                "Install pandas on the machine where you run this script."
            ) from exc
        pd = _pd
    return pd


def to_abs_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(value)))


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return re.sub(r"_+", "_", value).strip("_").lower() or "scenario"


def parse_scenarios(value: str) -> list[dict[str, Any]]:
    scenarios = []
    for index, item in enumerate(parse_csv_list(value), start=1):
        parts = item.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("--scenarios entries must look like lookback:horizon or lookback:horizon:name")
        lookback = int(parts[0])
        horizon = int(parts[1])
        name = parts[2] if len(parts) == 3 else f"h{horizon}_lb{lookback}"
        if lookback <= 0 or horizon <= 0:
            raise ValueError(f"lookback and horizon must be positive: {item}")
        scenarios.append(
            {
                "name": name,
                "lookback_cycles": lookback,
                "horizon": horizon,
                "index": index,
            }
        )
    if not scenarios:
        raise ValueError("at least one scenario is required")
    return scenarios


def build_grid_scenarios(lookbacks: str, horizons: str) -> list[dict[str, Any]]:
    scenarios = []
    index = 1
    for horizon in parse_int_csv(horizons):
        for lookback in parse_int_csv(lookbacks):
            scenarios.append(
                {
                    "name": f"h{horizon}_lb{lookback}",
                    "lookback_cycles": lookback,
                    "horizon": horizon,
                    "index": index,
                }
            )
            index += 1
    if not scenarios:
        raise ValueError("--lookbacks and --horizons must produce at least one scenario")
    return scenarios


def load_scenarios(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.lookbacks or args.horizons:
        lookbacks = args.lookbacks or DEFAULT_LOOKBACKS
        horizons = args.horizons or DEFAULT_HORIZONS
        return build_grid_scenarios(lookbacks, horizons)
    return parse_scenarios(args.scenarios)


def ps_quote(value: str) -> str:
    if not value:
        return "''"
    if re.search(r"\s|'", value):
        return "'" + value.replace("'", "''") + "'"
    return value


def command_to_text(command: list[str]) -> str:
    return " ".join(ps_quote(str(part)) for part in command)


def build_command(
    args: argparse.Namespace,
    scenario: dict[str, Any],
    seed: int,
    split_seed: int,
    out_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(args.compare_script),
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
        args.models,
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
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--mlp-embed-dim",
        str(args.mlp_embed_dim),
        "--gru-embed-dim",
        str(args.gru_embed_dim),
        "--model-hidden",
        str(args.model_hidden),
        "--gru-hidden",
        str(args.gru_hidden),
        "--dsconv-channels",
        str(args.dsconv_channels),
        "--dropout",
        str(args.dropout),
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--huber-delta",
        str(args.huber_delta),
        "--clip-grad-norm",
        str(args.clip_grad_norm),
        "--lr-scheduler-patience",
        str(args.lr_scheduler_patience),
        "--lr-scheduler-factor",
        str(args.lr_scheduler_factor),
        "--target-scale",
        str(args.target_scale),
    ]
    if args.split_mode == "condition-gap-within-file":
        command.extend(["--split-gap", str(args.split_gap)])
    if args.eval_domain and args.split_mode == "same-domain-eval":
        command.extend(["--eval-domain", args.eval_domain])
    if args.max_files > 0:
        command.extend(["--max-files", str(args.max_files)])
    if args.include_regex:
        command.extend(["--include-regex", args.include_regex])
    if args.exclude_regex:
        command.extend(["--exclude-regex", args.exclude_regex])
    if args.zero_output_init:
        command.append("--zero-output-init")
    return command


def valid_metrics_file(path: Path, expected_models: list[str]) -> bool:
    if not path.exists():
        return False
    pandas = require_pandas()
    try:
        frame = pandas.read_csv(path)
    except Exception:
        return False
    if "model" not in frame.columns:
        return False
    found = set(frame["model"].astype(str))
    return set(expected_models).issubset(found)


def read_metrics(
    out_dir: Path,
    scenario: dict[str, Any],
    seed: int,
    split_seed: int,
) -> Any:
    pandas = require_pandas()
    metrics_path = out_dir / "model_comparison_metrics.csv"
    frame = pandas.read_csv(metrics_path)
    frame.insert(0, "scenario_name", scenario["name"])
    frame.insert(1, "lookback_cycles", scenario["lookback_cycles"])
    frame.insert(2, "horizon", scenario["horizon"])
    frame["seed"] = seed
    frame["split_seed"] = split_seed
    frame["output_dir"] = str(out_dir)
    frame["metrics_path"] = str(metrics_path)
    frame["selection_problem"] = (
        "horizon="
        + frame["horizon"].astype(str)
        + "|lookback="
        + frame["lookback_cycles"].astype(str)
        + "|split_seed="
        + frame["split_seed"].astype(str)
        + "|seed="
        + frame["seed"].astype(str)
    )
    return frame


def read_group_metrics(
    out_dir: Path,
    scenario: dict[str, Any],
    seed: int,
    split_seed: int,
) -> Any | None:
    group_path = out_dir / "model_group_metrics.csv"
    if not group_path.exists():
        return None
    pandas = require_pandas()
    frame = pandas.read_csv(group_path)
    frame.insert(0, "scenario_name", scenario["name"])
    frame.insert(1, "lookback_cycles", scenario["lookback_cycles"])
    frame.insert(2, "horizon", scenario["horizon"])
    frame["seed"] = seed
    frame["split_seed"] = split_seed
    frame["output_dir"] = str(out_dir)
    return frame


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_or_inf(value: Any) -> float:
    number = finite_or_none(value)
    return number if number is not None else float("inf")


def metric_or_neg_inf(value: Any) -> float:
    number = finite_or_none(value)
    return number if number is not None else float("-inf")


def rank_records(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    exclude_models: set[str],
    rmse_key: str,
    mae_key: str,
    r2_key: str,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("model", "")) in exclude_models:
            continue
        key = tuple(row.get(column) for column in group_keys)
        groups.setdefault(key, []).append(row)

    for key in sorted(groups):
        candidates = sorted(
            groups[key],
            key=lambda row: (
                metric_or_inf(row.get(rmse_key)),
                metric_or_inf(row.get(mae_key)),
                -metric_or_neg_inf(row.get(r2_key)),
                str(row.get("model", "")),
            ),
        )
        for rank, row in enumerate(candidates, start=1):
            item = dict(row)
            item["test_selection_rank"] = rank
            item["selected_final_model"] = rank == 1
            item["selection_basis"] = f"{rmse_key}, then {mae_key}; validation metrics are report-only"
            ranked.append(item)
    return ranked


def flatten_columns(frame: Any) -> Any:
    frame.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


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


def summarize(trials: Any, group_trials: Any | None, output_root: Path, exclude_models: set[str]) -> list[Path]:
    pandas = require_pandas()
    trials = trials.sort_values(["scenario_name", "split_seed", "seed", "model"])
    trials.to_csv(output_root / "final_test_trials.csv", index=False)

    metric_columns = [
        column
        for column in [
            "RMSE",
            "MAE",
            "R2",
            "MAPE_percent",
            "EOL_Error_cycles",
            "val_RMSE",
            "val_MAE",
            "val_R2",
            "val_MAPE_percent",
            "val_EOL_Error_cycles",
        ]
        if column in trials.columns
    ]
    group_columns = ["scenario_name", "lookback_cycles", "horizon", "model"]
    summary = (
        trials.groupby(group_columns, dropna=False)[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary = flatten_columns(summary)
    counts = trials.groupby(group_columns, dropna=False).size().reset_index(name="n_trials")
    summary = summary.merge(counts, on=group_columns, how="left")
    summary = summary.sort_values(
        [column for column in ["horizon", "RMSE_mean", "MAE_mean", "val_RMSE_mean"] if column in summary.columns],
        ascending=True,
    )
    summary.to_csv(output_root / "final_test_model_summary_by_scenario.csv", index=False)

    summary_records = summary.astype(object).where(pandas.notna(summary), None).to_dict(orient="records")
    summary_ranks = rank_records(
        summary_records,
        ["scenario_name", "lookback_cycles", "horizon"],
        exclude_models,
        "RMSE_mean",
        "MAE_mean",
        "R2_mean",
    )
    write_csv(output_root / "final_test_selection_by_scenario.csv", summary_ranks)
    best_summary = [row for row in summary_ranks if row["selected_final_model"]]
    write_csv(output_root / "final_test_best_model_by_scenario.csv", best_summary)
    (output_root / "final_test_best_model_by_scenario.json").write_text(
        json.dumps(best_summary, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )

    trial_records = trials.astype(object).where(pandas.notna(trials), None).to_dict(orient="records")
    trial_ranks = rank_records(
        trial_records,
        ["scenario_name", "lookback_cycles", "horizon", "split_seed", "seed"],
        exclude_models,
        "RMSE",
        "MAE",
        "R2",
    )
    write_csv(output_root / "final_test_trial_ranks.csv", trial_ranks)

    if group_trials is not None and not group_trials.empty:
        group_trials = group_trials.sort_values(
            ["scenario_name", "split_seed", "seed", "model", "split", "group_by", "group_value"]
        )
        group_trials.to_csv(output_root / "final_test_group_metrics.csv", index=False)
        group_metric_columns = [
            column
            for column in ["RMSE", "MAE", "R2", "MAPE_percent", "EOL_Error_cycles", "n_samples", "n_cells"]
            if column in group_trials.columns
        ]
        group_summary_columns = [
            "scenario_name",
            "lookback_cycles",
            "horizon",
            "model",
            "split",
            "group_by",
            "group_value",
        ]
        group_summary = (
            group_trials.groupby(group_summary_columns, dropna=False)[group_metric_columns]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        group_summary = flatten_columns(group_summary)
        group_summary.to_csv(output_root / "final_test_group_summary.csv", index=False)

    write_report(output_root, best_summary, summary_records, exclude_models)
    return [Path(path) for path in trials["metrics_path"].drop_duplicates().tolist()]


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    number = finite_or_none(value)
    if number is not None:
        return f"{number:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    columns = [column for column in columns if any(column in row for row in rows)]
    if not rows or not columns:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def write_report(
    output_root: Path,
    best_summary: list[dict[str, Any]],
    summary_records: list[dict[str, Any]],
    exclude_models: set[str],
) -> None:
    preview_columns = [
        "scenario_name",
        "lookback_cycles",
        "horizon",
        "model",
        "RMSE_mean",
        "MAE_mean",
        "R2_mean",
        "val_RMSE_mean",
        "n_trials",
        "test_selection_rank",
    ]
    ranked_summary = rank_records(
        summary_records,
        ["scenario_name", "lookback_cycles", "horizon"],
        exclude_models,
        "RMSE_mean",
        "MAE_mean",
        "R2_mean",
    )
    lines = [
        "# Final Test Model Comparison",
        "",
        "Selection rule: mean test RMSE, then mean test MAE across the requested seeds and split seeds.",
        "Validation metrics are written for diagnostics only; final ranking uses test metrics after the scenario is fixed.",
        "Default dataset policy is Li-ion only via the non-Li-ion filename exclusion regex.",
        "No tuned/Optuna config file is loaded; all models share the same command-line training defaults unless explicitly overridden.",
        "",
        "## Best Learned Model By Scenario",
        "",
        markdown_table(best_summary, preview_columns),
        "",
        "## Ranked Summary",
        "",
        markdown_table(ranked_summary, preview_columns, limit=80),
        "",
        "## Output Files",
        "",
        "- `final_test_trials.csv`: every scenario/seed/split/model test row.",
        "- `final_test_trial_ranks.csv`: per-scenario, per-seed test ranking.",
        "- `final_test_model_summary_by_scenario.csv`: mean/std/min/max by scenario and model.",
        "- `final_test_selection_by_scenario.csv`: final learned-model ranking by mean test metrics.",
        "- `final_test_group_metrics.csv`: dataset, condition_group, and SOH-band metrics from each run.",
        "- `policy_audit_selection/`: optional split/data-policy audit from `select_final_model_by_test.py`.",
        "",
    ]
    (output_root / "final_test_model_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_policy_audit(args: argparse.Namespace, output_root: Path, metrics_paths: list[Path]) -> None:
    if args.skip_policy_audit or not metrics_paths:
        return
    audit_dir = output_root / "policy_audit_selection"
    command = [
        sys.executable,
        str(args.select_script),
        "--search-root",
        str(args.search_root),
        "--data-dir",
        str(args.data_dir),
        "--metrics-roots",
        ",".join(str(path) for path in metrics_paths),
        "--output-dir",
        str(audit_dir),
        "--exclude-models",
        args.exclude_models_from_final,
        "--exclude-tuned-outputs",
        "--require-multi-model-metrics",
        "--report-limit",
        str(args.report_limit),
    ]
    print("\n[policy-audit] " + command_to_text(command), flush=True)
    subprocess.run(command, check=True)


def run(args: argparse.Namespace) -> None:
    args.search_root = to_abs_path(args.search_root)
    args.compare_script = to_abs_path(args.compare_script)
    args.select_script = to_abs_path(args.select_script)
    args.data_dir = to_abs_path(args.data_dir)
    output_root = to_abs_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(args)
    if args.split_mode == "condition-gap-within-file":
        max_lookback = max(int(item["lookback_cycles"]) for item in scenarios)
        if args.split_gap < max_lookback:
            raise ValueError(
                f"--split-gap={args.split_gap} is smaller than the largest lookback ({max_lookback}). "
                "Use at least 20 for the dataset plan, and at least the largest lookback."
            )
    requested_model_names = parse_csv_list(args.models)
    if not requested_model_names:
        raise ValueError("--models cannot be empty")
    model_names = parse_csv_list(DEFAULT_MODELS) if requested_model_names == ["all"] else requested_model_names
    seeds = parse_int_csv(args.seeds)
    split_seeds = parse_int_csv(args.split_seeds)
    exclude_models = set(parse_csv_list(args.exclude_models_from_final))

    plan = []
    for scenario in scenarios:
        scenario_dir = f"{scenario['lookback_cycles']:02d}lb_{scenario['horizon']:03d}hz_{slugify(scenario['name'])}"
        for split_seed in split_seeds:
            for seed in seeds:
                out_dir = output_root / scenario_dir / f"split_{split_seed}" / f"seed_{seed}"
                command = build_command(args, scenario, seed, split_seed, out_dir)
                plan.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "split_seed": split_seed,
                        "output_dir": out_dir,
                        "command": command,
                    }
                )

    command_rows = [
        {
            "scenario_name": item["scenario"]["name"],
            "lookback_cycles": item["scenario"]["lookback_cycles"],
            "horizon": item["scenario"]["horizon"],
            "seed": item["seed"],
            "split_seed": item["split_seed"],
            "output_dir": str(item["output_dir"]),
            "command_text": command_to_text(item["command"]),
        }
        for item in plan
    ]
    (output_root / "final_test_plan.json").write_text(
        json.dumps(command_rows, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (output_root / "final_test_commands.ps1").write_text(
        "\n".join(row["command_text"] for row in command_rows) + "\n",
        encoding="utf-8",
    )

    print(f"planned comparisons: {len(plan)}", flush=True)
    print(f"output root: {output_root}", flush=True)
    if args.dry_run:
        for index, item in enumerate(plan, start=1):
            print(f"\n[dry-run {index}/{len(plan)}] {command_to_text(item['command'])}", flush=True)
        return

    pandas = require_pandas()
    trial_frames = []
    group_frames = []
    completed = 0
    for index, item in enumerate(plan, start=1):
        out_dir = item["output_dir"]
        metrics_path = out_dir / "model_comparison_metrics.csv"
        if args.resume and valid_metrics_file(metrics_path, model_names):
            print(f"\n[skip {index}/{len(plan)}] {metrics_path}", flush=True)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            print("\n" + "=" * 100, flush=True)
            print(
                f"[comparison {index}/{len(plan)}] scenario={item['scenario']['name']} "
                f"lookback={item['scenario']['lookback_cycles']} horizon={item['scenario']['horizon']} "
                f"split_seed={item['split_seed']} seed={item['seed']}",
                flush=True,
            )
            print(command_to_text(item["command"]), flush=True)
            try:
                subprocess.run(item["command"], check=True)
            except subprocess.CalledProcessError:
                if args.continue_on_error:
                    print(f"[error] comparison failed but continuing: {out_dir}", flush=True)
                    continue
                raise
        if valid_metrics_file(metrics_path, model_names):
            trial_frames.append(read_metrics(out_dir, item["scenario"], item["seed"], item["split_seed"]))
            group_frame = read_group_metrics(out_dir, item["scenario"], item["seed"], item["split_seed"])
            if group_frame is not None:
                group_frames.append(group_frame)
            completed += 1

    if trial_frames:
        group_trials = pandas.concat(group_frames, ignore_index=True) if group_frames else None
        metrics_paths = summarize(pandas.concat(trial_frames, ignore_index=True), group_trials, output_root, exclude_models)
        run_policy_audit(args, output_root, metrics_paths)

    print(f"\ncompleted comparisons with full metrics: {completed}/{len(plan)}", flush=True)
    print(f"summary: {output_root / 'final_test_model_summary_by_scenario.csv'}", flush=True)
    print(f"report:  {output_root / 'final_test_model_comparison_report.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run locked final test comparisons for all implemented SOH model classes, "
            "then rank learned models by test RMSE/MAE under the dataset usage plan."
        )
    )
    parser.add_argument("--search-root", default=".")
    parser.add_argument("--compare-script", default=Path(__file__).with_name("compare_soh_models.py"))
    parser.add_argument("--select-script", default=Path(__file__).with_name("select_final_model_by_test.py"))
    parser.add_argument("--data-dir", default="raw_samples")
    parser.add_argument("--output-root", default="final_test_model_comparison")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument(
        "--scenarios",
        default=DEFAULT_SCENARIOS,
        help=(
            "Comma-separated lookback:horizon:name entries. Default compares "
            "lookback 5/10/20 across horizons 10/50/100 while keeping gap fixed at 20."
        ),
    )
    parser.add_argument(
        "--lookbacks",
        default="",
        help="Optional comma-separated lookback cycles. Use with --horizons to build a grid.",
    )
    parser.add_argument(
        "--horizons",
        default="",
        help="Optional comma-separated horizons. Use with --lookbacks to build a grid.",
    )
    parser.add_argument("--fixed-len", type=int, default=60)
    parser.add_argument("--target-mode", choices=["absolute", "delta"], default="delta")
    parser.add_argument("--feature-mode", default="practical")
    parser.add_argument(
        "--split-mode",
        choices=[
            "battery",
            "condition-group",
            "same-domain-eval",
            "chronological-within-file",
            "condition-gap-within-file",
        ],
        default="condition-group",
    )
    parser.add_argument("--split-gap", type=int, default=20)
    parser.add_argument("--eval-domain", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mlp-embed-dim", type=int, default=64)
    parser.add_argument("--gru-embed-dim", type=int, default=64)
    parser.add_argument("--model-hidden", type=int, default=256)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dsconv-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=0.02)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-patience", type=int, default=0)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--zero-output-init", dest="zero_output_init", action="store_true", default=False)
    parser.add_argument("--no-zero-output-init", dest="zero_output_init", action="store_false")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--split-seeds", default="42")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--include-regex", default="")
    parser.add_argument("--exclude-regex", default=DEFAULT_LIION_EXCLUDE_REGEX)
    parser.add_argument("--exclude-models-from-final", default="persistence")
    parser.add_argument("--report-limit", type=int, default=100)
    parser.add_argument("--skip-policy-audit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
