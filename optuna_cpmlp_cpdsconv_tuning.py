from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import optuna
    from optuna.trial import TrialState
except ModuleNotFoundError:
    optuna = None
    TrialState = None

pd = None


MODEL_NAME = "cpmlp_cpdsconv_fusion"
BASELINE_MODEL = "persistence"

BASE_CONFIG = {
    "model": MODEL_NAME,
    "config_name": "optuna_base",
    "target_mode": "delta",
    "feature_mode": "practical",
    "split_mode": "condition-gap-within-file",
    "split_gap": 20,
    "fixed_len": 60,
    "early_cycle": 5,
    "horizon": 50,
    "epochs": 30,
    "batch_size": 128,
    "patience": 8,
    "min_delta": 0.0,
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "mlp_embed_dim": 64,
    "gru_embed_dim": 64,
    "model_hidden": 256,
    "gru_hidden": 64,
    "dsconv_channels": 64,
    "dropout": 0.10,
    "huber_delta": 0.05,
    "clip_grad_norm": 1.0,
    "lr_scheduler_patience": 0,
    "lr_scheduler_factor": 0.5,
    "target_scale": 1.0,
    "zero_output_init": False,
}

COMPARE_NUMERIC_ARGS = {
    "fixed_len": "--fixed-len",
    "early_cycle": "--early-cycle",
    "horizon": "--horizon",
    "epochs": "--epochs",
    "batch_size": "--batch-size",
    "target_scale": "--target-scale",
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
}


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def require_pandas():
    global pd
    if pd is None:
        try:
            import pandas as _pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pandas is required to read and summarize experiment metrics. "
                "Install it with: python -m pip install pandas"
            ) from exc
        pd = _pd
    return pd


def slugify(value: str) -> str:
    value = value.replace(".", "p")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return re.sub(r"_+", "_", value).strip("_").lower() or "item"


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def to_builtin(value: Any) -> Any:
    pandas = require_pandas()
    if pandas.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def mean_std(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return float("inf"), float("inf")
    if len(finite) == 1:
        return finite[0], 0.0
    return statistics.mean(finite), statistics.stdev(finite)


def load_base_config(path: str) -> dict[str, Any]:
    config = dict(BASE_CONFIG)
    if path:
        config_path = Path(path)
        if config_path.exists():
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            config.update(loaded)
        else:
            raise FileNotFoundError(f"base config not found: {config_path}")
    return normalize_config(config)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(BASE_CONFIG)
    normalized.update(config)
    int_keys = {
        "split_gap",
        "fixed_len",
        "early_cycle",
        "horizon",
        "epochs",
        "batch_size",
        "patience",
        "mlp_embed_dim",
        "gru_embed_dim",
        "model_hidden",
        "gru_hidden",
        "dsconv_channels",
        "lr_scheduler_patience",
    }
    bool_keys = {"zero_output_init"}
    float_keys = {
        "min_delta",
        "lr",
        "weight_decay",
        "dropout",
        "huber_delta",
        "clip_grad_norm",
        "lr_scheduler_factor",
        "target_scale",
    }
    for key in int_keys:
        normalized[key] = int(normalized[key])
    for key in float_keys:
        normalized[key] = float(normalized[key])
    for key in bool_keys:
        value = normalized[key]
        if isinstance(value, str):
            normalized[key] = value.strip().lower() in {"1", "true", "yes", "y"}
        else:
            normalized[key] = bool(value)
    if normalized["target_scale"] <= 0:
        raise ValueError("target_scale must be positive")
    normalized["model"] = MODEL_NAME
    normalized["target_mode"] = "delta"
    normalized["feature_mode"] = str(normalized.get("feature_mode", "practical"))
    normalized["split_mode"] = str(normalized.get("split_mode", "condition-gap-within-file"))
    return normalized


def ps_quote(value: str) -> str:
    if not value:
        return "''"
    if re.search(r"\s|'", value):
        return "'" + value.replace("'", "''") + "'"
    return value


def command_to_text(command: list[str]) -> str:
    return " ".join(ps_quote(str(part)) for part in command)


def build_compare_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
    model_seed: int,
    split_seed: int,
    *,
    skip_test_eval: bool,
) -> list[str]:
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
        "delta",
        "--feature-mode",
        str(config["feature_mode"]),
        "--split-mode",
        str(config["split_mode"]),
        "--seed",
        str(model_seed),
        "--split-seed",
        str(split_seed),
        "--device",
        args.device,
    ]
    for key, flag in COMPARE_NUMERIC_ARGS.items():
        command.extend([flag, str(config[key])])
    if config["split_mode"] == "condition-gap-within-file":
        command.extend(["--split-gap", str(config["split_gap"])])
    if args.eval_domain and config["split_mode"] == "same-domain-eval":
        command.extend(["--eval-domain", args.eval_domain])
    if args.max_files > 0:
        command.extend(["--max-files", str(args.max_files)])
    if args.include_regex:
        command.extend(["--include-regex", args.include_regex])
    if args.exclude_regex:
        command.extend(["--exclude-regex", args.exclude_regex])
    if config["zero_output_init"]:
        command.append("--zero-output-init")
    if skip_test_eval:
        command.append("--skip-test-eval")
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


def read_metrics(out_dir: Path, config: dict[str, Any], stage: str, horizon: int, model_seed: int, split_seed: int) -> dict[str, Any]:
    pandas = require_pandas()
    metrics_path = out_dir / "model_comparison_metrics.csv"
    if not valid_metrics_file(metrics_path):
        raise FileNotFoundError(f"missing valid metrics file: {metrics_path}")
    metrics = pandas.read_csv(metrics_path)
    model_row = metrics.loc[metrics["model"].astype(str) == MODEL_NAME].iloc[0].to_dict()
    baseline_row = metrics.loc[metrics["model"].astype(str) == BASELINE_MODEL].iloc[0].to_dict()

    val_rmse = float(model_row["val_RMSE"])
    val_mae = float(model_row["val_MAE"])
    baseline_val_rmse = float(baseline_row["val_RMSE"])
    baseline_val_mae = float(baseline_row["val_MAE"])
    row: dict[str, Any] = {
        "stage": stage,
        "horizon": horizon,
        "model": MODEL_NAME,
        "model_seed": model_seed,
        "split_seed": split_seed,
        "output_dir": str(out_dir),
        "val_RMSE": val_rmse,
        "val_MAE": val_mae,
        "persistence_val_RMSE": baseline_val_rmse,
        "persistence_val_MAE": baseline_val_mae,
        "val_RMSE_improvement_vs_persistence": baseline_val_rmse - val_rmse,
        "val_MAE_improvement_vs_persistence": baseline_val_mae - val_mae,
    }
    for key in ["RMSE", "MAE", "MAPE_percent", "R2", "EOL_Error_cycles"]:
        if key in model_row and pandas.notna(model_row[key]):
            row[f"test_{key}"] = to_builtin(model_row[key])
        if key in baseline_row and pandas.notna(baseline_row[key]):
            row[f"persistence_test_{key}"] = to_builtin(baseline_row[key])
    for key, value in config.items():
        if key not in row:
            row[key] = value
    return row


def summarize_rows(rows: list[dict[str, Any]], std_weight: float, persistence_penalty_weight: float) -> dict[str, Any]:
    val_rmse_mean, val_rmse_std = mean_std([float(row["val_RMSE"]) for row in rows])
    val_mae_mean, val_mae_std = mean_std([float(row["val_MAE"]) for row in rows])
    improvement_mean, improvement_std = mean_std(
        [float(row["val_RMSE_improvement_vs_persistence"]) for row in rows]
    )
    score = val_rmse_mean + std_weight * val_rmse_std
    if improvement_mean < 0:
        score += persistence_penalty_weight * abs(improvement_mean)
    return {
        "score": score,
        "val_RMSE_mean": val_rmse_mean,
        "val_RMSE_std": val_rmse_std,
        "val_MAE_mean": val_mae_mean,
        "val_MAE_std": val_mae_std,
        "val_RMSE_improvement_vs_persistence_mean": improvement_mean,
        "val_RMSE_improvement_vs_persistence_std": improvement_std,
        "n_runs": len(rows),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")


def evaluate_config(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    horizon: int,
    stage: str,
    trial_label: str,
    model_seeds: list[int],
    split_seeds: list[int],
    optuna_trial: Any | None = None,
    skip_test_eval: bool = True,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    config = normalize_config({**config, "horizon": horizon})
    trial_dir = args.output_root / f"horizon_{horizon}" / slugify(stage) / slugify(trial_label)
    trial_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    step = 0
    for split_seed in split_seeds:
        for model_seed in model_seeds:
            run_dir = trial_dir / f"split_{split_seed}_seed_{model_seed}"
            metrics_path = run_dir / "model_comparison_metrics.csv"
            command = build_compare_command(
                args,
                config,
                run_dir,
                model_seed,
                split_seed,
                skip_test_eval=skip_test_eval,
            )
            if args.dry_run:
                print(command_to_text(command), flush=True)
                row = {
                    "stage": stage,
                    "horizon": horizon,
                    "model": MODEL_NAME,
                    "model_seed": model_seed,
                    "split_seed": split_seed,
                    "output_dir": str(run_dir),
                    "val_RMSE": 0.0,
                    "val_MAE": 0.0,
                    "persistence_val_RMSE": 1.0,
                    "persistence_val_MAE": 1.0,
                    "val_RMSE_improvement_vs_persistence": 1.0,
                    "val_MAE_improvement_vs_persistence": 1.0,
                    **config,
                }
            else:
                if args.resume and valid_metrics_file(metrics_path):
                    print(f"[skip] {metrics_path}", flush=True)
                else:
                    print(command_to_text(command), flush=True)
                    start = time.perf_counter()
                    try:
                        subprocess.run(command, check=True)
                    except subprocess.CalledProcessError:
                        if args.continue_on_error:
                            print(f"[failed] {run_dir}", flush=True)
                            continue
                        raise
                    elapsed_sec = time.perf_counter() - start
                    print(f"[done] {run_dir} elapsed_sec={elapsed_sec:.1f}", flush=True)
                row = read_metrics(run_dir, config, stage, horizon, model_seed, split_seed)
            rows.append(row)
            append_jsonl(args.output_root / "optuna_evaluations.jsonl", row)

            if optuna_trial is not None and rows:
                partial = summarize_rows(rows, args.std_weight, args.persistence_penalty_weight)
                optuna_trial.report(float(partial["score"]), step=step)
                step += 1
                if optuna_trial.should_prune():
                    raise optuna.TrialPruned()

    if not rows:
        raise RuntimeError(f"no completed runs for stage={stage} horizon={horizon} trial={trial_label}")

    summary = summarize_rows(rows, args.std_weight, args.persistence_penalty_weight)
    summary.update(
        {
            "stage": stage,
            "horizon": horizon,
            "trial_label": trial_label,
            "config_hash": config_hash(config),
            "config": config,
        }
    )
    append_jsonl(args.output_root / "optuna_trial_summaries.jsonl", summary)
    return float(summary["score"]), summary, rows


def make_study(args: argparse.Namespace, name: str, sampler: Any, direction: str = "minimize") -> Any:
    study_name = f"gap{args.split_gap}_{name}"
    kwargs = {
        "study_name": study_name,
        "sampler": sampler,
        "direction": direction,
        "load_if_exists": args.resume,
    }
    if args.storage:
        kwargs["storage"] = args.storage
    if args.prune:
        kwargs["pruner"] = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    else:
        kwargs["pruner"] = optuna.pruners.NopPruner()
    return optuna.create_study(**kwargs)


def complete_trials(study: Any) -> list[Any]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def best_config_from_study(study: Any) -> dict[str, Any]:
    trials = complete_trials(study)
    if not trials:
        raise RuntimeError(f"study has no completed trials: {study.study_name}")
    trial = min(trials, key=lambda item: float(item.value))
    return json.loads(trial.user_attrs["config_json"])


def run_grid_stage(
    args: argparse.Namespace,
    *,
    horizon: int,
    stage: str,
    base_config: dict[str, Any],
    search_space: dict[str, list[Any]],
    model_seeds: list[int],
    split_seeds: list[int],
) -> tuple[dict[str, Any], Any]:
    sampler = optuna.samplers.GridSampler(search_space)
    study = make_study(args, f"h{horizon}_{stage}", sampler)

    def objective(trial: Any) -> float:
        updates = {key: trial.suggest_categorical(key, values) for key, values in search_space.items()}
        config = normalize_config({**base_config, **updates, "horizon": horizon})
        trial.set_user_attr("config_json", json.dumps(config, sort_keys=True))
        label = f"trial_{trial.number:04d}_{config_hash(config)}"
        score, summary, _ = evaluate_config(
            args,
            config,
            horizon=horizon,
            stage=stage,
            trial_label=label,
            model_seeds=model_seeds,
            split_seeds=split_seeds,
            optuna_trial=trial,
            skip_test_eval=True,
        )
        for key, value in summary.items():
            if key != "config":
                trial.set_user_attr(key, value)
        return score

    n_trials = 1
    for values in search_space.values():
        n_trials *= len(values)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    return best_config_from_study(study), study


def run_capacity_stage(
    args: argparse.Namespace,
    *,
    horizon: int,
    base_config: dict[str, Any],
    model_seeds: list[int],
    split_seeds: list[int],
) -> tuple[dict[str, Any], Any]:
    sampler = optuna.samplers.TPESampler(seed=args.optuna_seed, multivariate=True)
    stage = "capacity"
    study = make_study(args, f"h{horizon}_{stage}", sampler)

    def objective(trial: Any) -> float:
        config = normalize_config(
            {
                **base_config,
                "horizon": horizon,
                "mlp_embed_dim": trial.suggest_categorical("mlp_embed_dim", args.mlp_embed_dim_candidates),
                "model_hidden": trial.suggest_categorical("model_hidden", args.model_hidden_candidates),
                "dsconv_channels": trial.suggest_categorical("dsconv_channels", args.dsconv_channels_candidates),
                "dropout": trial.suggest_categorical("dropout", args.dropout_candidates),
            }
        )
        trial.set_user_attr("config_json", json.dumps(config, sort_keys=True))
        label = f"trial_{trial.number:04d}_{config_hash(config)}"
        score, summary, _ = evaluate_config(
            args,
            config,
            horizon=horizon,
            stage=stage,
            trial_label=label,
            model_seeds=model_seeds,
            split_seeds=split_seeds,
            optuna_trial=trial,
            skip_test_eval=True,
        )
        for key, value in summary.items():
            if key != "config":
                trial.set_user_attr(key, value)
        return score

    study.optimize(objective, n_trials=args.capacity_trials, gc_after_trial=True)
    return best_config_from_study(study), study


def top_configs_from_studies(studies: list[Any], top_k: int) -> list[dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for study in studies:
        for trial in complete_trials(study):
            config_json = trial.user_attrs.get("config_json")
            if config_json:
                candidates.append((float(trial.value), json.loads(config_json)))
    candidates = sorted(candidates, key=lambda item: item[0])
    unique: dict[str, dict[str, Any]] = {}
    for _, config in candidates:
        unique.setdefault(config_hash(config), config)
        if len(unique) >= top_k:
            break
    return list(unique.values())


def confirm_candidates(
    args: argparse.Namespace,
    *,
    horizon: int,
    candidates: list[dict[str, Any]],
    model_seeds: list[int],
    split_seeds: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    pandas = require_pandas()
    summaries: list[dict[str, Any]] = []
    for index, config in enumerate(candidates, start=1):
        config = normalize_config({**config, "horizon": horizon})
        label = f"candidate_{index:02d}_{config_hash(config)}"
        score, summary, _ = evaluate_config(
            args,
            config,
            horizon=horizon,
            stage="confirm_multiseed",
            trial_label=label,
            model_seeds=model_seeds,
            split_seeds=split_seeds,
            optuna_trial=None,
            skip_test_eval=True,
        )
        flat = {key: value for key, value in summary.items() if key != "config"}
        flat.update(config)
        flat["score"] = score
        summaries.append(flat)

    frame = pandas.DataFrame(summaries).sort_values(
        ["score", "val_RMSE_mean", "val_RMSE_std"],
        ascending=True,
    )
    if args.prefer_persistence_win and "val_RMSE_improvement_vs_persistence_mean" in frame.columns:
        winners = frame.loc[frame["val_RMSE_improvement_vs_persistence_mean"] > 0].copy()
        if not winners.empty:
            frame = pandas.concat([winners, frame.loc[~frame.index.isin(winners.index)]], ignore_index=True)
    best_row = frame.iloc[0].to_dict()
    best_config = normalize_config({key: best_row[key] for key in BASE_CONFIG if key in best_row})
    best_config["config_name"] = f"optuna_h{horizon}_{config_hash(best_config)}"
    return best_config, frame


def write_locked_test_artifacts(args: argparse.Namespace, final_configs: dict[int, dict[str, Any]]) -> None:
    commands: list[dict[str, Any]] = []
    ps_lines = [
        "# Run these only after all validation-based choices are fixed.",
        "# They intentionally omit --skip-test-eval.",
    ]
    for horizon, config in sorted(final_configs.items()):
        output_dir = args.output_root / "locked_test" / f"horizon_{horizon}"
        command = build_compare_command(
            args,
            config,
            output_dir,
            args.locked_test_seed,
            args.locked_test_split_seed,
            skip_test_eval=False,
        )
        commands.append({"horizon": horizon, "command": command, "command_text": command_to_text(command)})
        ps_lines.append(command_to_text(command))

    (args.output_root / "locked_test_commands.json").write_text(
        json.dumps(commands, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (args.output_root / "locked_test_commands.ps1").write_text("\n".join(ps_lines) + "\n", encoding="utf-8")


def run_locked_tests(args: argparse.Namespace, final_configs: dict[int, dict[str, Any]]) -> None:
    for horizon, config in sorted(final_configs.items()):
        output_dir = args.output_root / "locked_test" / f"horizon_{horizon}"
        command = build_compare_command(
            args,
            config,
            output_dir,
            args.locked_test_seed,
            args.locked_test_split_seed,
            skip_test_eval=False,
        )
        print(command_to_text(command), flush=True)
        subprocess.run(command, check=True)


def write_final_outputs(args: argparse.Namespace, final_configs: dict[int, dict[str, Any]], confirm_frames: dict[int, pd.DataFrame]) -> None:
    pandas = require_pandas()
    config_payload = {str(horizon): config for horizon, config in sorted(final_configs.items())}
    (args.output_root / "final_optuna_configs_by_horizon.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    for horizon, config in sorted(final_configs.items()):
        (args.output_root / f"final_optuna_config_h{horizon}.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    rows = []
    for horizon, frame in sorted(confirm_frames.items()):
        frame = frame.copy()
        frame["horizon"] = horizon
        frame = frame[["horizon"] + [column for column in frame.columns if column != "horizon"]]
        frame.to_csv(args.output_root / f"confirm_multiseed_h{horizon}.csv", index=False)
        rows.append(frame)
    if rows:
        pandas.concat(rows, ignore_index=True).to_csv(args.output_root / "confirm_multiseed_all_horizons.csv", index=False)
    write_locked_test_artifacts(args, final_configs)


def run(args: argparse.Namespace) -> None:
    if optuna is None:
        raise ModuleNotFoundError(
            "Optuna is required for this runner. Install it with: python -m pip install optuna"
        )
    args.script = Path(args.script).resolve()
    args.data_dir = Path(args.data_dir).resolve()
    args.output_root = Path(args.output_root).resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_base_config(args.base_config)
    base_config.update(
        {
            "feature_mode": args.feature_mode,
            "split_mode": args.split_mode,
            "split_gap": args.split_gap,
            "target_mode": "delta",
            "model": MODEL_NAME,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "min_delta": args.min_delta,
        }
    )
    base_config = normalize_config(base_config)

    horizons = args.horizons
    search_model_seeds = args.search_model_seeds
    search_split_seeds = args.search_split_seeds
    confirm_model_seeds = args.confirm_model_seeds
    confirm_split_seeds = args.confirm_split_seeds

    final_configs: dict[int, dict[str, Any]] = {}
    confirm_frames: dict[int, pd.DataFrame] = {}
    all_plans: list[dict[str, Any]] = []

    for horizon in horizons:
        print("\n" + "=" * 100, flush=True)
        print(f"[horizon {horizon}] start independent scenario optimization", flush=True)
        current = normalize_config({**base_config, "horizon": horizon})
        studies: list[Any] = []

        current, study = run_grid_stage(
            args,
            horizon=horizon,
            stage="early_cycle",
            base_config=current,
            search_space={"early_cycle": args.early_cycle_candidates},
            model_seeds=search_model_seeds,
            split_seeds=search_split_seeds,
        )
        studies.append(study)

        current, study = run_grid_stage(
            args,
            horizon=horizon,
            stage="fixed_len",
            base_config=current,
            search_space={"fixed_len": args.fixed_len_candidates},
            model_seeds=search_model_seeds,
            split_seeds=search_split_seeds,
        )
        studies.append(study)

        current, study = run_grid_stage(
            args,
            horizon=horizon,
            stage="train_params",
            base_config=current,
            search_space={"lr": args.lr_candidates, "huber_delta": args.huber_delta_candidates},
            model_seeds=search_model_seeds,
            split_seeds=search_split_seeds,
        )
        studies.append(study)

        current, study = run_grid_stage(
            args,
            horizon=horizon,
            stage="delta_stability",
            base_config=current,
            search_space={"target_scale": args.target_scale_candidates, "zero_output_init": args.zero_output_init_candidates},
            model_seeds=search_model_seeds,
            split_seeds=search_split_seeds,
        )
        studies.append(study)

        current, study = run_capacity_stage(
            args,
            horizon=horizon,
            base_config=current,
            model_seeds=search_model_seeds,
            split_seeds=search_split_seeds,
        )
        studies.append(study)

        candidates = top_configs_from_studies(studies, args.confirm_top_k)
        if config_hash(current) not in {config_hash(item) for item in candidates}:
            candidates.insert(0, current)
        best_config, confirm_frame = confirm_candidates(
            args,
            horizon=horizon,
            candidates=candidates[: args.confirm_top_k],
            model_seeds=confirm_model_seeds,
            split_seeds=confirm_split_seeds,
        )
        final_configs[horizon] = best_config
        confirm_frames[horizon] = confirm_frame
        all_plans.append(
            {
                "horizon": horizon,
                "best_config": best_config,
                "selection_rule": "score = val_RMSE_mean + std_weight * val_RMSE_std, with persistence penalty if worse than baseline",
                "std_weight": args.std_weight,
                "persistence_penalty_weight": args.persistence_penalty_weight,
            }
        )

    (args.output_root / "optuna_tuning_plan.json").write_text(
        json.dumps(all_plans, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    write_final_outputs(args, final_configs, confirm_frames)
    if args.run_locked_test:
        run_locked_tests(args, final_configs)

    print("\nfinished", flush=True)
    print(f"final configs: {args.output_root / 'final_optuna_configs_by_horizon.json'}", flush=True)
    print(f"locked test commands: {args.output_root / 'locked_test_commands.ps1'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default=Path(__file__).with_name("compare_soh_models.py"))
    parser.add_argument("--data-dir", default="raw_samples")
    parser.add_argument("--output-root", default="optuna_cpmlp_cpdsconv_tuning_gap20")
    parser.add_argument("--base-config", default="final_tuned_cpmlp_cpdsconv_config.json")
    parser.add_argument("--storage", default="", help="Optional Optuna storage URL, e.g. sqlite:///optuna.db")
    parser.add_argument("--feature-mode", default="practical")
    parser.add_argument(
        "--split-mode",
        choices=["battery", "same-domain-eval", "chronological-within-file", "condition-gap-within-file"],
        default="condition-gap-within-file",
    )
    parser.add_argument("--split-gap", type=int, default=20)
    parser.add_argument("--eval-domain", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--include-regex", default="")
    parser.add_argument("--exclude-regex", default="")

    parser.add_argument("--horizons", type=parse_csv_ints, default=parse_csv_ints("10,50,100"))
    parser.add_argument("--early-cycle-candidates", type=parse_csv_ints, default=parse_csv_ints("5,10,20"))
    parser.add_argument("--fixed-len-candidates", type=parse_csv_ints, default=parse_csv_ints("60,100,150"))
    parser.add_argument("--lr-candidates", type=parse_csv_floats, default=parse_csv_floats("0.0003,0.0005,0.0007"))
    parser.add_argument("--huber-delta-candidates", type=parse_csv_floats, default=parse_csv_floats("0.03,0.05,0.08,0.10"))
    parser.add_argument("--target-scale-candidates", type=parse_csv_floats, default=parse_csv_floats("1.0,5.0,10.0"))
    parser.add_argument(
        "--zero-output-init-candidates",
        type=lambda value: [item.strip().lower() in {"1", "true", "yes", "y"} for item in value.split(",") if item.strip()],
        default=[False, True],
    )
    parser.add_argument("--mlp-embed-dim-candidates", type=parse_csv_ints, default=parse_csv_ints("32,64,128"))
    parser.add_argument("--model-hidden-candidates", type=parse_csv_ints, default=parse_csv_ints("128,256,512"))
    parser.add_argument("--dsconv-channels-candidates", type=parse_csv_ints, default=parse_csv_ints("32,64,128"))
    parser.add_argument("--dropout-candidates", type=parse_csv_floats, default=parse_csv_floats("0.05,0.1,0.2"))
    parser.add_argument("--capacity-trials", type=int, default=24)

    parser.add_argument("--search-model-seeds", type=parse_csv_ints, default=parse_csv_ints("42"))
    parser.add_argument("--search-split-seeds", type=parse_csv_ints, default=parse_csv_ints("42"))
    parser.add_argument("--confirm-model-seeds", type=parse_csv_ints, default=parse_csv_ints("42,43,44"))
    parser.add_argument("--confirm-split-seeds", type=parse_csv_ints, default=parse_csv_ints("42,43,44"))
    parser.add_argument("--confirm-top-k", type=int, default=5)
    parser.add_argument("--std-weight", type=float, default=1.0)
    parser.add_argument("--persistence-penalty-weight", type=float, default=1.0)
    parser.add_argument(
        "--allow-no-persistence-win",
        action="store_false",
        dest="prefer_persistence_win",
        help="Do not prefer candidates that beat persistence on validation during final confirmation.",
    )
    parser.set_defaults(prefer_persistence_win=True)
    parser.add_argument("--optuna-seed", type=int, default=42)
    parser.add_argument("--prune", action="store_true")

    parser.add_argument("--locked-test-seed", type=int, default=42)
    parser.add_argument("--locked-test-split-seed", type=int, default=42)
    parser.add_argument("--run-locked-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    args.zero_output_init_candidates = list(dict.fromkeys(args.zero_output_init_candidates))
    if not args.zero_output_init_candidates:
        raise ValueError("--zero-output-init-candidates cannot be empty")
    if args.confirm_top_k <= 0:
        raise ValueError("--confirm-top-k must be positive")
    if args.split_mode == "condition-gap-within-file":
        min_gap = max(args.early_cycle_candidates) - 1
        if args.split_gap < min_gap:
            raise ValueError(
                f"--split-gap={args.split_gap} is smaller than max early_cycle - 1 ({min_gap}). "
                "Increase --split-gap or reduce --early-cycle-candidates to avoid overlapping input windows."
            )
    return args


if __name__ == "__main__":
    run(parse_args())
