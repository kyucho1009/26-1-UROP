from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

try:
    import optuna
    from optuna.trial import TrialState
except ModuleNotFoundError:
    optuna = None
    TrialState = None


ROOT = Path(__file__).resolve().parents[1]
STEP7_SCRIPT = ROOT / "scripts" / "run_matr_step7_validation_selection.py"

DEFAULT_TUNE_MODELS = [
    "cpmlp_gru_fusion",
    "cpmlp_cpgru_fusion",
    "cpmlp_cpdsconv_fusion",
]
REFERENCE_MODELS = ["persistence", "cpmlp"]


def parse_int_list(values: Sequence[str] | None, default: Sequence[int]) -> list[int]:
    if not values:
        return [int(item) for item in default]
    parsed: list[int] = []
    for value in values:
        parsed.extend(int(item.strip()) for item in str(value).split(",") if item.strip())
    return parsed


def parse_model_list(values: Sequence[str] | None, default: Sequence[str]) -> list[str]:
    if not values:
        return list(default)
    parsed: list[str] = []
    for value in values:
        parsed.extend(item.strip() for item in str(value).split(",") if item.strip())
    return parsed


def slugify(value: str) -> str:
    value = value.replace(".", "p")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return re.sub(r"_+", "_", value).strip("_").lower() or "item"


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_sanitize(item) for item in value]
    if hasattr(value, "item"):
        return json_sanitize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")


def command_text(command: Sequence[str]) -> str:
    return " ".join(str(item) for item in command)


def suggest_config(trial: Any, args: argparse.Namespace) -> dict[str, Any]:
    size = trial.suggest_categorical("size_preset", args.size_presets)
    if size == "small":
        defaults = {
            "mlp_embed_dim": 32,
            "gru_embed_dim": 32,
            "model_hidden": 128,
            "gru_hidden": 32,
            "dsconv_channels": 32,
        }
    elif size == "medium":
        defaults = {
            "mlp_embed_dim": 32,
            "gru_embed_dim": 64,
            "model_hidden": 128,
            "gru_hidden": 64,
            "dsconv_channels": 64,
        }
    elif size == "large":
        defaults = {
            "mlp_embed_dim": 64,
            "gru_embed_dim": 64,
            "model_hidden": 256,
            "gru_hidden": 64,
            "dsconv_channels": 64,
        }
    else:
        raise ValueError(f"unknown size preset: {size}")

    return {
        "size_preset": size,
        "lr": trial.suggest_float("lr", args.lr_low, args.lr_high, log=True),
        "weight_decay": trial.suggest_float("weight_decay", args.weight_decay_low, args.weight_decay_high, log=True),
        "dropout": trial.suggest_categorical("dropout", args.dropout_candidates),
        "batch_size": trial.suggest_categorical("batch_size", args.batch_size_candidates),
        "epochs": trial.suggest_categorical("epochs", args.epochs_candidates),
        "patience": trial.suggest_categorical("patience", args.patience_candidates),
        **defaults,
    }


def build_step7_command(
    args: argparse.Namespace,
    *,
    model: str,
    config: dict[str, Any],
    output_dir: Path,
    seeds: Sequence[int],
    horizons: Sequence[int],
) -> list[str]:
    models = REFERENCE_MODELS + [model]
    command = [
        sys.executable,
        str(args.step7_script),
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(output_dir),
        "--lookback",
        str(args.lookback),
        "--horizons",
        *[str(item) for item in horizons],
        "--seeds",
        *[str(item) for item in seeds],
        "--models",
        *models,
        "--device",
        args.device,
        "--target-scale",
        str(args.target_scale),
        "--fixed-len",
        str(args.fixed_len),
        "--epochs",
        str(config["epochs"]),
        "--patience",
        str(config["patience"]),
        "--lr",
        str(config["lr"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--batch-size",
        str(config["batch_size"]),
        "--mlp-embed-dim",
        str(config["mlp_embed_dim"]),
        "--gru-embed-dim",
        str(config["gru_embed_dim"]),
        "--model-hidden",
        str(config["model_hidden"]),
        "--gru-hidden",
        str(config["gru_hidden"]),
        "--dsconv-channels",
        str(config["dsconv_channels"]),
        "--dropout",
        str(config["dropout"]),
        "--clip-grad-norm",
        str(args.clip_grad_norm),
    ]
    if args.zero_output_init:
        command.append("--zero-output-init")
    return command


def read_model_summary(output_dir: Path, model: str) -> dict[str, Any]:
    path = output_dir / "model_selection_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing model_selection_summary.csv: {path}")
    df = pd.read_csv(path)
    row = df.loc[df["model"] == model]
    if row.empty:
        raise ValueError(f"model {model!r} not found in {path}")
    return row.iloc[0].to_dict()


def score_from_summary(summary: dict[str, Any], args: argparse.Namespace) -> float:
    mae = float(summary["avg_MAE_mean"])
    rmse = float(summary.get("avg_RMSE_mean", 0.0))
    mape = float(summary.get("avg_MAPE_percent_mean", 0.0))
    std_mae = float(summary.get("std_MAE_mean", 0.0))
    return mae + args.rmse_score_weight * rmse + args.mape_score_weight * (mape / 100.0) + args.std_score_weight * std_mae


def run_step7(
    args: argparse.Namespace,
    *,
    model: str,
    config: dict[str, Any],
    output_dir: Path,
    seeds: Sequence[int],
    horizons: Sequence[int],
) -> dict[str, Any]:
    command = build_step7_command(
        args,
        model=model,
        config=config,
        output_dir=output_dir,
        seeds=seeds,
        horizons=horizons,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "optuna_config.json",
        {
            "model": model,
            "target_scale": args.target_scale,
            "seeds": list(seeds),
            "horizons": list(horizons),
            "config": config,
            "command": command,
            "command_text": command_text(command),
        },
    )

    if args.dry_run:
        print(command_text(command), flush=True)
        return {
            "model": model,
            "avg_MAE_mean": 0.0,
            "avg_RMSE_mean": 0.0,
            "avg_MAPE_percent_mean": 0.0,
            "std_MAE_mean": 0.0,
            "average_MAE_improvement_percent_vs_cpmlp": 0.0,
            "average_Skill_MAE_vs_persistence": 0.0,
            "output_dir": str(output_dir),
        }

    if args.resume and (output_dir / "model_selection_summary.csv").exists():
        print(f"[skip] {output_dir}", flush=True)
    else:
        print(command_text(command), flush=True)
        subprocess.run(command, check=True)
    summary = read_model_summary(output_dir, model)
    summary["output_dir"] = str(output_dir)
    return summary


def set_trial_attrs(trial: Any, payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            trial.set_user_attr(key, json_sanitize(value))


def complete_trials(study: Any) -> list[Any]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def top_trial_payloads(study: Any, top_k: int) -> list[dict[str, Any]]:
    trials = sorted(complete_trials(study), key=lambda trial: float(trial.value))
    payloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trial in trials:
        config_json = trial.user_attrs.get("config_json")
        if not config_json:
            continue
        if config_json in seen:
            continue
        seen.add(config_json)
        payloads.append(
            {
                "trial_number": trial.number,
                "score": float(trial.value),
                "config": json.loads(config_json),
                "summary": json.loads(trial.user_attrs.get("summary_json", "{}")),
            }
        )
        if len(payloads) >= top_k:
            break
    return payloads


def run_model_study(args: argparse.Namespace, model: str, search_seeds: list[int], search_horizons: list[int]) -> Any:
    if optuna is None:
        raise ModuleNotFoundError("optuna is required. Install it with: python -m pip install optuna")

    sampler = optuna.samplers.TPESampler(seed=args.optuna_seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=args.pruner_startup_trials) if args.prune else optuna.pruners.NopPruner()
    study_name = f"matr_step7_{model}_scale{slugify(str(args.target_scale))}"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage or None,
        load_if_exists=args.resume,
    )

    def objective(trial: Any) -> float:
        config = suggest_config(trial, args)
        trial.set_user_attr("config_json", json.dumps(config, sort_keys=True))
        trial_dir = args.output_root / "search" / model / f"trial_{trial.number:04d}"
        summary = run_step7(
            args,
            model=model,
            config=config,
            output_dir=trial_dir,
            seeds=search_seeds,
            horizons=search_horizons,
        )
        score = score_from_summary(summary, args)
        payload = {"model": model, "score": score, **config, **summary}
        trial.set_user_attr("summary_json", json.dumps(json_sanitize(payload), sort_keys=True))
        set_trial_attrs(trial, payload)
        append_jsonl(args.output_root / "optuna_search_trials.jsonl", payload)
        return score

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    return study


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_sanitize(payload), sort_keys=True) + "\n")


def confirm_top_configs(
    args: argparse.Namespace,
    model: str,
    top_payloads: list[dict[str, Any]],
    confirm_seeds: list[int],
    confirm_horizons: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, payload in enumerate(top_payloads, start=1):
        config = payload["config"]
        confirm_dir = args.output_root / "confirm" / model / f"candidate_{index:02d}_trial_{payload['trial_number']:04d}"
        summary = run_step7(
            args,
            model=model,
            config=config,
            output_dir=confirm_dir,
            seeds=confirm_seeds,
            horizons=confirm_horizons,
        )
        score = score_from_summary(summary, args)
        row = {
            "stage": "confirm",
            "model": model,
            "candidate_index": index,
            "source_trial_number": payload["trial_number"],
            "search_score": payload["score"],
            "confirm_score": score,
            **config,
            **summary,
        }
        append_jsonl(args.output_root / "optuna_confirm_candidates.jsonl", row)
        rows.append(row)
    return rows


def write_summary_artifacts(args: argparse.Namespace, search_rows: list[dict[str, Any]], confirm_rows: list[dict[str, Any]]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    if search_rows:
        search_df = pd.DataFrame(search_rows).sort_values(["model", "score", "avg_MAE_mean"])
        search_df.to_csv(args.output_root / "optuna_search_summary.csv", index=False)
    if confirm_rows:
        confirm_df = pd.DataFrame(confirm_rows).sort_values(
            [
                "confirm_score",
                "avg_MAE_mean",
                "avg_RMSE_mean",
                "avg_MAPE_percent_mean",
                "std_MAE_mean",
                "model",
            ],
            ascending=[True, True, True, True, True, True],
        )
        confirm_df.to_csv(args.output_root / "optuna_confirm_summary.csv", index=False)
        best = confirm_df.iloc[0].to_dict()
    elif search_rows:
        search_df = pd.DataFrame(search_rows).sort_values(["score", "avg_MAE_mean", "model"])
        best = search_df.iloc[0].to_dict()
    else:
        best = {}

    save_json(
        args.output_root / "best_optuna_config.json",
        {
            "dataset": "MATR",
            "selection_stage": "post_step7_fusion_hyperparameter_tuning",
            "test_metrics_used": False,
            "target_scale": args.target_scale,
            "lookback_cycles": args.lookback,
            "search_models": args.models,
            "objective": "minimize_avg_validation_MAE_with_optional_small_tie_weights",
            "best": best,
        },
    )
    readme = f"""# MATR Step 7 Fusion Optuna Tuning

This directory was produced by `scripts/tune_matr_step7_fusion_optuna.py`.

- Tuned models: {', '.join(args.models)}
- Target scale: {args.target_scale}
- Test metrics used: false
- Search trials per model: {args.n_trials}
- Confirm top-k per model: {args.confirm_top_k}
- Objective: validation `avg_MAE_mean` from the selected fusion model row.

Each Optuna trial calls `scripts/run_matr_step7_validation_selection.py` with
`persistence`, `cpmlp`, and one fusion model. Baselines are references; the
trial score is computed from the tuned fusion model only.
"""
    (args.output_root / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune top MATR Step 7 fusion models with Optuna.")
    parser.add_argument("--data-root", default="MATR")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/matr_step7_fusion_optuna"))
    parser.add_argument("--step7-script", type=Path, default=STEP7_SCRIPT)
    parser.add_argument("--models", nargs="+", default=DEFAULT_TUNE_MODELS)
    parser.add_argument("--target-scale", type=float, default=100.0)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--fixed-len", type=int, default=100)
    parser.add_argument("--search-horizons", nargs="+", default=["10", "50", "100"])
    parser.add_argument("--search-seeds", nargs="+", default=["42"])
    parser.add_argument("--confirm-horizons", nargs="+", default=["10", "50", "100"])
    parser.add_argument("--confirm-seeds", nargs="+", default=["42", "43", "44"])
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--confirm-top-k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optuna-seed", type=int, default=42)
    parser.add_argument("--storage", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--pruner-startup-trials", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--zero-output-init", action="store_true")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--rmse-score-weight", type=float, default=0.0)
    parser.add_argument("--mape-score-weight", type=float, default=0.0)
    parser.add_argument("--std-score-weight", type=float, default=0.0)
    parser.add_argument("--lr-low", type=float, default=1e-4)
    parser.add_argument("--lr-high", type=float, default=8e-4)
    parser.add_argument("--weight-decay-low", type=float, default=1e-6)
    parser.add_argument("--weight-decay-high", type=float, default=1e-4)
    parser.add_argument("--dropout-candidates", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.35])
    parser.add_argument("--batch-size-candidates", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--epochs-candidates", type=int, nargs="+", default=[80, 100, 120])
    parser.add_argument("--patience-candidates", type=int, nargs="+", default=[12, 15, 20])
    parser.add_argument("--size-presets", nargs="+", default=["small", "medium", "large"])
    args = parser.parse_args()

    args.models = parse_model_list(args.models, DEFAULT_TUNE_MODELS)
    unknown = sorted(set(args.models) - set(DEFAULT_TUNE_MODELS))
    if unknown:
        raise ValueError(f"unsupported tune model(s): {unknown}; allowed={DEFAULT_TUNE_MODELS}")
    args.search_horizons = parse_int_list(args.search_horizons, [10, 50, 100])
    args.search_seeds = parse_int_list(args.search_seeds, [42])
    args.confirm_horizons = parse_int_list(args.confirm_horizons, [10, 50, 100])
    args.confirm_seeds = parse_int_list(args.confirm_seeds, [42, 43, 44])
    args.data_root = Path(args.data_root)
    args.step7_script = Path(args.step7_script)
    if args.target_scale <= 0:
        raise ValueError("--target-scale must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_root / "optuna_tuning_config.json",
        {
            "models": args.models,
            "target_scale": args.target_scale,
            "search_horizons": args.search_horizons,
            "search_seeds": args.search_seeds,
            "confirm_horizons": args.confirm_horizons,
            "confirm_seeds": args.confirm_seeds,
            "n_trials": args.n_trials,
            "confirm_top_k": args.confirm_top_k,
            "device": args.device,
            "search_space": {
                "lr": [args.lr_low, args.lr_high],
                "weight_decay": [args.weight_decay_low, args.weight_decay_high],
                "dropout": args.dropout_candidates,
                "batch_size": args.batch_size_candidates,
                "epochs": args.epochs_candidates,
                "patience": args.patience_candidates,
                "size_presets": args.size_presets,
            },
        },
    )

    if optuna is None and not args.dry_run:
        raise ModuleNotFoundError("optuna is required. Install it with: python -m pip install optuna")

    all_search_rows: list[dict[str, Any]] = []
    all_confirm_rows: list[dict[str, Any]] = []

    if args.dry_run:
        for model in args.models:
            config = {
                "size_preset": "medium",
                "lr": 3e-4,
                "weight_decay": 1e-5,
                "dropout": 0.2,
                "batch_size": 16,
                "epochs": 100,
                "patience": 15,
                "mlp_embed_dim": 32,
                "gru_embed_dim": 64,
                "model_hidden": 128,
                "gru_hidden": 64,
                "dsconv_channels": 64,
            }
            summary = run_step7(
                args,
                model=model,
                config=config,
                output_dir=args.output_root / "dry_run" / model,
                seeds=args.search_seeds,
                horizons=args.search_horizons,
            )
            all_search_rows.append({"model": model, "score": 0.0, **config, **summary})
        write_summary_artifacts(args, all_search_rows, all_confirm_rows)
        return

    studies: dict[str, Any] = {}
    for model in args.models:
        print(f"=== Optuna search: {model} ===", flush=True)
        study = run_model_study(args, model, args.search_seeds, args.search_horizons)
        studies[model] = study
        for trial in complete_trials(study):
            summary = json.loads(trial.user_attrs.get("summary_json", "{}"))
            if summary:
                all_search_rows.append(summary)

    if args.confirm_top_k > 0:
        for model, study in studies.items():
            top_payloads = top_trial_payloads(study, args.confirm_top_k)
            print(f"=== Confirm top {len(top_payloads)}: {model} ===", flush=True)
            all_confirm_rows.extend(
                confirm_top_configs(
                    args,
                    model,
                    top_payloads,
                    args.confirm_seeds,
                    args.confirm_horizons,
                )
            )

    write_summary_artifacts(args, all_search_rows, all_confirm_rows)
    print(f"wrote {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
