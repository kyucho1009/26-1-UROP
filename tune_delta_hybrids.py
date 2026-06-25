from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


DEFAULT_MODELS = "cpmlp_cpgru_fusion,cpmlp_dsconv_fusion"


BASE_CONFIG = {
    "mlp_embed_dim": 64,
    "gru_embed_dim": 64,
    "model_hidden": 256,
    "gru_hidden": 64,
    "dsconv_channels": 64,
    "dropout": 0.10,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "huber_delta": 0.02,
    "clip_grad_norm": 1.0,
    "lr_scheduler_patience": 0,
    "lr_scheduler_factor": 0.5,
    "zero_output_init": False,
    "target_scale": 1.0,
}


CONFIG_VALUE_KEYS = tuple(BASE_CONFIG.keys())


def _cfg(name: str, **updates: float | int) -> dict:
    config = {"name": name, **BASE_CONFIG}
    config.update(updates)
    return config


def cpgru_fusion_configs(preset: str) -> list[dict]:
    configs = [
        _cfg("base"),
        _cfg("drop005", dropout=0.05),
        _cfg("drop020", dropout=0.20),
        _cfg("drop030", dropout=0.30),
        _cfg("lr5e4", lr=5e-4),
        _cfg("lr3e4", lr=3e-4),
        _cfg("wd1e4", weight_decay=1e-4),
        _cfg("mlp96", mlp_embed_dim=96),
        _cfg("mlp128", mlp_embed_dim=128, model_hidden=384, dropout=0.10),
        _cfg("gru96", gru_embed_dim=96, gru_hidden=96),
        _cfg("gru128", gru_embed_dim=128, gru_hidden=128, model_hidden=384, dropout=0.10),
        _cfg("hidden384", model_hidden=384),
        _cfg("wide_balanced", mlp_embed_dim=96, gru_embed_dim=96, model_hidden=384, gru_hidden=96, dropout=0.10),
        _cfg("wide_lowdrop", mlp_embed_dim=96, gru_embed_dim=96, model_hidden=384, gru_hidden=96, dropout=0.05, lr=5e-4),
        _cfg("compact_regularized", model_hidden=192, dropout=0.20, weight_decay=1e-4),
        _cfg("compact_low_lr", mlp_embed_dim=64, gru_embed_dim=48, model_hidden=192, gru_hidden=48, dropout=0.15, lr=5e-4),
        _cfg("wide_regularized", mlp_embed_dim=96, gru_embed_dim=96, model_hidden=384, gru_hidden=96, dropout=0.20, lr=5e-4, weight_decay=1e-4),
        _cfg("clip05", clip_grad_norm=0.5),
        _cfg("clip20", clip_grad_norm=2.0),
        _cfg("huber005", huber_delta=0.005),
        _cfg("huber010", huber_delta=0.01),
        _cfg("huber050", huber_delta=0.05),
        _cfg("plateau_lr", lr_scheduler_patience=3, lr_scheduler_factor=0.5),
        _cfg("plateau_low_lr", lr=5e-4, lr_scheduler_patience=3, lr_scheduler_factor=0.5),
        _cfg("zero_init", zero_output_init=True),
        _cfg("zero_init_lr5e4", lr=5e-4, zero_output_init=True),
        _cfg("zero_init_regularized", lr=5e-4, dropout=0.15, weight_decay=1e-4, zero_output_init=True),
        _cfg("zero_init_scale10", lr=5e-4, huber_delta=0.1, zero_output_init=True, target_scale=10.0),
        _cfg("zero_init_scale20", lr=3e-4, huber_delta=0.2, zero_output_init=True, target_scale=20.0),
    ]
    if preset == "smoke":
        return configs[:1]
    if preset == "quick":
        quick_names = {"base", "lr5e4", "compact_low_lr", "wide_lowdrop", "huber005", "zero_init_lr5e4", "zero_init_scale10"}
        return [config for config in configs if config["name"] in quick_names]
    return configs


def dsconv_fusion_configs(preset: str) -> list[dict]:
    configs = [
        _cfg("base"),
        _cfg("drop005", dropout=0.05),
        _cfg("drop020", dropout=0.20),
        _cfg("drop030", dropout=0.30),
        _cfg("lr5e4", lr=5e-4),
        _cfg("lr3e4", lr=3e-4),
        _cfg("wd1e4", weight_decay=1e-4),
        _cfg("mlp96", mlp_embed_dim=96),
        _cfg("mlp128", mlp_embed_dim=128, model_hidden=384, dropout=0.10),
        _cfg("gru48", gru_hidden=48),
        _cfg("gru96", gru_hidden=96),
        _cfg("dsconv96", dsconv_channels=96),
        _cfg("dsconv32", dsconv_channels=32),
        _cfg("dsconv128", dsconv_channels=128, dropout=0.15, lr=5e-4),
        _cfg("hidden384", model_hidden=384),
        _cfg("wide_balanced", mlp_embed_dim=96, model_hidden=384, gru_hidden=96, dsconv_channels=96, dropout=0.10),
        _cfg("wide_lowdrop", mlp_embed_dim=96, model_hidden=384, gru_hidden=96, dsconv_channels=96, dropout=0.05, lr=5e-4),
        _cfg("regularized_slow", dropout=0.20, lr=5e-4, weight_decay=1e-4),
        _cfg("stable_small_branch", mlp_embed_dim=64, model_hidden=192, gru_hidden=48, dsconv_channels=32, dropout=0.20, lr=5e-4, weight_decay=1e-4),
        _cfg("stable_low_lr", mlp_embed_dim=64, model_hidden=192, gru_hidden=48, dsconv_channels=48, dropout=0.20, lr=3e-4, weight_decay=1e-4, clip_grad_norm=0.5),
        _cfg("mlp_strong_small_dsconv", mlp_embed_dim=96, model_hidden=384, gru_hidden=48, dsconv_channels=48, dropout=0.15, lr=5e-4, weight_decay=1e-4),
        _cfg("clip05", clip_grad_norm=0.5),
        _cfg("clip20", clip_grad_norm=2.0),
        _cfg("huber005", huber_delta=0.005),
        _cfg("huber010", huber_delta=0.01),
        _cfg("huber050", huber_delta=0.05),
        _cfg("plateau_lr", lr_scheduler_patience=3, lr_scheduler_factor=0.5),
        _cfg("plateau_low_lr", lr=5e-4, lr_scheduler_patience=3, lr_scheduler_factor=0.5),
        _cfg("zero_init", zero_output_init=True),
        _cfg("zero_init_lr5e4", lr=5e-4, zero_output_init=True),
        _cfg("zero_init_stable", mlp_embed_dim=64, model_hidden=192, gru_hidden=48, dsconv_channels=48, dropout=0.20, lr=5e-4, weight_decay=1e-4, clip_grad_norm=0.5, zero_output_init=True),
        _cfg("zero_init_scale10", lr=5e-4, huber_delta=0.1, zero_output_init=True, target_scale=10.0),
        _cfg("zero_init_scale20", lr=3e-4, huber_delta=0.2, zero_output_init=True, target_scale=20.0),
    ]
    if preset == "smoke":
        return configs[:1]
    if preset == "quick":
        quick_names = {"base", "lr5e4", "dsconv32", "stable_small_branch", "mlp_strong_small_dsconv", "zero_init_lr5e4", "zero_init_scale10"}
        return [config for config in configs if config["name"] in quick_names]
    return configs


def cpdsconv_fusion_configs(preset: str) -> list[dict]:
    configs = [
        _cfg("base"),
        _cfg("drop005", dropout=0.05),
        _cfg("drop015", dropout=0.15),
        _cfg("drop020", dropout=0.20),
        _cfg("lr5e4", lr=5e-4),
        _cfg("lr3e4", lr=3e-4),
        _cfg("lr7e4", lr=7e-4),
        _cfg("wd1e4", weight_decay=1e-4),
        _cfg("wd3e4", weight_decay=3e-4),
        _cfg("mlp96", mlp_embed_dim=96, model_hidden=256),
        _cfg("mlp128_hidden384", mlp_embed_dim=128, model_hidden=384, dropout=0.10),
        _cfg("hidden384", model_hidden=384),
        _cfg("hidden512", model_hidden=512, dropout=0.15, lr=5e-4),
        _cfg("dsconv32", dsconv_channels=32),
        _cfg("dsconv48", dsconv_channels=48),
        _cfg("dsconv96", dsconv_channels=96),
        _cfg("dsconv128", dsconv_channels=128, dropout=0.15, lr=5e-4),
        _cfg("wide_balanced", mlp_embed_dim=96, model_hidden=384, dsconv_channels=96, dropout=0.10, lr=5e-4),
        _cfg("wide_lowdrop", mlp_embed_dim=96, model_hidden=384, dsconv_channels=96, dropout=0.05, lr=5e-4),
        _cfg("wide_regularized", mlp_embed_dim=96, model_hidden=384, dsconv_channels=96, dropout=0.20, lr=5e-4, weight_decay=1e-4),
        _cfg("mlp_strong_dsconv48", mlp_embed_dim=128, model_hidden=384, dsconv_channels=48, dropout=0.15, lr=5e-4, weight_decay=1e-4),
        _cfg("dsconv_strong_mlp64", mlp_embed_dim=64, model_hidden=256, dsconv_channels=128, dropout=0.15, lr=5e-4),
        _cfg("compact_regularized", mlp_embed_dim=64, model_hidden=192, dsconv_channels=48, dropout=0.20, lr=5e-4, weight_decay=1e-4),
        _cfg("compact_low_lr", mlp_embed_dim=64, model_hidden=192, dsconv_channels=48, dropout=0.15, lr=3e-4, weight_decay=1e-4),
        _cfg("clip05", clip_grad_norm=0.5),
        _cfg("clip20", clip_grad_norm=2.0),
        _cfg("huber005", huber_delta=0.005),
        _cfg("huber010", huber_delta=0.01),
        _cfg("huber050", huber_delta=0.05),
        _cfg("plateau_lr", lr_scheduler_patience=3, lr_scheduler_factor=0.5),
        _cfg("plateau_low_lr", lr=5e-4, lr_scheduler_patience=3, lr_scheduler_factor=0.5),
        _cfg("zero_init", zero_output_init=True),
        _cfg("zero_init_lr5e4", lr=5e-4, zero_output_init=True),
        _cfg("zero_init_stable", mlp_embed_dim=96, model_hidden=384, dsconv_channels=96, dropout=0.15, lr=5e-4, weight_decay=1e-4, zero_output_init=True),
        _cfg("zero_init_scale10", lr=5e-4, huber_delta=0.1, zero_output_init=True, target_scale=10.0),
        _cfg("zero_init_scale20", lr=3e-4, huber_delta=0.2, zero_output_init=True, target_scale=20.0),
    ]
    if preset == "smoke":
        return configs[:1]
    if preset == "quick":
        quick_names = {
            "base",
            "lr5e4",
            "drop015",
            "dsconv96",
            "wide_lowdrop",
            "mlp_strong_dsconv48",
            "compact_low_lr",
            "zero_init_lr5e4",
            "zero_init_scale10",
        }
        return [config for config in configs if config["name"] in quick_names]
    return configs


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def slugify(value: str) -> str:
    value = value.replace(".", "p")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")


def config_for_model(model: str, preset: str) -> list[dict]:
    if model == "cpmlp_cpgru_fusion":
        return cpgru_fusion_configs(preset)
    if model == "cpmlp_dsconv_fusion":
        return dsconv_fusion_configs(preset)
    if model == "cpmlp_cpdsconv_fusion":
        return cpdsconv_fusion_configs(preset)
    raise ValueError(f"unsupported tuning model: {model}")


def normalize_config(raw: dict, fallback_name: str) -> dict:
    config = {"name": str(raw.get("name") or fallback_name), **BASE_CONFIG}
    for key in CONFIG_VALUE_KEYS:
        if key in raw and pd.notna(raw[key]):
            config[key] = raw[key]

    int_keys = {
        "mlp_embed_dim",
        "gru_embed_dim",
        "model_hidden",
        "gru_hidden",
        "dsconv_channels",
        "lr_scheduler_patience",
    }
    bool_keys = {"zero_output_init"}
    for key in int_keys:
        config[key] = int(config[key])
    for key in bool_keys:
        value = config[key]
        if isinstance(value, str):
            config[key] = value.strip().lower() in {"1", "true", "yes", "y"}
        else:
            config[key] = bool(value)
    for key in set(CONFIG_VALUE_KEYS) - int_keys - bool_keys:
        config[key] = float(config[key])
    if config["target_scale"] <= 0:
        raise ValueError("target_scale must be positive")
    return config


def load_config_file(path: str | Path, models: list[str]) -> dict[str, list[dict]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))

    raw_configs: list[dict] = []
    if isinstance(data, list):
        raw_configs = data
    elif isinstance(data, dict) and isinstance(data.get("configs"), list):
        raw_configs = data["configs"]
    elif isinstance(data, dict):
        for model, configs in data.items():
            if not isinstance(configs, list):
                continue
            for config in configs:
                raw_configs.append({"model": model, **config})
    else:
        raise ValueError(f"unsupported config file structure: {config_path}")

    result = {model: [] for model in models}
    for idx, raw in enumerate(raw_configs, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"config entry #{idx} is not an object")
        raw_model = raw.get("model")
        targets = models if raw_model in (None, "", "*") else [str(raw_model)]
        for target in targets:
            if target not in result:
                continue
            result[target].append(normalize_config(raw, fallback_name=f"custom_{idx:03d}"))

    missing = [model for model in models if not result[model]]
    if missing:
        raise ValueError(f"config file did not provide configs for models: {missing}")
    return result


def sort_columns_for_frame(frame: pd.DataFrame, aggregate: bool = False) -> list[str]:
    candidates = (
        ["val_RMSE_mean", "val_MAE_mean", "RMSE_mean", "MAE_mean"]
        if aggregate
        else ["val_RMSE", "val_MAE", "RMSE", "MAE"]
    )
    return [column for column in candidates if column in frame.columns]


def top_configs_by_model(path: str | Path, top_k: int, models: list[str]) -> dict[str, set[str]]:
    if top_k < 1:
        raise ValueError("--top-k-per-model must be >= 1")
    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(f"top-config source not found: {summary_path}")
    summary = pd.read_csv(summary_path)
    required = {"model", "config_name"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"{summary_path} is missing columns: {sorted(missing)}")

    if models:
        summary = summary.loc[summary["model"].isin(models)]
    sort_columns = sort_columns_for_frame(summary, aggregate="RMSE_mean" in summary.columns)
    if not sort_columns:
        raise ValueError(f"{summary_path} needs validation or test RMSE/MAE columns")

    selected = summary.sort_values(sort_columns, ascending=True).groupby("model", group_keys=False).head(top_k)
    result: dict[str, set[str]] = {}
    for model, group in selected.groupby("model"):
        result[str(model)] = set(group["config_name"].astype(str))
    return result


def build_command(args: argparse.Namespace, model: str, seed: int, config: dict, out_dir: Path) -> list[str]:
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
        str(args.early_cycle),
        "--horizon",
        str(args.horizon),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--models",
        model,
        "--target-mode",
        "delta",
        "--feature-mode",
        args.feature_mode,
        "--target-scale",
        str(config["target_scale"]),
        "--seed",
        str(seed),
        "--split-seed",
        str(args.split_seed),
        "--split-mode",
        args.split_mode,
        "--device",
        args.device,
        "--lr",
        str(config["lr"]),
        "--weight-decay",
        str(config["weight_decay"]),
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
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--huber-delta",
        str(config["huber_delta"]),
        "--clip-grad-norm",
        str(config["clip_grad_norm"]),
        "--lr-scheduler-patience",
        str(config["lr_scheduler_patience"]),
        "--lr-scheduler-factor",
        str(config["lr_scheduler_factor"]),
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
    if config.get("zero_output_init"):
        command.append("--zero-output-init")
    return command


def read_trial_metrics(
    out_dir: Path,
    model: str,
    seed: int,
    config: dict,
    elapsed_sec: float,
    patience: int,
    min_delta: float,
    split_seed: int,
) -> dict:
    metrics_path = out_dir / "model_comparison_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics file: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    rows = metrics.loc[metrics["model"] == model]
    if rows.empty:
        raise ValueError(f"metrics file does not contain model={model}: {metrics_path}")
    row = rows.iloc[0].to_dict()
    row.update(
        {
            "seed": seed,
            "split_seed": split_seed,
            "config_name": config["name"],
            "trial_dir": str(out_dir),
            "elapsed_sec": round(elapsed_sec, 3),
            "patience": patience,
            "min_delta": min_delta,
        }
    )
    for key, value in config.items():
        if key != "name":
            row[key] = value
    checkpoint_path = out_dir / "checkpoints" / f"{model}.pt"
    if ("checkpoint_path" not in row or pd.isna(row.get("checkpoint_path"))) and checkpoint_path.exists():
        row["checkpoint_path"] = str(checkpoint_path)
    return row


def write_summaries(rows: list[dict], output_root: Path) -> None:
    if not rows:
        return
    summary = pd.DataFrame(rows)
    key_columns = ["model", "split_seed", "seed", "config_name"] if "split_seed" in summary.columns else ["model", "seed", "config_name"]
    if set(key_columns).issubset(summary.columns):
        summary["_row_order"] = range(len(summary))
        summary["_has_val_metric"] = summary["val_RMSE"].notna() if "val_RMSE" in summary.columns else False
        summary["_has_checkpoint"] = (
            summary["checkpoint_path"].notna() & (summary["checkpoint_path"].astype(str).str.len() > 0)
            if "checkpoint_path" in summary.columns
            else False
        )
        summary = (
            summary.sort_values(key_columns + ["_has_val_metric", "_has_checkpoint", "_row_order"])
            .drop_duplicates(key_columns, keep="last")
            .drop(columns=["_row_order", "_has_val_metric", "_has_checkpoint"])
        )
    sort_columns = sort_columns_for_frame(summary, aggregate=False)
    if sort_columns:
        summary = summary.sort_values(sort_columns, ascending=True)
    summary.to_csv(output_root / "tuning_summary.csv", index=False)

    if "model" in summary.columns and sort_columns:
        best = summary.groupby("model", as_index=False, group_keys=False).head(1)
        best.to_csv(output_root / "best_by_model.csv", index=False)
        best_json = best.astype(object).where(pd.notna(best), None)
        (output_root / "best_by_model.json").write_text(
            json.dumps(best_json.to_dict(orient="records"), indent=2, allow_nan=False),
            encoding="utf-8",
        )

    group_columns = [
        "model",
        "config_name",
        "mlp_embed_dim",
        "gru_embed_dim",
        "model_hidden",
        "gru_hidden",
        "dsconv_channels",
        "dropout",
        "lr",
        "weight_decay",
        "huber_delta",
        "clip_grad_norm",
        "lr_scheduler_patience",
        "lr_scheduler_factor",
        "zero_output_init",
        "target_scale",
        "patience",
        "min_delta",
    ]
    if "split_seed" in summary.columns:
        group_columns.insert(1, "split_seed")
    metric_columns = [
        column
        for column in [
            "val_RMSE",
            "val_MAE",
            "val_MAPE_percent",
            "val_R2",
            "RMSE",
            "MAE",
            "MAPE_percent",
            "R2",
        ]
        if column in summary.columns
    ]
    if set(group_columns).issubset(summary.columns) and metric_columns:
        grouped = (
            summary.groupby(group_columns, dropna=False)[metric_columns]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        grouped.columns = [
            "_".join(str(part) for part in column if part)
            if isinstance(column, tuple)
            else str(column)
            for column in grouped.columns
        ]
        counts = (
            summary.groupby(group_columns, dropna=False)
            .size()
            .reset_index(name="n_trials")
        )
        grouped = grouped.merge(counts, on=group_columns, how="left")
        grouped_sort_columns = sort_columns_for_frame(grouped, aggregate=True)
        grouped = grouped.sort_values(grouped_sort_columns, ascending=True)
        grouped.to_csv(output_root / "tuning_summary_by_config.csv", index=False)

        best_config = grouped.groupby("model", as_index=False, group_keys=False).head(1)
        best_config.to_csv(output_root / "best_config_by_model.csv", index=False)
        best_config_json = best_config.astype(object).where(pd.notna(best_config), None)
        (output_root / "best_config_by_model.json").write_text(
            json.dumps(best_config_json.to_dict(orient="records"), indent=2, allow_nan=False),
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> None:
    args.script = Path(args.script).resolve()
    args.data_dir = Path(args.data_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    models = parse_csv_list(args.models)
    seeds = parse_seed_list(args.seeds)
    config_names = set(parse_csv_list(args.config_names))
    custom_configs = load_config_file(args.config_file, models) if args.config_file else {}
    top_config_names = (
        top_configs_by_model(args.top_configs_from, args.top_k_per_model, models)
        if args.top_configs_from
        else {}
    )

    planned: list[tuple[str, int, dict]] = []
    for model in models:
        model_configs = custom_configs[model] if custom_configs else config_for_model(model, args.preset)
        if top_config_names:
            if model not in top_config_names:
                raise ValueError(f"--top-configs-from did not include model={model}")
            selected_names = top_config_names[model]
        else:
            selected_names = config_names
        if selected_names:
            model_configs = [config for config in model_configs if config["name"] in selected_names]
            if not model_configs:
                raise ValueError(f"no configs matched for model={model}: {sorted(selected_names)}")
        for seed in seeds:
            for config in model_configs:
                planned.append((model, seed, config))

    if args.limit_trials > 0:
        planned = planned[: args.limit_trials]

    plan_path = output_root / "tuning_plan.json"
    plan_path.write_text(
        json.dumps(
            [
                {"model": model, "seed": seed, **config}
                for model, seed, config in planned
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: list[dict] = []
    existing_summary = output_root / "tuning_summary.csv"
    if args.resume and existing_summary.exists():
        rows = pd.read_csv(existing_summary).to_dict(orient="records")
    completed_keys = {
        (
            str(row.get("model")),
            int(row.get("split_seed", args.split_seed)),
            int(row.get("seed")),
            str(row.get("config_name")),
        )
        for row in rows
        if row.get("model") is not None and row.get("seed") is not None and row.get("config_name") is not None
    }

    print(f"planned_trials={len(planned)} output_root={output_root}", flush=True)
    for trial_idx, (model, seed, config) in enumerate(planned, start=1):
        name = slugify(f"{trial_idx:03d}_{model}_{config['name']}_split{args.split_seed}_seed{seed}")
        out_dir = output_root / name
        metrics_path = out_dir / "model_comparison_metrics.csv"
        checkpoint_path = out_dir / "checkpoints" / f"{model}.pt"
        missing_checkpoint = args.rerun_missing_checkpoints and not checkpoint_path.exists()

        if args.resume and metrics_path.exists() and not missing_checkpoint:
            print(f"[skip] {name}", flush=True)
            try:
                key = (model, args.split_seed, seed, config["name"])
                if key not in completed_keys:
                    rows.append(
                        read_trial_metrics(
                            out_dir,
                            model,
                            seed,
                            config,
                            elapsed_sec=0.0,
                            patience=args.patience,
                            min_delta=args.min_delta,
                            split_seed=args.split_seed,
                        )
                    )
                    completed_keys.add(key)
                write_summaries(rows, output_root)
            except Exception as exc:
                print(f"[skip-warning] could not read existing metrics for {name}: {exc}", flush=True)
            continue
        if args.resume and metrics_path.exists() and missing_checkpoint:
            print(f"[rerun] {name} missing checkpoint: {checkpoint_path}", flush=True)

        command = build_command(args, model, seed, config, out_dir)
        print("\n" + "=" * 100, flush=True)
        print(f"[trial {trial_idx}/{len(planned)}] model={model} config={config['name']} seed={seed}", flush=True)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue

        start = time.perf_counter()
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError:
            if args.continue_on_error:
                print(f"[failed] {name}", flush=True)
                continue
            raise
        elapsed = time.perf_counter() - start
        rows.append(
            read_trial_metrics(
                out_dir,
                model,
                seed,
                config,
                elapsed,
                patience=args.patience,
                min_delta=args.min_delta,
                split_seed=args.split_seed,
            )
        )
        completed_keys.add((model, args.split_seed, seed, config["name"]))
        write_summaries(rows, output_root)

    if not args.dry_run:
        write_summaries(rows, output_root)
        print(f"\nsummary: {output_root / 'tuning_summary.csv'}", flush=True)
        print(f"best:    {output_root / 'best_by_model.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default=Path(__file__).with_name("compare_soh_models.py"))
    parser.add_argument("--data-dir", default="raw_samples")
    parser.add_argument("--output-root", default="tuning_outputs_delta_hybrids")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--config-file", default="")
    parser.add_argument("--config-names", default="")
    parser.add_argument("--top-configs-from", default="")
    parser.add_argument("--top-k-per-model", type=int, default=1)
    parser.add_argument("--preset", choices=["smoke", "quick", "focused"], default="focused")
    parser.add_argument("--fixed-len", type=int, default=60)
    parser.add_argument("--early-cycle", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--feature-mode", default="practical")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=["battery", "same-domain-eval", "chronological-within-file", "condition-gap-within-file"],
        default="condition-gap-within-file",
    )
    parser.add_argument("--split-gap", type=int, default=5)
    parser.add_argument("--eval-domain", default="")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--include-regex", default="")
    parser.add_argument("--exclude-regex", default="")
    parser.add_argument("--limit-trials", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-missing-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
