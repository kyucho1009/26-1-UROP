from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_matr_step7_validation_selection as step7

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("PyTorch is required for locked test evaluation.") from exc


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_optuna_context(config: dict[str, Any], context: dict[str, Any], context_path: Path) -> dict[str, Any]:
    merged = dict(config)
    passthrough_keys = [
        "target_scale",
        "sample_mode",
        "fixed_len",
        "search_horizons",
        "search_seeds",
        "confirm_horizons",
        "confirm_seeds",
        "search_reference_models",
        "confirm_reference_models",
        "batches",
    ]
    for key in passthrough_keys:
        if merged.get(key) in (None, "", []):
            value = context.get(key)
            if value not in (None, "", []):
                merged[key] = value
    if merged.get("lookback_cycles") in (None, "", []):
        value = context.get("lookback") or context.get("lookback_cycles")
        if value not in (None, "", []):
            merged["lookback_cycles"] = value
    if merged.get("search_models") in (None, "", []):
        value = context.get("models")
        if value not in (None, "", []):
            merged["search_models"] = value
    merged["_optuna_tuning_config_path"] = str(context_path)
    return merged


def load_locked_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = load_json(config_path)
    sidecar_path = config_path.with_name("optuna_tuning_config.json")
    if sidecar_path.exists():
        config = merge_optuna_context(config, load_json(sidecar_path), sidecar_path)
    return config


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(step7.json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")


def unique_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in config:
        return config[key]
    hyper = config.get("hyperparameters", {})
    if key in hyper:
        return hyper[key]
    selected_hyper = config.get("selected_hyperparameters", {})
    if key in selected_hyper:
        return selected_hyper[key]
    best = config.get("best", {})
    if isinstance(best, dict) and key in best:
        return best[key]
    return default


def config_sequence(config: dict[str, Any], keys: Sequence[str], default: Sequence[Any]) -> list[Any]:
    for key in keys:
        value = config.get(key)
        if value:
            return list(value)
    return list(default)


def build_runtime_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    best = config.get("best", {})
    if not isinstance(best, dict):
        best = {}
    hyper = config.get("hyperparameters", {}) or config.get("selected_hyperparameters", {}) or best
    selected_model = str(config.get("selected_model") or best.get("model"))
    if not selected_model or selected_model == "None":
        raise ValueError("locked config must include selected_model")
    if bool(config.get("test_metrics_used", False)):
        raise ValueError("locked config unexpectedly says test metrics were used for selection")
    if best and not (args.sample_mode or config.get("sample_mode")):
        raise ValueError(
            "Optuna best config is missing sample_mode. Use a best_optuna_config.json "
            "created with the updated tuner, keep optuna_tuning_config.json in the same "
            "folder, or pass --sample-mode explicitly."
        )

    models = [selected_model]
    if args.include_references:
        models = ["persistence", "cpmlp", selected_model]
    if args.models:
        models = args.models
    models = unique_preserve_order(models)
    unknown = sorted(set(models) - set(step7.DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"unsupported model(s): {unknown}; allowed={step7.DEFAULT_MODELS}")
    batch_values = args.batches if args.batches is not None else config.get("batches")
    batches = step7.parse_batch_filter(batch_values)

    return {
        "selected_model": selected_model,
        "models": models,
        "batches": batches,
        "batch_filter_applied": bool(batches),
        "lookback": int(args.lookback or config.get("lookback_cycles", 20)),
        "sample_mode": str(args.sample_mode or config.get("sample_mode", step7.SAMPLE_MODE_FIRST_WINDOW)),
        "horizons": [
            int(item)
            for item in (
                args.horizons
                or config_sequence(config, ["horizons", "confirm_horizons", "search_horizons"], [10, 50, 100])
            )
        ],
        "seeds": [
            int(item)
            for item in (
                args.seeds
                or config_sequence(config, ["seeds", "confirm_seeds", "search_seeds"], [42, 43, 44])
            )
        ],
        "target_scale": float(args.target_scale or config_value(config, "target_scale", 1.0)),
        "fixed_len": int(args.fixed_len or config_value(config, "fixed_len", 100)),
        "batch_size": int(args.batch_size or hyper.get("batch_size", 16)),
        "dropout": float(args.dropout if args.dropout is not None else hyper.get("dropout", 0.1)),
        "dsconv_channels": int(args.dsconv_channels or hyper.get("dsconv_channels", 64)),
        "epochs": int(args.epochs or hyper.get("epochs", 100)),
        "gru_embed_dim": int(args.gru_embed_dim or hyper.get("gru_embed_dim", 64)),
        "gru_hidden": int(args.gru_hidden or hyper.get("gru_hidden", 64)),
        "lr": float(args.lr or hyper.get("lr", 3e-4)),
        "mlp_embed_dim": int(args.mlp_embed_dim or hyper.get("mlp_embed_dim", 64)),
        "model_hidden": int(args.model_hidden or hyper.get("model_hidden", 256)),
        "patience": int(args.patience or hyper.get("patience", 12)),
        "weight_decay": float(args.weight_decay or hyper.get("weight_decay", 1e-5)),
        "clip_grad_norm": float(args.clip_grad_norm or hyper.get("clip_grad_norm", 1.0)),
    }


def split_metrics_row(
    *,
    split_name: str,
    seed: int,
    horizon: int,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence_mae: float,
    best_epoch: int,
    checkpoint_path: str,
) -> dict[str, Any]:
    metrics = step7.metric_row_with_skill(y_true, y_pred, persistence_mae)
    return {
        "dataset": step7.DATASET,
        "stage": "locked_test_evaluation",
        "split": split_name,
        "seed": seed,
        "horizon": horizon,
        "model": model,
        "n_samples": int(len(y_true)),
        **metrics,
        "best_epoch": int(best_epoch),
        "checkpoint_path": checkpoint_path,
    }


def persistence_predictions(split: step7.HorizonSplit) -> np.ndarray:
    return split.current_soh.astype(np.float32)


def scaled_delta_loader(
    split: step7.HorizonSplit,
    X_norm: np.ndarray,
    target_scale: float,
    batch_size: int,
    shuffle: bool,
):
    return step7.make_loader(
        X_norm,
        (split.y_delta * target_scale).astype(np.float32),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_model_on_split(
    model,
    split: step7.HorizonSplit,
    X_norm: np.ndarray,
    target_scale: float,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    loader = scaled_delta_loader(split, X_norm, target_scale, batch_size=batch_size, shuffle=False)
    pred_delta_scaled, _ = step7.predict_delta(model, loader, device=device)
    pred_delta = pred_delta_scaled / target_scale
    pred_soh = split.current_soh - pred_delta
    return pred_soh.astype(np.float32), pred_delta.astype(np.float32)


def make_model(name: str, cfg: dict[str, Any]):
    model = step7.model_lib.make_model(
        name,
        early_cycle=cfg["lookback"],
        fixed_len=cfg["fixed_len"],
        mlp_embed_dim=cfg["mlp_embed_dim"],
        gru_embed_dim=cfg["gru_embed_dim"],
        model_hidden=cfg["model_hidden"],
        gru_hidden=cfg["gru_hidden"],
        dsconv_channels=cfg["dsconv_channels"],
        dropout=cfg["dropout"],
    )
    return model


def prediction_frame(
    split: step7.HorizonSplit,
    *,
    seed: int,
    model: str,
    pred_soh: np.ndarray,
    pred_delta: np.ndarray,
) -> pd.DataFrame:
    if len(split.meta) != len(pred_soh) or len(split.meta) != len(pred_delta):
        raise ValueError("prediction count does not match split metadata")
    frame = pd.DataFrame(split.meta)
    frame["stage"] = "locked_test_evaluation"
    frame["seed"] = int(seed)
    frame["model"] = model
    frame["actual_soh"] = split.y_soh_target.astype(np.float64)
    frame["current_soh"] = split.current_soh.astype(np.float64)
    frame["actual_delta_soh"] = split.y_delta.astype(np.float64)
    frame["pred_delta_soh"] = np.asarray(pred_delta, dtype=np.float64)
    frame["pred_soh"] = np.asarray(pred_soh, dtype=np.float64)
    frame["abs_error"] = np.abs(frame["actual_soh"] - frame["pred_soh"])
    frame["squared_error"] = (frame["actual_soh"] - frame["pred_soh"]) ** 2
    return frame


def load_source_split_manifest(
    manifest_root: Path,
    seed: int,
    records_by_id: dict[str, step7.BatteryRecord],
    cfg: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any], Path]:
    path = manifest_root / f"split_manifest_seed{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing source split manifest: {path}")
    manifest = load_json(path)
    if int(manifest.get("seed", -1)) != int(seed):
        raise ValueError(f"split manifest seed mismatch in {path}")
    if manifest.get("dataset") != step7.DATASET:
        raise ValueError(f"split manifest is not a {step7.DATASET} manifest: {path}")
    if int(manifest.get("lookback_cycles", -1)) != int(cfg["lookback"]):
        raise ValueError(f"lookback mismatch between runtime config and {path}")
    if manifest.get("sample_mode") != cfg["sample_mode"]:
        raise ValueError(f"sample_mode mismatch between runtime config and {path}")
    manifest_horizons = {int(item) for item in manifest.get("horizons", [])}
    missing_horizons = sorted(set(cfg["horizons"]) - manifest_horizons)
    if missing_horizons:
        raise ValueError(f"source split manifest {path} does not cover horizons {missing_horizons}")

    split_ids = {
        "train": [str(item) for item in manifest.get("train_battery_ids", [])],
        "validation": [str(item) for item in manifest.get("validation_battery_ids", [])],
        "test": [str(item) for item in manifest.get("test_battery_ids", [])],
    }
    if any(not ids for ids in split_ids.values()):
        raise ValueError(f"source split manifest has an empty split: {path}")
    step7.verify_no_split_overlap(split_ids)

    manifest_ids = set().union(*[set(ids) for ids in split_ids.values()])
    dataset_ids = set(records_by_id)
    missing_from_dataset = sorted(manifest_ids - dataset_ids)
    missing_from_manifest = sorted(dataset_ids - manifest_ids)
    if missing_from_dataset or missing_from_manifest:
        raise ValueError(
            f"dataset/split manifest battery IDs differ for seed {seed}; "
            f"missing_from_dataset={missing_from_dataset[:10]}, "
            f"missing_from_manifest={missing_from_manifest[:10]}"
        )
    return split_ids, manifest, path


def load_checkpoint_for_inference(
    checkpoint_path: Path,
    *,
    seed: int,
    horizon: int,
    model_name: str,
    cfg: dict[str, Any],
    device: str,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")

    if str(payload.get("model")) != model_name:
        raise ValueError(f"checkpoint model mismatch: {checkpoint_path}")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"checkpoint seed mismatch: {checkpoint_path}")
    if int(payload.get("horizon", -1)) != int(horizon):
        raise ValueError(f"checkpoint horizon mismatch: {checkpoint_path}")
    if "model_state_dict" not in payload:
        raise ValueError(f"checkpoint has no model_state_dict: {checkpoint_path}")
    if "normalization_mean" not in payload or "normalization_std" not in payload:
        raise ValueError(f"checkpoint has no saved train normalization statistics: {checkpoint_path}")

    checkpoint_cfg = dict(cfg)
    checkpoint_cfg.update(payload.get("runtime_config", {}))
    for key in ["lookback", "fixed_len"]:
        if int(checkpoint_cfg[key]) != int(cfg[key]):
            raise ValueError(f"checkpoint {key} mismatch: {checkpoint_path}")
    if checkpoint_cfg.get("sample_mode") != cfg["sample_mode"]:
        raise ValueError(f"checkpoint sample_mode mismatch: {checkpoint_path}")

    mean = np.asarray(payload["normalization_mean"], dtype=np.float32)
    std = np.asarray(payload["normalization_std"], dtype=np.float32)
    if mean.shape != std.shape or mean.size == 0 or not np.all(np.isfinite(mean)):
        raise ValueError(f"invalid checkpoint normalization arrays: {checkpoint_path}")
    if not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError(f"invalid checkpoint normalization std: {checkpoint_path}")

    model = make_model(model_name, checkpoint_cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload, checkpoint_cfg, mean, std


def parse_checkpoint_roots(values: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        values = [values]
    roots = [Path(value) for value in values]
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise FileNotFoundError("checkpoint root(s) do not exist: " + ", ".join(missing))
    return roots


def find_checkpoint_path(
    checkpoint_roots: Sequence[Path],
    *,
    seed: int,
    horizon: int,
    model_name: str,
) -> Path:
    candidates = [
        root / f"seed{seed}" / f"horizon{horizon}" / f"{model_name}.pt"
        for root in checkpoint_roots
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"no checkpoint found for seed={seed}, horizon={horizon}, model={model_name}; "
        f"attempted: {attempted}"
    )


def build_batch_reports(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required = {
        "dataset",
        "stage",
        "split",
        "seed",
        "batch_id",
        "horizon",
        "model",
        "battery_id",
        "cell_id",
        "actual_soh",
        "pred_soh",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"prediction table is missing batch-report columns: {missing}")

    batch_group_cols = ["dataset", "stage", "split", "seed", "batch_id", "horizon"]
    batch_rows: list[dict[str, Any]] = []
    for group_key, group in predictions.groupby(batch_group_cols, sort=True, dropna=False):
        key_values = dict(zip(batch_group_cols, group_key))
        persistence = group.loc[group["model"] == "persistence"]
        if persistence.empty:
            raise ValueError(f"missing persistence predictions for batch group {key_values}")
        persistence_mae = step7.compute_metrics(
            persistence["actual_soh"].to_numpy(), persistence["pred_soh"].to_numpy()
        )["MAE"]
        for model_name, model_group in group.groupby("model", sort=True):
            metrics = step7.metric_row_with_skill(
                model_group["actual_soh"].to_numpy(),
                model_group["pred_soh"].to_numpy(),
                persistence_mae,
            )
            batch_rows.append(
                {
                    **key_values,
                    "model": model_name,
                    "n_batteries": int(model_group["battery_id"].nunique()),
                    "n_windows": int(len(model_group)),
                    **metrics,
                }
            )
    by_seed = pd.DataFrame(batch_rows).sort_values(
        ["horizon", "batch_id", "seed", "MAE", "RMSE", "model"]
    )
    by_seed = step7.add_cpmlp_comparison_columns(
        by_seed,
        group_cols=batch_group_cols,
        mae_col="MAE",
        rmse_col="RMSE",
        mape_col="MAPE_percent",
    )

    summary_group_cols = ["dataset", "stage", "split", "batch_id", "horizon", "model"]
    by_batch_horizon = (
        by_seed.groupby(summary_group_cols, as_index=False)
        .agg(
            seeds_evaluated=("seed", "nunique"),
            n_batteries_mean=("n_batteries", "mean"),
            n_windows_mean=("n_windows", "mean"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAPE_percent_mean=("MAPE_percent", "mean"),
            MAPE_percent_std=("MAPE_percent", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            Skill_MAE_vs_persistence_mean=("Skill_MAE_vs_persistence", "mean"),
            Skill_MAE_vs_persistence_std=("Skill_MAE_vs_persistence", "std"),
            MAE_improvement_vs_cpmlp_mean=("MAE_improvement_vs_cpmlp", "mean"),
            MAE_improvement_percent_vs_cpmlp_mean=("MAE_improvement_percent_vs_cpmlp", "mean"),
            Skill_MAE_vs_cpmlp_mean=("Skill_MAE_vs_cpmlp", "mean"),
        )
        .sort_values(["horizon", "batch_id", "MAE_mean", "RMSE_mean", "model"])
    )

    cell_group_cols = batch_group_cols + ["battery_id", "cell_id"]
    cell_rows: list[dict[str, Any]] = []
    for group_key, group in predictions.groupby(cell_group_cols, sort=True, dropna=False):
        key_values = dict(zip(cell_group_cols, group_key))
        persistence = group.loc[group["model"] == "persistence"]
        if persistence.empty:
            raise ValueError(f"missing persistence predictions for cell group {key_values}")
        persistence_mae = step7.compute_metrics(
            persistence["actual_soh"].to_numpy(), persistence["pred_soh"].to_numpy()
        )["MAE"]
        for model_name, model_group in group.groupby("model", sort=True):
            metrics = step7.metric_row_with_skill(
                model_group["actual_soh"].to_numpy(),
                model_group["pred_soh"].to_numpy(),
                persistence_mae,
            )
            cell_rows.append(
                {
                    **key_values,
                    "model": model_name,
                    "n_windows": int(len(model_group)),
                    **metrics,
                }
            )
    by_cell = pd.DataFrame(cell_rows).sort_values(
        ["horizon", "batch_id", "seed", "battery_id", "MAE", "model"]
    )

    macro_group_cols = batch_group_cols + ["model"]
    macro_by_seed = (
        by_cell.groupby(macro_group_cols, as_index=False)
        .agg(
            n_batteries=("battery_id", "nunique"),
            n_windows=("n_windows", "sum"),
            macro_cell_MAE=("MAE", "mean"),
            macro_cell_RMSE=("RMSE", "mean"),
            macro_cell_MAPE_percent=("MAPE_percent", "mean"),
            macro_cell_R2=("R2", "mean"),
        )
        .sort_values(["horizon", "batch_id", "seed", "macro_cell_MAE", "model"])
    )
    macro_base_cols = batch_group_cols
    macro_persistence = macro_by_seed.loc[
        macro_by_seed["model"] == "persistence", macro_base_cols + ["macro_cell_MAE"]
    ].rename(columns={"macro_cell_MAE": "_persistence_macro_cell_MAE"})
    macro_by_seed = macro_by_seed.merge(macro_persistence, on=macro_base_cols, how="left")
    macro_by_seed["macro_cell_Skill_MAE_vs_persistence"] = np.where(
        macro_by_seed["_persistence_macro_cell_MAE"] > 0,
        1.0 - macro_by_seed["macro_cell_MAE"] / macro_by_seed["_persistence_macro_cell_MAE"],
        np.nan,
    )
    macro_by_seed = macro_by_seed.drop(columns=["_persistence_macro_cell_MAE"])
    macro_by_seed = step7.add_cpmlp_comparison_columns(
        macro_by_seed,
        group_cols=macro_base_cols,
        mae_col="macro_cell_MAE",
        rmse_col="macro_cell_RMSE",
        mape_col="macro_cell_MAPE_percent",
    )

    macro_summary_group_cols = ["dataset", "stage", "split", "batch_id", "horizon", "model"]
    macro_by_batch_horizon = (
        macro_by_seed.groupby(macro_summary_group_cols, as_index=False)
        .agg(
            seeds_evaluated=("seed", "nunique"),
            n_batteries_mean=("n_batteries", "mean"),
            n_windows_mean=("n_windows", "mean"),
            macro_cell_MAE_mean=("macro_cell_MAE", "mean"),
            macro_cell_MAE_std=("macro_cell_MAE", "std"),
            macro_cell_RMSE_mean=("macro_cell_RMSE", "mean"),
            macro_cell_RMSE_std=("macro_cell_RMSE", "std"),
            macro_cell_MAPE_percent_mean=("macro_cell_MAPE_percent", "mean"),
            macro_cell_MAPE_percent_std=("macro_cell_MAPE_percent", "std"),
            macro_cell_R2_mean=("macro_cell_R2", "mean"),
            macro_cell_R2_std=("macro_cell_R2", "std"),
            macro_cell_Skill_MAE_vs_persistence_mean=("macro_cell_Skill_MAE_vs_persistence", "mean"),
            macro_cell_Skill_MAE_vs_persistence_std=("macro_cell_Skill_MAE_vs_persistence", "std"),
            MAE_improvement_vs_cpmlp_mean=("MAE_improvement_vs_cpmlp", "mean"),
            MAE_improvement_percent_vs_cpmlp_mean=("MAE_improvement_percent_vs_cpmlp", "mean"),
            Skill_MAE_vs_cpmlp_mean=("Skill_MAE_vs_cpmlp", "mean"),
        )
        .sort_values(["horizon", "batch_id", "macro_cell_MAE_mean", "model"])
    )
    return {
        "test_results_by_seed_batch_model_horizon.csv": by_seed,
        "test_summary_by_batch_model_horizon.csv": by_batch_horizon,
        "test_metrics_by_cell.csv": by_cell,
        "test_results_macro_cell_by_seed_batch_model_horizon.csv": macro_by_seed,
        "test_summary_macro_cell_by_batch_model_horizon.csv": macro_by_batch_horizon,
    }


def write_inference_readme(output_dir: Path, cfg: dict[str, Any], args: argparse.Namespace) -> None:
    evaluated_models = unique_preserve_order(["persistence", *cfg["models"]])
    checkpoint_roots = [str(path) for path in parse_checkpoint_roots(args.checkpoint_root)]
    text = f"""# MATR Locked Test Inference-Only Batch Analysis

This folder reuses the previously trained global-batch checkpoints and the
original battery-level test split. No training, validation selection, Optuna
tuning, or batch-specific fine-tuning was performed in this run.

- Selected model: `{cfg["selected_model"]}`
- Models evaluated: {evaluated_models}
- Lookback cycles: {cfg["lookback"]}
- Horizons: {cfg["horizons"]}
- Seeds: {cfg["seeds"]}
- Dataset batch filter: none (all batches loaded)
- Checkpoint roots, searched in order: {checkpoint_roots}
- Source split manifest root: `{args.split_manifest_root}`
- Target scale: loaded from each checkpoint

Important files:

- `test_predictions.csv`: window-level predictions, including `batch_id`
- `validation_predictions.csv`: validation predictions when `--write-validation-predictions` is enabled
- `test_summary_by_model_horizon.csv`: global test metrics reproduced from checkpoints
- `test_results_by_seed_batch_model_horizon.csv`: window-weighted batch metrics per seed
- `test_summary_by_batch_model_horizon.csv`: seed-aggregated window-weighted batch metrics
- `test_metrics_by_cell.csv`: metrics for each held-out battery cell
- `test_summary_macro_cell_by_batch_model_horizon.csv`: seed-aggregated cell-macro batch metrics
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run_checkpoint_inference_only(
    args: argparse.Namespace,
    locked_config: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    if cfg["batches"]:
        raise ValueError(
            "--inference-only must use the original all-batch dataset; remove --batches "
            "and use --report-by-batch to stratify the existing test predictions"
        )
    if not args.checkpoint_root:
        raise ValueError("--checkpoint-root is required with --inference-only")
    if not args.split_manifest_root:
        raise ValueError("--split-manifest-root is required with --inference-only")

    checkpoint_roots = parse_checkpoint_roots(args.checkpoint_root)
    split_manifest_root = Path(args.split_manifest_root)
    if not split_manifest_root.is_dir():
        raise FileNotFoundError(f"split manifest root does not exist: {split_manifest_root}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_matr_files, excluded_files = step7.find_matr_files(args.data_root)
    records, dataset_manifest = step7.load_dataset_manifest(
        all_matr_files,
        excluded_files,
        data_root=Path(args.data_root),
        lookback=cfg["lookback"],
        horizons=cfg["horizons"],
        fixed_len=cfg["fixed_len"],
        sample_mode=cfg["sample_mode"],
        requested_batches=None,
        excluded_batch_files=None,
    )
    records_by_id = {record.battery_id: record for record in records}
    if len(dataset_manifest.get("used_batches", [])) < 2:
        raise ValueError("inference-only batch analysis expected the unfiltered multi-batch MATR dataset")
    learned_models = [model for model in cfg["models"] if model != "persistence"]
    evaluated_models = ["persistence", *learned_models]

    save_json(
        output_dir / "locked_test_config.json",
        {
            "execution_mode": "checkpoint_inference_only",
            "training_performed": False,
            "validation_selection_performed": False,
            "hyperparameter_tuning_performed": False,
            "batch_specific_fine_tuning_performed": False,
            "locked_config_path": str(args.config_path),
            "locked_config": locked_config,
            "runtime_config": cfg,
            "evaluated_models": evaluated_models,
            "device": device,
            "checkpoint_roots": [str(path) for path in checkpoint_roots],
            "split_manifest_root": str(split_manifest_root),
            "validation_predictions_written": bool(args.write_validation_predictions),
            "test_metrics_used_for_selection": False,
        },
    )
    save_json(output_dir / "dataset_manifest.json", dataset_manifest)

    all_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    validation_prediction_frames: list[pd.DataFrame] = []

    for seed in cfg["seeds"]:
        split_ids, split_manifest, source_manifest_path = load_source_split_manifest(
            split_manifest_root, seed, records_by_id, cfg
        )
        copied_manifest = dict(split_manifest)
        copied_manifest["source_manifest_path"] = str(source_manifest_path)
        copied_manifest["reused_without_resplitting"] = True
        save_json(output_dir / f"split_manifest_seed{seed}.json", copied_manifest)

        for horizon in cfg["horizons"]:
            validation_split = None
            if args.write_validation_predictions:
                validation_split = step7.build_horizon_split(
                    records_by_id,
                    split_ids["validation"],
                    "validation",
                    horizon,
                    cfg["lookback"],
                    cfg["sample_mode"],
                )
                step7.ensure_non_empty_split(validation_split, "validation", seed, horizon)
            test_split = step7.build_horizon_split(
                records_by_id,
                split_ids["test"],
                "test",
                horizon,
                cfg["lookback"],
                cfg["sample_mode"],
            )
            step7.ensure_non_empty_split(test_split, "test", seed, horizon)

            persistence_pred = persistence_predictions(test_split)
            persistence_delta = np.zeros_like(test_split.y_delta, dtype=np.float32)
            persistence_mae = step7.compute_metrics(test_split.y_soh_target, persistence_pred)["MAE"]
            all_rows.append(
                split_metrics_row(
                    split_name="test",
                    seed=seed,
                    horizon=horizon,
                    model="persistence",
                    y_true=test_split.y_soh_target,
                    y_pred=persistence_pred,
                    persistence_mae=persistence_mae,
                    best_epoch=0,
                    checkpoint_path="",
                )
            )
            prediction_frames.append(
                prediction_frame(
                    test_split,
                    seed=seed,
                    model="persistence",
                    pred_soh=persistence_pred,
                    pred_delta=persistence_delta,
                )
            )
            if validation_split is not None:
                validation_persistence_pred = persistence_predictions(validation_split)
                validation_prediction_frames.append(
                    prediction_frame(
                        validation_split,
                        seed=seed,
                        model="persistence",
                        pred_soh=validation_persistence_pred,
                        pred_delta=np.zeros_like(validation_split.y_delta, dtype=np.float32),
                    )
                )

            for model_name in learned_models:
                checkpoint_path = find_checkpoint_path(
                    checkpoint_roots,
                    seed=seed,
                    horizon=horizon,
                    model_name=model_name,
                )
                print(
                    f"[inference only] seed={seed} horizon={horizon} "
                    f"model={model_name} device={device}",
                    flush=True,
                )
                model, payload, checkpoint_cfg, mean, std = load_checkpoint_for_inference(
                    checkpoint_path,
                    seed=seed,
                    horizon=horizon,
                    model_name=model_name,
                    cfg=cfg,
                    device=device,
                )
                try:
                    X_test = step7.normalize(test_split.X, mean, std)
                except ValueError as exc:
                    raise ValueError(f"normalization shape mismatch for {checkpoint_path}") from exc
                target_scale = float(checkpoint_cfg["target_scale"])
                pred_soh, pred_delta = evaluate_model_on_split(
                    model,
                    test_split,
                    X_test,
                    target_scale,
                    cfg["batch_size"],
                    device,
                )
                best_epoch = int(payload.get("best_epoch", 0))
                all_rows.append(
                    split_metrics_row(
                        split_name="test",
                        seed=seed,
                        horizon=horizon,
                        model=model_name,
                        y_true=test_split.y_soh_target,
                        y_pred=pred_soh,
                        persistence_mae=persistence_mae,
                        best_epoch=best_epoch,
                        checkpoint_path=str(checkpoint_path),
                    )
                )
                prediction_frames.append(
                    prediction_frame(
                        test_split,
                        seed=seed,
                        model=model_name,
                        pred_soh=pred_soh,
                        pred_delta=pred_delta,
                    )
                )
                if validation_split is not None:
                    try:
                        X_validation = step7.normalize(validation_split.X, mean, std)
                    except ValueError as exc:
                        raise ValueError(
                            f"validation normalization shape mismatch for {checkpoint_path}"
                        ) from exc
                    validation_pred_soh, validation_pred_delta = evaluate_model_on_split(
                        model,
                        validation_split,
                        X_validation,
                        target_scale,
                        cfg["batch_size"],
                        device,
                    )
                    validation_prediction_frames.append(
                        prediction_frame(
                            validation_split,
                            seed=seed,
                            model=model_name,
                            pred_soh=validation_pred_soh,
                            pred_delta=validation_pred_delta,
                        )
                    )
                del model
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()

    raw = pd.DataFrame(all_rows)
    raw.to_csv(output_dir / "metrics_raw.csv", index=False)
    raw.to_csv(output_dir / "test_results_raw.csv", index=False)
    test_by_seed, test_by_horizon, test_overall = aggregate(all_rows, "test")
    test_by_seed.to_csv(output_dir / "test_results_by_seed.csv", index=False)
    test_by_horizon.to_csv(output_dir / "test_summary_by_model_horizon.csv", index=False)
    test_overall.to_csv(output_dir / "locked_test_summary.csv", index=False)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    if validation_prediction_frames:
        validation_predictions = pd.concat(validation_prediction_frames, ignore_index=True)
        validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    if args.report_by_batch:
        for filename, frame in build_batch_reports(predictions).items():
            frame.to_csv(output_dir / filename, index=False)

    selected_row = test_overall.loc[test_overall["model"] == cfg["selected_model"]]
    save_json(
        output_dir / "selected_model_test_summary.json",
        {
            "dataset": step7.DATASET,
            "stage": "locked_test_evaluation",
            "execution_mode": "checkpoint_inference_only",
            "training_performed": False,
            "test_metrics_used_for_selection": False,
            "selected_model": cfg["selected_model"],
            "locked_config_path": str(args.config_path),
            "runtime_config": cfg,
            "evaluated_models": evaluated_models,
            "test_summary": selected_row.iloc[0].to_dict() if not selected_row.empty else None,
        },
    )
    write_inference_readme(output_dir, cfg, args)
    print("\n=== Locked checkpoint inference-only test summary ===", flush=True)
    print(test_overall.to_string(index=False), flush=True)


def aggregate(rows: list[dict[str, Any]], split_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame([row for row in rows if row["split"] == split_name])
    if df.empty:
        return df, df, df
    by_seed = (
        df.groupby(["dataset", "stage", "split", "seed", "horizon", "model"], as_index=False)
        .agg(
            n_samples=("n_samples", "sum"),
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            MAPE_percent=("MAPE_percent", "mean"),
            R2=("R2", "mean"),
            Skill_MAE_vs_persistence=("Skill_MAE_vs_persistence", "mean"),
            best_epoch=("best_epoch", "first"),
            checkpoint_path=("checkpoint_path", "first"),
        )
        .sort_values(["horizon", "seed", "MAE", "RMSE", "model"])
    )
    by_seed = step7.add_cpmlp_comparison_columns(
        by_seed,
        group_cols=["dataset", "stage", "split", "seed", "horizon"],
        mae_col="MAE",
        rmse_col="RMSE",
        mape_col="MAPE_percent",
    )
    by_horizon = (
        by_seed.groupby(["dataset", "stage", "split", "horizon", "model"], as_index=False)
        .agg(
            seeds_evaluated=("seed", "nunique"),
            n_samples_mean=("n_samples", "mean"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAPE_percent_mean=("MAPE_percent", "mean"),
            MAPE_percent_std=("MAPE_percent", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            Skill_MAE_vs_persistence_mean=("Skill_MAE_vs_persistence", "mean"),
            MAE_improvement_vs_cpmlp_mean=("MAE_improvement_vs_cpmlp", "mean"),
            MAE_improvement_percent_vs_cpmlp_mean=("MAE_improvement_percent_vs_cpmlp", "mean"),
            Skill_MAE_vs_cpmlp_mean=("Skill_MAE_vs_cpmlp", "mean"),
        )
        .sort_values(["horizon", "MAE_mean", "RMSE_mean", "model"])
    )
    overall = (
        by_horizon.groupby(["dataset", "stage", "split", "model"], as_index=False)
        .agg(
            horizons_evaluated=("horizon", "nunique"),
            avg_MAE_mean=("MAE_mean", "mean"),
            avg_RMSE_mean=("RMSE_mean", "mean"),
            avg_MAPE_percent_mean=("MAPE_percent_mean", "mean"),
            std_MAE_mean=("MAE_mean", "std"),
            worst_MAE_mean=("MAE_mean", "max"),
            average_Skill_MAE_vs_persistence=("Skill_MAE_vs_persistence_mean", "mean"),
            average_MAE_improvement_vs_cpmlp=("MAE_improvement_vs_cpmlp_mean", "mean"),
            average_MAE_improvement_percent_vs_cpmlp=("MAE_improvement_percent_vs_cpmlp_mean", "mean"),
            average_Skill_MAE_vs_cpmlp=("Skill_MAE_vs_cpmlp_mean", "mean"),
        )
        .sort_values(["avg_MAE_mean", "avg_RMSE_mean", "avg_MAPE_percent_mean", "model"])
    )
    overall["std_MAE_mean"] = overall["std_MAE_mean"].fillna(0.0)
    return by_seed, by_horizon, overall


def write_readme(output_dir: Path, cfg: dict[str, Any]) -> None:
    text = f"""# MATR Locked Test Evaluation

This folder evaluates the validation-locked model configuration on the held-out
test split. Test metrics are reported only after model and hyperparameters were
locked.

- Selected model: `{cfg["selected_model"]}`
- Test metrics used for selection: false
- Lookback cycles: {cfg["lookback"]}
- Horizons: {cfg["horizons"]}
- Seeds: {cfg["seeds"]}
- Batches: {cfg["batches"] or "all"}
- Target scale: {cfg["target_scale"]}
- Features: {step7.FEATURES}

Important files:

- `test_results_raw.csv`: test metrics per model, seed, and horizon
- `test_summary_by_model_horizon.csv`: seed-aggregated test metrics per horizon
- `locked_test_summary.csv`: final cross-horizon test summary
- `selected_model_test_summary.json`: selected model test result summary
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    locked_config = load_locked_config(args.config_path)
    cfg = build_runtime_config(locked_config, args)
    if args.inference_only:
        run_checkpoint_inference_only(args, locked_config, cfg)
        return
    if args.split_manifest_root and cfg["batches"]:
        raise ValueError(
            "a source --split-manifest-root cannot be combined with --batches; "
            "the source manifests describe the original all-batch split"
        )
    if args.debug:
        cfg["epochs"] = min(cfg["epochs"], 3)
        cfg["patience"] = min(cfg["patience"], 2)
        cfg["fixed_len"] = min(cfg["fixed_len"], 40)
        cfg["batch_size"] = min(cfg["batch_size"], 16)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if not bool(locked_config.get("test_metrics_used", False)) is False:
        raise ValueError("locked config must not have used test metrics for selection")

    all_matr_files, excluded_files = step7.find_matr_files(args.data_root)
    matr_files, excluded_batch_files = step7.filter_matr_files_by_batches(all_matr_files, cfg["batches"])
    records, dataset_manifest = step7.load_dataset_manifest(
        matr_files,
        excluded_files,
        data_root=Path(args.data_root),
        lookback=cfg["lookback"],
        horizons=cfg["horizons"],
        fixed_len=cfg["fixed_len"],
        sample_mode=cfg["sample_mode"],
        requested_batches=cfg["batches"],
        excluded_batch_files=excluded_batch_files,
    )
    records_by_id = {record.battery_id: record for record in records}

    save_json(
        output_dir / "locked_test_config.json",
        {
            "locked_config_path": str(args.config_path),
            "locked_config": locked_config,
            "runtime_config": cfg,
            "device": device,
            "source_split_manifest_root": str(args.split_manifest_root) if args.split_manifest_root else None,
            "test_metrics_used_for_selection": False,
        },
    )
    save_json(output_dir / "dataset_manifest.json", dataset_manifest)

    all_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    split_manifests: list[dict[str, Any]] = []

    for seed in cfg["seeds"]:
        step7.pipe.set_seed(seed)
        if args.split_manifest_root:
            split_ids, source_manifest, source_manifest_path = load_source_split_manifest(
                Path(args.split_manifest_root), seed, records_by_id, cfg
            )
            split_manifest = dict(source_manifest)
            split_manifest["source_manifest_path"] = str(source_manifest_path)
            split_manifest["reused_without_resplitting"] = True
        else:
            split_ids = step7.split_battery_ids(records, seed=seed)
            split_manifest = step7.make_split_manifest(
                seed=seed,
                split_ids=split_ids,
                records_by_id=records_by_id,
                horizons=cfg["horizons"],
                lookback=cfg["lookback"],
                sample_mode=cfg["sample_mode"],
            )
        save_json(output_dir / f"split_manifest_seed{seed}.json", split_manifest)
        split_manifests.append(split_manifest)

        for horizon in cfg["horizons"]:
            train_split = step7.build_horizon_split(
                records_by_id, split_ids["train"], "train", horizon, cfg["lookback"], cfg["sample_mode"]
            )
            val_split = step7.build_horizon_split(
                records_by_id, split_ids["validation"], "validation", horizon, cfg["lookback"], cfg["sample_mode"]
            )
            test_split = step7.build_horizon_split(
                records_by_id, split_ids["test"], "test", horizon, cfg["lookback"], cfg["sample_mode"]
            )
            step7.ensure_non_empty_split(train_split, "train", seed, horizon)
            step7.ensure_non_empty_split(val_split, "validation", seed, horizon)
            step7.ensure_non_empty_split(test_split, "test", seed, horizon)

            mean, std = step7.fit_train_normalizer(train_split.X)
            X_train = step7.normalize(train_split.X, mean, std)
            X_val = step7.normalize(val_split.X, mean, std)
            X_test = step7.normalize(test_split.X, mean, std)

            val_persistence_pred = persistence_predictions(val_split)
            test_persistence_pred = persistence_predictions(test_split)
            val_persistence_mae = step7.compute_metrics(val_split.y_soh_target, val_persistence_pred)["MAE"]
            test_persistence_mae = step7.compute_metrics(test_split.y_soh_target, test_persistence_pred)["MAE"]

            for split_name, split, pred, persistence_mae in [
                ("validation", val_split, val_persistence_pred, val_persistence_mae),
                ("test", test_split, test_persistence_pred, test_persistence_mae),
            ]:
                all_rows.append(
                    split_metrics_row(
                        split_name=split_name,
                        seed=seed,
                        horizon=horizon,
                        model="persistence",
                        y_true=split.y_soh_target,
                        y_pred=pred,
                        persistence_mae=persistence_mae,
                        best_epoch=0,
                        checkpoint_path="",
                    )
                )

            for model_name in cfg["models"]:
                if model_name == "persistence":
                    continue
                step7.pipe.set_seed(seed)
                model = make_model(model_name, cfg)
                train_loader = scaled_delta_loader(train_split, X_train, cfg["target_scale"], cfg["batch_size"], shuffle=True)
                val_loader = scaled_delta_loader(val_split, X_val, cfg["target_scale"], cfg["batch_size"], shuffle=False)
                print(f"[train locked] seed={seed} horizon={horizon} model={model_name} device={device}", flush=True)
                model, history, best_epoch, best_val_mae = step7.train_delta_model(
                    model,
                    train_loader,
                    val_loader,
                    val_target_soh=val_split.y_soh_target,
                    val_current_soh=val_split.current_soh,
                    target_scale=cfg["target_scale"],
                    epochs=cfg["epochs"],
                    lr=cfg["lr"],
                    weight_decay=cfg["weight_decay"],
                    patience=cfg["patience"],
                    min_delta=0.0,
                    clip_grad_norm=cfg["clip_grad_norm"],
                    device=device,
                )
                model_dir = checkpoint_root / f"seed{seed}" / f"horizon{horizon}"
                model_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = model_dir / f"{model_name}.pt"
                history_path = model_dir / f"{model_name}_history.csv"
                history.to_csv(history_path, index=False)
                torch.save(
                    {
                        "stage": "locked_test_evaluation",
                        "test_metrics_used_for_selection": False,
                        "model": model_name,
                        "seed": seed,
                        "horizon": horizon,
                        "locked_config": locked_config,
                        "runtime_config": cfg,
                        "best_epoch": int(best_epoch),
                        "best_validation_mae_reconstructed_soh": float(best_val_mae),
                        "normalization_mean": mean.astype(np.float32),
                        "normalization_std": std.astype(np.float32),
                        "model_state_dict": model.state_dict(),
                    },
                    checkpoint_path,
                )

                for split_name, split, X_norm, persistence_mae in [
                    ("validation", val_split, X_val, val_persistence_mae),
                    ("test", test_split, X_test, test_persistence_mae),
                ]:
                    pred_soh, pred_delta = evaluate_model_on_split(
                        model,
                        split,
                        X_norm,
                        cfg["target_scale"],
                        cfg["batch_size"],
                        device,
                    )
                    all_rows.append(
                        split_metrics_row(
                            split_name=split_name,
                            seed=seed,
                            horizon=horizon,
                            model=model_name,
                            y_true=split.y_soh_target,
                            y_pred=pred_soh,
                            persistence_mae=persistence_mae,
                            best_epoch=int(best_epoch),
                            checkpoint_path=str(checkpoint_path),
                        )
                    )
                    if split_name == "test":
                        for meta, true_soh, current_soh, delta_true, soh_pred, delta_pred in zip(
                            split.meta,
                            split.y_soh_target,
                            split.current_soh,
                            split.y_delta,
                            pred_soh,
                            pred_delta,
                        ):
                            prediction_rows.append(
                                {
                                    **meta,
                                    "seed": seed,
                                    "model": model_name,
                                    "actual_soh": float(true_soh),
                                    "current_soh": float(current_soh),
                                    "actual_delta_soh": float(delta_true),
                                    "pred_delta_soh": float(delta_pred),
                                    "pred_soh": float(soh_pred),
                                    "abs_error": float(abs(true_soh - soh_pred)),
                                }
                            )

            pd.DataFrame(all_rows).to_csv(output_dir / "metrics_raw.csv", index=False)

    raw = pd.DataFrame(all_rows)
    raw.to_csv(output_dir / "metrics_raw.csv", index=False)
    raw.loc[raw["split"] == "validation"].to_csv(output_dir / "validation_results_raw.csv", index=False)
    raw.loc[raw["split"] == "test"].to_csv(output_dir / "test_results_raw.csv", index=False)
    if prediction_rows:
        pd.DataFrame(prediction_rows).to_csv(output_dir / "test_predictions.csv", index=False)

    val_by_seed, val_by_horizon, val_overall = aggregate(all_rows, "validation")
    test_by_seed, test_by_horizon, test_overall = aggregate(all_rows, "test")
    val_by_seed.to_csv(output_dir / "validation_results_by_seed.csv", index=False)
    val_by_horizon.to_csv(output_dir / "validation_summary_by_model_horizon.csv", index=False)
    val_overall.to_csv(output_dir / "locked_validation_summary.csv", index=False)
    test_by_seed.to_csv(output_dir / "test_results_by_seed.csv", index=False)
    test_by_horizon.to_csv(output_dir / "test_summary_by_model_horizon.csv", index=False)
    test_overall.to_csv(output_dir / "locked_test_summary.csv", index=False)

    selected_row = test_overall.loc[test_overall["model"] == cfg["selected_model"]]
    selected_payload = {
        "dataset": step7.DATASET,
        "stage": "locked_test_evaluation",
        "test_metrics_used_for_selection": False,
        "selected_model": cfg["selected_model"],
        "locked_config_path": str(args.config_path),
        "runtime_config": cfg,
        "test_summary": selected_row.iloc[0].to_dict() if not selected_row.empty else None,
    }
    save_json(output_dir / "selected_model_test_summary.json", selected_payload)
    write_readme(output_dir, cfg)
    print("\n=== Locked test summary ===", flush=True)
    print(test_overall.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the locked MATR Step 7 model on held-out test splits.")
    parser.add_argument("--data-root", default="MATR")
    parser.add_argument("--config-path", default="outputs/matr_step7_locked_validation_config/final_validation_config.json")
    parser.add_argument("--output-dir", default="outputs/matr_locked_test_evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Reuse saved global checkpoints and original split manifests; perform checkpoint "
            "inference only, with no training or validation-based selection."
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        nargs="+",
        default=None,
        help=(
            "One or more checkpoint directories containing "
            "seed<seed>/horizon<horizon>/<model>.pt; searched in the given order."
        ),
    )
    parser.add_argument(
        "--split-manifest-root",
        default=None,
        help="Directory containing the original split_manifest_seed<seed>.json files.",
    )
    parser.add_argument(
        "--report-by-batch",
        action="store_true",
        help="Write window-weighted and cell-macro test summaries stratified by MATR batch.",
    )
    parser.add_argument(
        "--write-validation-predictions",
        action="store_true",
        help=(
            "In inference-only mode, also write validation_predictions.csv using the saved "
            "train normalization and original validation battery IDs."
        ),
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--include-references", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--sample-mode", choices=step7.SAMPLE_MODES, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument(
        "--batches",
        nargs="+",
        default=None,
        help="Filter MATR files to protocol batch ids before battery-level split, e.g. --batches b1 or --batches b1 b2.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--fixed-len", type=int, default=None)
    parser.add_argument("--target-scale", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--mlp-embed-dim", type=int, default=None)
    parser.add_argument("--gru-embed-dim", type=int, default=None)
    parser.add_argument("--model-hidden", type=int, default=None)
    parser.add_argument("--gru-hidden", type=int, default=None)
    parser.add_argument("--dsconv-channels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--clip-grad-norm", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
