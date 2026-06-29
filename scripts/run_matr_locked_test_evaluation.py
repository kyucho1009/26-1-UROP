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
    return default


def build_runtime_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    hyper = config.get("hyperparameters", {}) or config.get("selected_hyperparameters", {})
    selected_model = str(config.get("selected_model"))
    if not selected_model or selected_model == "None":
        raise ValueError("locked config must include selected_model")
    if bool(config.get("test_metrics_used", False)):
        raise ValueError("locked config unexpectedly says test metrics were used for selection")

    models = [selected_model]
    if args.include_references:
        models = ["persistence", "cpmlp", selected_model]
    if args.models:
        models = args.models
    models = unique_preserve_order(models)
    unknown = sorted(set(models) - set(step7.DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"unsupported model(s): {unknown}; allowed={step7.DEFAULT_MODELS}")

    return {
        "selected_model": selected_model,
        "models": models,
        "lookback": int(args.lookback or config.get("lookback_cycles", 20)),
        "horizons": [int(item) for item in (args.horizons or config.get("horizons", [10, 50, 100]))],
        "seeds": [int(item) for item in (args.seeds or config.get("seeds", [42, 43, 44]))],
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
    locked_config = load_json(args.config_path)
    cfg = build_runtime_config(locked_config, args)
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

    matr_files, excluded_files = step7.find_matr_files(args.data_root)
    records, dataset_manifest = step7.load_dataset_manifest(
        matr_files,
        excluded_files,
        data_root=Path(args.data_root),
        lookback=cfg["lookback"],
        horizons=cfg["horizons"],
        fixed_len=cfg["fixed_len"],
    )
    records_by_id = {record.battery_id: record for record in records}

    save_json(
        output_dir / "locked_test_config.json",
        {
            "locked_config_path": str(args.config_path),
            "locked_config": locked_config,
            "runtime_config": cfg,
            "device": device,
            "test_metrics_used_for_selection": False,
        },
    )
    save_json(output_dir / "dataset_manifest.json", dataset_manifest)

    all_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    split_manifests: list[dict[str, Any]] = []

    for seed in cfg["seeds"]:
        step7.pipe.set_seed(seed)
        split_ids = step7.split_battery_ids(records, seed=seed)
        split_manifest = step7.make_split_manifest(
            seed=seed,
            split_ids=split_ids,
            records_by_id=records_by_id,
            horizons=cfg["horizons"],
            lookback=cfg["lookback"],
        )
        save_json(output_dir / f"split_manifest_seed{seed}.json", split_manifest)
        split_manifests.append(split_manifest)

        for horizon in cfg["horizons"]:
            train_split = step7.build_horizon_split(records_by_id, split_ids["train"], "train", horizon, cfg["lookback"])
            val_split = step7.build_horizon_split(records_by_id, split_ids["validation"], "validation", horizon, cfg["lookback"])
            test_split = step7.build_horizon_split(records_by_id, split_ids["test"], "test", horizon, cfg["lookback"])
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
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--include-references", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
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
