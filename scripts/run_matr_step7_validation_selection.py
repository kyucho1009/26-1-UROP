from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import compare_soh_models as model_lib
import soh_gru_dsconv_pipeline as pipe

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyTorch is required for MATR Step 7 validation selection."
    ) from exc


DATASET = "MATR"
SELECTION_STAGE = "step7_validation_only"
FEATURES = ["current", "voltage", "dV"]
TARGET = "delta_soh"
DEFAULT_MODELS = [
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
    "cpmlp_cpgru_fusion",
    "cpmlp_dsconv_fusion",
    "cpmlp_cpdsconv_fusion",
]
METRIC_COLUMNS = ["MAE", "RMSE", "MAPE_percent", "R2", "Skill_MAE_vs_persistence"]


@dataclass(frozen=True)
class BatteryRecord:
    battery_id: str
    cell_id: str
    file_path: Path
    X_early: np.ndarray
    soh_current: float
    target_soh_by_horizon: dict[int, float]
    available_cycles: list[int]
    reference_capacity: float


@dataclass(frozen=True)
class HorizonSplit:
    X: np.ndarray
    y_soh_target: np.ndarray
    y_delta: np.ndarray
    current_soh: np.ndarray
    meta: list[dict[str, Any]]


def finite_1d(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return arr[np.isfinite(arr)]


def extract_practical_features(cycle: dict[str, Any], fixed_len: int) -> np.ndarray:
    voltage = finite_1d(cycle.get("voltage_in_V", []))
    current = finite_1d(cycle.get("current_in_A", []))
    min_len = min(len(voltage), len(current))
    if min_len < 5:
        raise ValueError("not enough voltage/current feature points")
    voltage = voltage[:min_len]
    current = current[:min_len]
    d_voltage = np.gradient(voltage).astype(np.float32)
    features = np.stack([current, voltage, d_voltage], axis=1)
    return pipe.resample_cycle(features, fixed_len=fixed_len)


def is_matr_file(path: str | Path) -> bool:
    return Path(path).name.startswith("MATR")


def find_all_pkl_files(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    if root.is_file():
        if root.suffix.lower() != ".pkl":
            raise ValueError(f"--data-root file is not a pkl: {root}")
        return [root]
    return sorted(root.rglob("*.pkl"))


def find_matr_files(data_root: str | Path) -> tuple[list[Path], list[Path]]:
    all_files = find_all_pkl_files(data_root)
    matr_files = [path for path in all_files if is_matr_file(path)]
    excluded = [path for path in all_files if not is_matr_file(path)]
    if not matr_files:
        raise ValueError(f"no MATR pkl files found under {data_root}")
    assert_matr_only(matr_files)
    return sorted(matr_files), sorted(excluded)


def assert_matr_only(files: Iterable[Path]) -> None:
    non_matr = [str(path) for path in files if not is_matr_file(path)]
    if non_matr:
        raise ValueError("non-MATR dataset files would be used: " + ", ".join(non_matr[:10]))


def load_matr_battery(
    path: Path,
    lookback: int,
    horizons: Sequence[int],
    fixed_len: int,
) -> tuple[BatteryRecord | None, dict[str, Any]]:
    if not is_matr_file(path):
        raise ValueError(f"refusing to load non-MATR file: {path}")

    with path.open("rb") as handle:
        cell = pickle.load(handle)

    cycles = list(cell.get("cycle_data", []))
    cycles = sorted(cycles, key=lambda item: int(item.get("cycle_number", len(cycles))))
    cell_id = str(cell.get("cell_id", path.stem))
    battery_id = path.stem
    reference_capacity = pipe.infer_reference_capacity(cell, cycles)

    features_by_cycle: dict[int, np.ndarray] = {}
    soh_by_cycle: dict[int, float] = {}
    feature_errors: list[dict[str, Any]] = []
    soh_errors: list[dict[str, Any]] = []

    for idx, cycle in enumerate(cycles):
        cycle_number = int(cycle.get("cycle_number", idx + 1))
        try:
            soh_by_cycle[cycle_number] = float(pipe.extract_soh_label(cycle, reference_capacity))
        except Exception as exc:
            soh_errors.append({"cycle_number": cycle_number, "reason": str(exc)})

        if 1 <= cycle_number <= lookback:
            try:
                features_by_cycle[cycle_number] = extract_practical_features(cycle, fixed_len=fixed_len)
            except Exception as exc:
                feature_errors.append({"cycle_number": cycle_number, "reason": str(exc)})

    missing_input_cycles = [
        cycle_number
        for cycle_number in range(1, lookback + 1)
        if cycle_number not in features_by_cycle
    ]
    missing_current_soh = lookback not in soh_by_cycle
    target_soh_by_horizon = {
        int(horizon): float(soh_by_cycle[lookback + int(horizon)])
        for horizon in horizons
        if lookback + int(horizon) in soh_by_cycle
    }

    manifest_item: dict[str, Any] = {
        "battery_id": battery_id,
        "cell_id": cell_id,
        "file": path.name,
        "dataset": DATASET,
        "n_raw_cycles": len(cycles),
        "n_soh_cycles": len(soh_by_cycle),
        "available_cycle_min": min(soh_by_cycle) if soh_by_cycle else None,
        "available_cycle_max": max(soh_by_cycle) if soh_by_cycle else None,
        "lookback_cycles": lookback,
        "target_cycles_requested": [lookback + int(horizon) for horizon in horizons],
        "available_horizons": sorted(target_soh_by_horizon),
        "missing_horizons": [
            int(horizon)
            for horizon in horizons
            if int(horizon) not in target_soh_by_horizon
        ],
        "reference_capacity": float(reference_capacity),
        "capacity_used_as_input": False,
        "feature_order": FEATURES,
    }

    if missing_input_cycles or missing_current_soh or not target_soh_by_horizon:
        reasons = []
        if missing_input_cycles:
            reasons.append(f"missing input cycles {missing_input_cycles}")
        if missing_current_soh:
            reasons.append(f"missing SOH at cycle {lookback}")
        if not target_soh_by_horizon:
            reasons.append("no requested target horizon available")
        manifest_item.update(
            {
                "status": "skipped",
                "reason": "; ".join(reasons),
                "feature_errors": feature_errors[:5],
                "soh_errors": soh_errors[:5],
            }
        )
        return None, manifest_item

    X_early = np.stack([features_by_cycle[i] for i in range(1, lookback + 1)]).astype(np.float32)
    record = BatteryRecord(
        battery_id=battery_id,
        cell_id=cell_id,
        file_path=path,
        X_early=X_early,
        soh_current=float(soh_by_cycle[lookback]),
        target_soh_by_horizon=target_soh_by_horizon,
        available_cycles=sorted(soh_by_cycle),
        reference_capacity=float(reference_capacity),
    )
    manifest_item.update(
        {
            "status": "used",
            "soh_current_cycle_20": float(record.soh_current),
            "input_shape": list(X_early.shape),
            "feature_errors": feature_errors[:5],
            "soh_errors": soh_errors[:5],
        }
    )
    return record, manifest_item


def load_dataset_manifest(
    matr_files: Sequence[Path],
    excluded_files: Sequence[Path],
    data_root: Path,
    lookback: int,
    horizons: Sequence[int],
    fixed_len: int,
) -> tuple[list[BatteryRecord], dict[str, Any]]:
    records: list[BatteryRecord] = []
    battery_items = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []

    for path in matr_files:
        record, item = load_matr_battery(path, lookback=lookback, horizons=horizons, fixed_len=fixed_len)
        battery_items.append(item)
        if record is None:
            continue
        if record.battery_id in seen_ids:
            duplicate_ids.append(record.battery_id)
        seen_ids.add(record.battery_id)
        records.append(record)

    if duplicate_ids:
        raise ValueError("duplicate battery IDs: " + ", ".join(sorted(set(duplicate_ids))))
    if len(records) < 3:
        raise ValueError(
            f"need at least 3 usable MATR batteries for 6:2:2 split; got {len(records)}"
        )

    horizon_counts = {
        str(horizon): int(sum(int(horizon) in record.target_soh_by_horizon for record in records))
        for horizon in horizons
    }
    manifest = {
        "dataset": DATASET,
        "data_root": str(data_root),
        "total_pkl_files_found": len(matr_files) + len(excluded_files),
        "used_matr_pkl_files": [path.name for path in matr_files],
        "excluded_non_matr_pkl_count": len(excluded_files),
        "excluded_non_matr_pkl_examples": [path.name for path in excluded_files[:20]],
        "capacity_used_as_input": False,
        "features": FEATURES,
        "target": TARGET,
        "lookback_cycles": lookback,
        "horizons": [int(horizon) for horizon in horizons],
        "fixed_len": fixed_len,
        "n_usable_batteries": len(records),
        "sample_counts_by_horizon": horizon_counts,
        "batteries": battery_items,
    }
    return records, manifest


def split_battery_ids(
    records: Sequence[BatteryRecord],
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> dict[str, list[str]]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("split ratios must sum to 1.0")
    ids = sorted(record.battery_id for record in records)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    if n < 3:
        raise ValueError("need at least 3 batteries for train/validation/test split")
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = 1, 1, n - 2

    return {
        "train": ids[:n_train],
        "validation": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def verify_no_split_overlap(split_ids: dict[str, list[str]]) -> None:
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for left, right in pairs:
        overlap = sorted(set(split_ids[left]) & set(split_ids[right]))
        if overlap:
            raise ValueError(f"battery ID overlap between {left} and {right}: {overlap}")


def build_horizon_split(
    records_by_id: dict[str, BatteryRecord],
    battery_ids: Sequence[str],
    split_name: str,
    horizon: int,
    lookback: int,
) -> HorizonSplit:
    X_list: list[np.ndarray] = []
    y_soh: list[float] = []
    y_delta: list[float] = []
    current_soh: list[float] = []
    meta: list[dict[str, Any]] = []

    for battery_id in battery_ids:
        record = records_by_id[battery_id]
        if horizon not in record.target_soh_by_horizon:
            continue
        target_cycle = lookback + horizon
        target_soh = float(record.target_soh_by_horizon[horizon])
        delta = float(record.soh_current - target_soh)
        X_list.append(record.X_early)
        y_soh.append(target_soh)
        y_delta.append(delta)
        current_soh.append(float(record.soh_current))
        meta.append(
            {
                "dataset": DATASET,
                "split": split_name,
                "battery_id": record.battery_id,
                "cell_id": record.cell_id,
                "file": record.file_path.name,
                "input_start_cycle": 1,
                "input_end_cycle": lookback,
                "target_cycle": target_cycle,
                "horizon": horizon,
                "soh_current_cycle_20": float(record.soh_current),
                "soh_target": target_soh,
                "delta_soh_true": delta,
            }
        )

    if not X_list:
        return HorizonSplit(
            X=np.empty((0, lookback, 0, len(FEATURES)), dtype=np.float32),
            y_soh_target=np.empty((0,), dtype=np.float32),
            y_delta=np.empty((0,), dtype=np.float32),
            current_soh=np.empty((0,), dtype=np.float32),
            meta=[],
        )
    return HorizonSplit(
        X=np.stack(X_list).astype(np.float32),
        y_soh_target=np.asarray(y_soh, dtype=np.float32),
        y_delta=np.asarray(y_delta, dtype=np.float32),
        current_soh=np.asarray(current_soh, dtype=np.float32),
        meta=meta,
    )


def make_split_manifest(
    seed: int,
    split_ids: dict[str, list[str]],
    records_by_id: dict[str, BatteryRecord],
    horizons: Sequence[int],
    lookback: int,
) -> dict[str, Any]:
    verify_no_split_overlap(split_ids)
    sample_counts: dict[str, dict[str, int]] = {}
    unavailable: dict[str, dict[str, list[str]]] = {}
    for split_name, ids in split_ids.items():
        sample_counts[split_name] = {}
        unavailable[split_name] = {}
        for horizon in horizons:
            available_ids = [
                battery_id
                for battery_id in ids
                if int(horizon) in records_by_id[battery_id].target_soh_by_horizon
            ]
            sample_counts[split_name][str(horizon)] = len(available_ids)
            unavailable[split_name][str(horizon)] = sorted(set(ids) - set(available_ids))

    return {
        "dataset": DATASET,
        "seed": seed,
        "split_level": "battery",
        "ratio": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "lookback_cycles": lookback,
        "horizons": [int(horizon) for horizon in horizons],
        "train_battery_ids": split_ids["train"],
        "validation_battery_ids": split_ids["validation"],
        "test_battery_ids": split_ids["test"],
        "counts": {split: len(ids) for split, ids in split_ids.items()},
        "sample_counts_by_split_horizon": sample_counts,
        "battery_ids_without_target_by_split_horizon": unavailable,
        "no_overlap_verified": True,
    }


def fit_train_normalizer(train_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train_X.size == 0:
        raise ValueError("cannot fit normalization statistics with empty train data")
    mean = train_X.mean(axis=(0, 1, 2), keepdims=True)
    std = train_X.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.any():
        raise ValueError("no finite predictions for metric computation")
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-8)) * 100.0)
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if denom <= 0 else float(1.0 - np.sum(err**2) / denom)
    return {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape, "R2": r2}


def metric_row_with_skill(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence_mae: float,
) -> dict[str, float]:
    row = compute_metrics(y_true, y_pred)
    if not np.isfinite(persistence_mae) or persistence_mae <= 0:
        row["Skill_MAE_vs_persistence"] = float("nan")
    else:
        row["Skill_MAE_vs_persistence"] = float(1.0 - row["MAE"] / persistence_mae)
    return row


def predict_delta(model: nn.Module, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for X_b, y_b in loader:
            output = model(X_b.to(device)).squeeze(-1).detach().cpu().numpy()
            preds.append(output)
            targets.append(y_b.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def train_delta_model(
    model: nn.Module,
    train_loader,
    val_loader,
    val_target_soh: np.ndarray,
    val_current_soh: np.ndarray,
    target_scale: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    clip_grad_norm: float,
    device: str,
) -> tuple[nn.Module, pd.DataFrame, int, float]:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_mae = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for X_b, y_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()
            pred_delta = model(X_b).squeeze(-1)
            loss = criterion(pred_delta, y_b)
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_pred_delta_scaled, val_true_delta_scaled = predict_delta(model, val_loader, device=device)
        val_pred_delta = val_pred_delta_scaled / target_scale
        val_pred_soh = val_current_soh - val_pred_delta
        val_err = val_target_soh - val_pred_soh
        val_mae = float(np.mean(np.abs(val_err)))
        val_rmse = float(np.sqrt(np.mean(val_err**2)))
        val_delta_mse = float(np.mean((val_true_delta_scaled - val_pred_delta_scaled) ** 2))
        row = {
            "epoch": epoch,
            "target_scale": float(target_scale),
            "train_scaled_delta_mse": float(np.mean(train_losses)),
            "val_scaled_delta_mse": val_delta_mse,
            "val_MAE_reconstructed_soh": val_mae,
            "val_RMSE_reconstructed_soh": val_rmse,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_scaled_delta_mse={row['train_scaled_delta_mse']:.6g} "
            f"val_MAE_soh={val_mae:.6g}"
        )

        improved = val_mae < (best_val_mae - min_delta)
        if improved:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            print(
                f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                f"best_val_MAE_soh={best_val_mae:.6g}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch, best_val_mae


def ensure_non_empty_split(split: HorizonSplit, split_name: str, seed: int, horizon: int) -> None:
    if len(split.y_soh_target) == 0:
        raise ValueError(
            f"seed={seed} horizon={horizon} has no {split_name} samples after target-cycle filtering"
        )


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [json_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return json_sanitize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text or any(ch in text for ch in ":#[]{}&,*\n"):
        return json.dumps(text)
    return text


def write_simple_yaml(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []

    def emit(key: str, value: Any, indent: int = 0) -> None:
        prefix = " " * indent
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            for child_key, child_value in value.items():
                emit(str(child_key), child_value, indent + 2)
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}  - {json.dumps(item)}")
                else:
                    lines.append(f"{prefix}  - {yaml_scalar(item)}")
        else:
            lines.append(f"{prefix}{key}: {yaml_scalar(value)}")

    for key, value in payload.items():
        emit(key, value)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_readme(path: Path, config: dict[str, Any], selected_model: str | None) -> None:
    text = f"""# MATR Step 7 Validation Selection

This directory was produced by `scripts/run_matr_step7_validation_selection.py`.

- Dataset used: MATR only
- Selection stage: {SELECTION_STAGE}
- Test metrics used for selection: false
- Lookback cycles: {config["lookback_cycles"]}
- Horizons: {config["horizons"]}
- Features: {FEATURES}
- Target: delta_soh = SOH_20 - SOH_(20+h)
- Model selection: validation-only metric values; lowest average MAE across horizons,
  then lower average RMSE, lower average MAPE, lower MAE standard deviation, and
  higher average skill versus persistence
- Selected model: {selected_model or "not selected"}

The test split is saved only in split manifests. No test predictions or test
metrics are computed in this Step 7 pipeline.
"""
    path.write_text(text, encoding="utf-8")


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    return pipe.make_loader(X, y, batch_size=batch_size, shuffle=shuffle)


def evaluate_persistence(
    seed: int,
    horizon: int,
    val_split: HorizonSplit,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], float]:
    pred_delta = np.zeros_like(val_split.y_delta, dtype=np.float32)
    pred_soh = val_split.current_soh - pred_delta
    metrics = compute_metrics(val_split.y_soh_target, pred_soh)
    persistence_mae = metrics["MAE"]
    row = {
        "dataset": DATASET,
        "selection_stage": SELECTION_STAGE,
        "seed": seed,
        "horizon": horizon,
        "model": "persistence",
        "n_validation_samples": int(len(val_split.y_soh_target)),
        **metrics,
        "Skill_MAE_vs_persistence": 0.0,
        "best_epoch": 0,
        "checkpoint_path": str(checkpoint_dir / "persistence.json"),
    }
    save_json(
        checkpoint_dir / "persistence.json",
        {
            "model": "persistence",
            "dataset": DATASET,
            "selection_stage": SELECTION_STAGE,
            "seed": seed,
            "horizon": horizon,
            "rule": "delta_soh_pred=0; soh_pred=soh_current_at_cycle_20",
            "metrics": {key: row[key] for key in METRIC_COLUMNS},
        },
    )
    return row, persistence_mae


def evaluate_neural_model(
    model_name: str,
    seed: int,
    horizon: int,
    lookback: int,
    fixed_len: int,
    train_split: HorizonSplit,
    val_split: HorizonSplit,
    mean: np.ndarray,
    std: np.ndarray,
    persistence_mae: float,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    pipe.set_seed(seed)
    model = model_lib.make_model(
        model_name,
        early_cycle=lookback,
        fixed_len=fixed_len,
        mlp_embed_dim=args.mlp_embed_dim,
        gru_embed_dim=args.gru_embed_dim,
        model_hidden=args.model_hidden,
        gru_hidden=args.gru_hidden,
        dsconv_channels=args.dsconv_channels,
        dropout=args.dropout,
    )
    if args.zero_output_init:
        model_lib.zero_last_linear(model)

    X_train = normalize(train_split.X, mean, std)
    X_val = normalize(val_split.X, mean, std)
    y_train_delta_scaled = (train_split.y_delta * args.target_scale).astype(np.float32)
    y_val_delta_scaled = (val_split.y_delta * args.target_scale).astype(np.float32)
    train_loader = make_loader(X_train, y_train_delta_scaled, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val_delta_scaled, batch_size=args.batch_size, shuffle=False)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[train] seed={seed} horizon={horizon} model={model_name} device={device}")
    model, history, best_epoch, best_val_mae = train_delta_model(
        model,
        train_loader,
        val_loader,
        val_target_soh=val_split.y_soh_target,
        val_current_soh=val_split.current_soh,
        target_scale=args.target_scale,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        clip_grad_norm=args.clip_grad_norm,
        device=device,
    )

    pred_delta_scaled, true_delta_scaled = predict_delta(model, val_loader, device=device)
    pred_delta = pred_delta_scaled / args.target_scale
    pred_soh = val_split.current_soh - pred_delta
    metrics = metric_row_with_skill(val_split.y_soh_target, pred_soh, persistence_mae)
    checkpoint_path = checkpoint_dir / f"{model_name}.pt"
    history_path = checkpoint_dir / f"{model_name}_history.csv"
    history.to_csv(history_path, index=False)

    torch.save(
        {
            "dataset": DATASET,
            "selection_stage": SELECTION_STAGE,
            "test_metrics_used": False,
            "model": model_name,
            "seed": seed,
            "horizon": horizon,
            "lookback_cycles": lookback,
            "fixed_len": fixed_len,
            "features": FEATURES,
            "target": TARGET,
            "target_scale": float(args.target_scale),
            "model_state_dict": model.state_dict(),
            "normalization_mean": mean.astype(np.float32),
            "normalization_std": std.astype(np.float32),
            "best_epoch": best_epoch,
            "best_validation_mae_reconstructed_soh": best_val_mae,
            "metrics": metrics,
        },
        checkpoint_path,
    )

    return {
        "dataset": DATASET,
        "selection_stage": SELECTION_STAGE,
        "seed": seed,
        "horizon": horizon,
        "model": model_name,
        "target_scale": float(args.target_scale),
        "n_validation_samples": int(len(val_split.y_soh_target)),
        **metrics,
        "best_epoch": int(best_epoch),
        "checkpoint_path": str(checkpoint_path),
    }


def aggregate_results(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_seed = (
        raw_df.groupby(["dataset", "selection_stage", "seed", "horizon", "model"], as_index=False)
        .agg(
            n_validation_samples=("n_validation_samples", "sum"),
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            MAPE_percent=("MAPE_percent", "mean"),
            R2=("R2", "mean"),
            Skill_MAE_vs_persistence=("Skill_MAE_vs_persistence", "mean"),
            checkpoint_path=("checkpoint_path", "first"),
        )
        .sort_values(["horizon", "seed", "MAE", "RMSE", "model"])
    )
    by_seed = add_cpmlp_comparison_columns(
        by_seed,
        group_cols=["dataset", "selection_stage", "seed", "horizon"],
        mae_col="MAE",
        rmse_col="RMSE",
        mape_col="MAPE_percent",
    )

    summary = (
        by_seed.groupby(["dataset", "selection_stage", "horizon", "model"], as_index=False)
        .agg(
            seeds_evaluated=("seed", "nunique"),
            n_validation_samples_mean=("n_validation_samples", "mean"),
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
            RMSE_improvement_vs_cpmlp_mean=("RMSE_improvement_vs_cpmlp", "mean"),
            MAPE_improvement_vs_cpmlp_mean=("MAPE_improvement_vs_cpmlp", "mean"),
            Skill_MAE_vs_cpmlp_mean=("Skill_MAE_vs_cpmlp", "mean"),
        )
        .sort_values(["horizon", "MAE_mean", "RMSE_mean", "model"])
    )

    ranks = summary.copy()
    ranks["rank_MAE"] = ranks.groupby("horizon")["MAE_mean"].rank(method="average", ascending=True)
    ranks["rank_RMSE"] = ranks.groupby("horizon")["RMSE_mean"].rank(method="average", ascending=True)
    ranks["rank_MAPE"] = ranks.groupby("horizon")["MAPE_percent_mean"].rank(method="average", ascending=True)
    ranks["scenario_rank"] = ranks[["rank_MAE", "rank_RMSE", "rank_MAPE"]].mean(axis=1)
    ranks = ranks.sort_values(["horizon", "scenario_rank", "MAE_mean", "RMSE_mean", "model"])
    return by_seed, summary, ranks


def add_cpmlp_comparison_columns(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    mae_col: str,
    rmse_col: str,
    mape_col: str,
) -> pd.DataFrame:
    """Attach CPMLP-anchored metric deltas without affecting model selection."""
    out = df.copy()
    cpmlp = out[out["model"] == "cpmlp"][list(group_cols) + [mae_col, rmse_col, mape_col]].copy()
    cpmlp = cpmlp.rename(
        columns={
            mae_col: "_cpmlp_mae",
            rmse_col: "_cpmlp_rmse",
            mape_col: "_cpmlp_mape",
        }
    )
    out = out.merge(cpmlp, on=list(group_cols), how="left")
    out["MAE_improvement_vs_cpmlp"] = out["_cpmlp_mae"] - out[mae_col]
    out["MAE_improvement_percent_vs_cpmlp"] = np.where(
        out["_cpmlp_mae"] > 0,
        (out["_cpmlp_mae"] - out[mae_col]) / out["_cpmlp_mae"] * 100.0,
        np.nan,
    )
    out["RMSE_improvement_vs_cpmlp"] = out["_cpmlp_rmse"] - out[rmse_col]
    out["MAPE_improvement_vs_cpmlp"] = out["_cpmlp_mape"] - out[mape_col]
    out["Skill_MAE_vs_cpmlp"] = np.where(
        out["_cpmlp_mae"] > 0,
        1.0 - out[mae_col] / out["_cpmlp_mae"],
        np.nan,
    )
    return out.drop(columns=["_cpmlp_mae", "_cpmlp_rmse", "_cpmlp_mape"])


def select_models(summary: pd.DataFrame, horizons: Sequence[int]) -> tuple[pd.DataFrame, dict[str, Any], str]:
    horizon_specific: dict[str, Any] = {}
    for horizon in horizons:
        hdf = summary[summary["horizon"] == int(horizon)].copy()
        if hdf.empty:
            raise ValueError(f"missing validation summary rows for horizon {horizon}")
        hdf = hdf.sort_values(
            [
                "MAE_mean",
                "RMSE_mean",
                "MAPE_percent_mean",
                "MAE_std",
                "Skill_MAE_vs_persistence_mean",
                "model",
            ],
            ascending=[True, True, True, True, False, True],
        )
        best = hdf.iloc[0].to_dict()
        horizon_specific[str(horizon)] = {
            "horizon": int(horizon),
            "selected_model": str(best["model"]),
            "MAE_mean": float(best["MAE_mean"]),
            "RMSE_mean": float(best["RMSE_mean"]),
            "MAPE_percent_mean": float(best["MAPE_percent_mean"]),
            "Skill_MAE_vs_persistence_mean": float(best["Skill_MAE_vs_persistence_mean"]),
            "MAE_improvement_vs_cpmlp_mean": float(best["MAE_improvement_vs_cpmlp_mean"]),
            "MAE_improvement_percent_vs_cpmlp_mean": float(best["MAE_improvement_percent_vs_cpmlp_mean"]),
            "Skill_MAE_vs_cpmlp_mean": float(best["Skill_MAE_vs_cpmlp_mean"]),
        }

    selection = (
        summary.groupby("model", as_index=False)
        .agg(
            avg_MAE_mean=("MAE_mean", "mean"),
            avg_RMSE_mean=("RMSE_mean", "mean"),
            avg_MAPE_percent_mean=("MAPE_percent_mean", "mean"),
            std_MAE_mean=("MAE_mean", "std"),
            worst_MAE_mean=("MAE_mean", "max"),
            average_Skill_MAE_vs_persistence=("Skill_MAE_vs_persistence_mean", "mean"),
            average_MAE_improvement_vs_cpmlp=("MAE_improvement_vs_cpmlp_mean", "mean"),
            average_MAE_improvement_percent_vs_cpmlp=("MAE_improvement_percent_vs_cpmlp_mean", "mean"),
            average_Skill_MAE_vs_cpmlp=("Skill_MAE_vs_cpmlp_mean", "mean"),
            horizons_evaluated=("horizon", "nunique"),
        )
        .copy()
    )
    selection["std_MAE_mean"] = selection["std_MAE_mean"].fillna(0.0)
    expected_horizons = len(set(int(h) for h in horizons))
    missing = selection[selection["horizons_evaluated"] != expected_horizons]
    if not missing.empty:
        raise ValueError(
            "all models must have validation rows for every horizon; incomplete models: "
            + ", ".join(missing["model"].astype(str).tolist())
        )
    selection = selection.sort_values(
        [
            "avg_MAE_mean",
            "avg_RMSE_mean",
            "avg_MAPE_percent_mean",
            "std_MAE_mean",
            "average_Skill_MAE_vs_persistence",
            "model",
        ],
        ascending=[True, True, True, True, False, True],
    ).reset_index(drop=True)
    selected_model = str(selection.iloc[0]["model"])
    selection["selected"] = selection["model"] == selected_model
    selection["selection_rule"] = "lowest_avg_validation_MAE_then_RMSE_MAPE_stdMAE_skill"
    return selection, horizon_specific, selected_model


def assert_no_test_metric_columns(output_dir: Path) -> None:
    metric_tokens = ["mae", "rmse", "mape", "r2", "skill", "metric"]
    expected_csvs = [
        "val_results_raw.csv",
        "val_results_by_seed.csv",
        "val_summary_by_model_horizon.csv",
        "val_rank_by_horizon.csv",
        "model_selection_summary.csv",
    ]
    bad_columns: list[str] = []
    for name in expected_csvs:
        path = output_dir / name
        if not path.exists():
            continue
        columns = pd.read_csv(path, nrows=0).columns
        for column in columns:
            lower = column.lower()
            if "test" in lower and any(token in lower for token in metric_tokens):
                bad_columns.append(f"{name}:{column}")
    if bad_columns:
        raise ValueError("Step 7 output contains test metric columns: " + ", ".join(bad_columns))


def validate_outputs(
    output_dir: Path,
    used_files: Sequence[Path],
    split_manifests: Sequence[dict[str, Any]],
    normalization_config: dict[str, Any],
    selected_payload: dict[str, Any],
    horizons: Sequence[int],
    raw_df: pd.DataFrame,
) -> None:
    assert_matr_only(used_files)
    for manifest in split_manifests:
        split_ids = {
            "train": manifest["train_battery_ids"],
            "validation": manifest["validation_battery_ids"],
            "test": manifest["test_battery_ids"],
        }
        verify_no_split_overlap(split_ids)

    for item in normalization_config.get("normalizers", []):
        if item.get("fitted_split") != "train":
            raise ValueError("normalization statistics must be fitted on train only")
        train_ids = set(item.get("train_battery_ids", []))
        val_ids = set(item.get("validation_battery_ids", []))
        test_ids = set(item.get("test_battery_ids", []))
        if train_ids & val_ids or train_ids & test_ids:
            raise ValueError("normalization train IDs overlap validation/test IDs")

    if selected_payload.get("test_metrics_used") is not False:
        raise ValueError("selected_model.json must set test_metrics_used=false")
    forbidden_selected_keys = [
        key
        for key in selected_payload
        if key.lower().startswith("test_") and key != "test_metrics_used"
    ]
    if forbidden_selected_keys:
        raise ValueError(
            "selected_model.json includes forbidden test metric keys: "
            + ", ".join(forbidden_selected_keys)
        )

    assert_no_test_metric_columns(output_dir)
    missing_persistence = [
        int(horizon)
        for horizon in horizons
        if raw_df[(raw_df["horizon"] == int(horizon)) & (raw_df["model"] == "persistence")].empty
    ]
    if missing_persistence:
        raise ValueError(
            "persistence baseline is missing for horizons: "
            + ", ".join(str(horizon) for horizon in missing_persistence)
        )


def apply_debug_overrides(args: argparse.Namespace) -> None:
    if not args.debug:
        return
    args.epochs = min(args.epochs, 3)
    args.patience = min(args.patience, 2)
    args.fixed_len = min(args.fixed_len, 40)
    args.batch_size = min(args.batch_size, 16)
    args.mlp_embed_dim = min(args.mlp_embed_dim, 16)
    args.gru_embed_dim = min(args.gru_embed_dim, 16)
    args.model_hidden = min(args.model_hidden, 64)
    args.gru_hidden = min(args.gru_hidden, 16)
    args.dsconv_channels = min(args.dsconv_channels, 16)


def run(args: argparse.Namespace) -> None:
    apply_debug_overrides(args)
    if args.target_scale <= 0:
        raise ValueError("--target-scale must be positive")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if args.lookback != 20:
        raise ValueError("Step 7 requires --lookback 20")
    horizons = [int(horizon) for horizon in args.horizons]
    seeds = [int(seed) for seed in args.seeds]
    models = [str(model).lower() for model in args.models]
    unknown = sorted(set(models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"unsupported Step 7 models: {unknown}; allowed={DEFAULT_MODELS}")
    if "persistence" not in models:
        raise ValueError("persistence baseline must be included in --models")

    matr_files, excluded_files = find_matr_files(data_root)
    records, dataset_manifest = load_dataset_manifest(
        matr_files,
        excluded_files,
        data_root=data_root,
        lookback=args.lookback,
        horizons=horizons,
        fixed_len=args.fixed_len,
    )
    records_by_id = {record.battery_id: record for record in records}

    config = {
        "dataset": DATASET,
        "selection_stage": SELECTION_STAGE,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "lookback_cycles": args.lookback,
        "horizons": horizons,
        "seeds": seeds,
        "models": models,
        "features": FEATURES,
        "target": TARGET,
        "target_scale": args.target_scale,
        "fixed_len": args.fixed_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": "Adam",
        "loss": "MSE(delta_soh)",
        "early_stopping_metric": "validation_MAE_reconstructed_SOH",
        "patience": args.patience,
        "min_delta": args.min_delta,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "mlp_embed_dim": args.mlp_embed_dim,
        "gru_embed_dim": args.gru_embed_dim,
        "model_hidden": args.model_hidden,
        "gru_hidden": args.gru_hidden,
        "dsconv_channels": args.dsconv_channels,
        "zero_output_init": args.zero_output_init,
        "debug": args.debug,
        "test_metrics_used": False,
        "selection_rule": "lowest_avg_validation_MAE_then_RMSE_MAPE_stdMAE_skill",
    }
    write_simple_yaml(output_dir / "config.yaml", config)
    save_json(output_dir / "dataset_manifest.json", dataset_manifest)

    raw_rows: list[dict[str, Any]] = []
    normalizers: list[dict[str, Any]] = []
    split_manifests: list[dict[str, Any]] = []

    for seed in seeds:
        pipe.set_seed(seed)
        split_ids = split_battery_ids(records, seed=seed)
        split_manifest = make_split_manifest(
            seed=seed,
            split_ids=split_ids,
            records_by_id=records_by_id,
            horizons=horizons,
            lookback=args.lookback,
        )
        save_json(output_dir / f"split_manifest_seed{seed}.json", split_manifest)
        split_manifests.append(split_manifest)

        for horizon in horizons:
            train_split = build_horizon_split(
                records_by_id, split_ids["train"], "train", horizon, args.lookback
            )
            val_split = build_horizon_split(
                records_by_id, split_ids["validation"], "validation", horizon, args.lookback
            )
            test_split = build_horizon_split(
                records_by_id, split_ids["test"], "test", horizon, args.lookback
            )
            ensure_non_empty_split(train_split, "train", seed, horizon)
            ensure_non_empty_split(val_split, "validation", seed, horizon)

            mean, std = fit_train_normalizer(train_split.X)
            normalizers.append(
                {
                    "seed": seed,
                    "horizon": horizon,
                    "fitted_split": "train",
                    "feature_order": FEATURES,
                    "mean": mean.reshape(-1).astype(float).tolist(),
                    "std": std.reshape(-1).astype(float).tolist(),
                    "train_battery_ids": [item["battery_id"] for item in train_split.meta],
                    "validation_battery_ids": [item["battery_id"] for item in val_split.meta],
                    "test_battery_ids": [item["battery_id"] for item in test_split.meta],
                    "n_train_samples": int(len(train_split.y_soh_target)),
                    "n_validation_samples": int(len(val_split.y_soh_target)),
                    "n_test_samples_manifest_only": int(len(test_split.y_soh_target)),
                }
            )

            checkpoint_dir = checkpoint_root / f"seed{seed}" / f"horizon{horizon}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            persistence_row, persistence_mae = evaluate_persistence(
                seed=seed,
                horizon=horizon,
                val_split=val_split,
                checkpoint_dir=checkpoint_dir,
            )
            raw_rows.append(persistence_row)

            for model_name in models:
                if model_name == "persistence":
                    continue
                row = evaluate_neural_model(
                    model_name=model_name,
                    seed=seed,
                    horizon=horizon,
                    lookback=args.lookback,
                    fixed_len=args.fixed_len,
                    train_split=train_split,
                    val_split=val_split,
                    mean=mean,
                    std=std,
                    persistence_mae=persistence_mae,
                    args=args,
                    checkpoint_dir=checkpoint_dir,
                )
                raw_rows.append(row)

            raw_df_partial = pd.DataFrame(raw_rows)
            raw_df_partial.to_csv(output_dir / "val_results_raw.csv", index=False)

    normalization_config = {
        "dataset": DATASET,
        "selection_stage": SELECTION_STAGE,
        "capacity_used_as_input": False,
        "features": FEATURES,
        "fit_policy": "train split only, separately per seed and horizon",
        "normalizers": normalizers,
    }
    save_json(output_dir / "normalization_config.json", normalization_config)

    raw_df = pd.DataFrame(raw_rows).sort_values(["horizon", "seed", "MAE", "RMSE", "model"])
    raw_df.to_csv(output_dir / "val_results_raw.csv", index=False)
    by_seed, summary, ranks = aggregate_results(raw_df)
    by_seed.to_csv(output_dir / "val_results_by_seed.csv", index=False)
    summary.to_csv(output_dir / "val_summary_by_model_horizon.csv", index=False)
    ranks.to_csv(output_dir / "val_rank_by_horizon.csv", index=False)

    selection, horizon_specific_best, selected_model = select_models(summary, horizons)
    selection.to_csv(output_dir / "model_selection_summary.csv", index=False)
    save_json(output_dir / "horizon_specific_best.json", horizon_specific_best)

    selected_payload = {
        "dataset": DATASET,
        "selection_stage": SELECTION_STAGE,
        "test_metrics_used": False,
        "lookback_cycles": args.lookback,
        "horizons": horizons,
        "features": FEATURES,
        "target": TARGET,
        "target_scale": args.target_scale,
        "selection_rule": "lowest_avg_validation_MAE_then_RMSE_MAPE_stdMAE_skill",
        "rank_used_for_selection": False,
        "selected_model": selected_model,
        "horizon_specific_best": horizon_specific_best,
    }
    save_json(output_dir / "selected_model.json", selected_payload)
    create_readme(output_dir / "README.md", config, selected_model)

    validate_outputs(
        output_dir=output_dir,
        used_files=[record.file_path for record in records],
        split_manifests=split_manifests,
        normalization_config=normalization_config,
        selected_payload=selected_payload,
        horizons=horizons,
        raw_df=raw_df,
    )

    print("\n=== MATR Step 7 validation-only selection complete ===")
    print(selection.to_string(index=False))
    print(f"\nselected_model={selected_model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MATR-only Step 7 future SOH validation model selection."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--fixed-len", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--target-scale",
        type=float,
        default=1.0,
        help="Scale delta_soh targets during neural training; SOH reconstruction divides predictions by this value.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--mlp-embed-dim", type=int, default=64)
    parser.add_argument("--gru-embed-dim", type=int, default=64)
    parser.add_argument("--model-hidden", type=int, default=256)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dsconv-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--zero-output-init", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
