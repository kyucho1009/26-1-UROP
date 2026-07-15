from __future__ import annotations

"""Battery-level repeated out-of-fold prediction for the locked MATR model.

This runner deliberately does not reuse the old locked-test checkpoints.  For
each outer fold it trains a fresh copy of the already selected model while the
outer batteries remain absent from training, normalization, and early
stopping. Outer folds balance protocol batch and, within each batch, the rank
distribution of the last observed cycle. Splitting always happens at battery
level before sliding windows are created.

The module also exposes the pure split helpers ``assign_stratified_folds`` and
``split_inner_train_validation`` so their leakage and balance properties can be
unit-tested without training a neural network.
"""

import argparse
import gc
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_matr_locked_test_evaluation as locked_eval
import scripts.run_matr_step7_validation_selection as step7

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyTorch is required for MATR out-of-fold cross-validation."
    ) from exc


STAGE = "oof_cross_validation"
DEFAULT_N_SPLITS = 5
DEFAULT_REPEAT_SEEDS = (42, 43, 44)
ROLE_TRAIN = "inner_train"
ROLE_VALIDATION = "inner_validation"
ROLE_OUTER = "outer_oof"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{context} is missing columns: {missing}")


def _stable_json(payload: Any) -> str:
    return json.dumps(
        step7.json_sanitize(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            step7.json_sanitize(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def battery_table_from_records(
    records: Sequence[step7.BatteryRecord],
) -> pd.DataFrame:
    """Return one identity row per physical battery."""

    rows: list[dict[str, Any]] = []
    for record in records:
        if not record.available_cycles:
            raise ValueError(
                f"Battery {record.battery_id} has no available SOH cycles"
            )
        rows.append(
            {
                "battery_id": str(record.battery_id),
                "cell_id": str(record.cell_id),
                "batch_id": str(record.batch_id),
                "observed_cycle_end": int(max(record.available_cycles)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No battery records were supplied")
    if frame["battery_id"].duplicated().any():
        duplicated = sorted(
            frame.loc[frame["battery_id"].duplicated(False), "battery_id"].unique()
        )
        raise ValueError(f"Duplicate battery IDs: {duplicated[:10]}")
    return frame.sort_values("battery_id").reset_index(drop=True)


def assign_stratified_folds(
    battery_table: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign every battery to a batch-and-observed-length-stratified fold.

    Each batch contributes either ``floor(n_batch / n_splits)`` or
    ``ceil(n_batch / n_splits)`` batteries to every fold. Within a batch,
    batteries are ordered by their last observed cycle and divided into local
    blocks of ``n_splits`` adjacent ranks. Every full rank block contributes
    exactly one battery to every fold. This balances short-, medium-, and
    long-observed cells without discretizing a scientifically meaningful life
    threshold. Assignment is deterministic for a given table and seed. No
    window-level data are accepted by this function.
    """

    if int(n_splits) < 3:
        raise ValueError("n_splits must be at least 3")
    _require_columns(
        battery_table,
        ["battery_id", "batch_id", "observed_cycle_end"],
        "battery fold table",
    )
    frame = battery_table.copy()
    frame["battery_id"] = frame["battery_id"].astype(str)
    frame["batch_id"] = frame["batch_id"].astype(str)
    frame["observed_cycle_end"] = pd.to_numeric(
        frame["observed_cycle_end"], errors="coerce"
    )
    if frame.empty:
        raise ValueError("battery fold table is empty")
    if frame["battery_id"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["battery_id"].duplicated(False), "battery_id"].unique()
        )
        raise ValueError(
            "Fold assignment requires one row per battery; duplicates="
            + repr(duplicates[:10])
        )
    observed_end = frame["observed_cycle_end"].to_numpy(dtype=float)
    if not np.isfinite(observed_end).all() or (observed_end < 1).any():
        raise ValueError("observed_cycle_end must contain finite positive values")
    batch_sizes = frame.groupby("batch_id")["battery_id"].nunique()
    too_small = batch_sizes[batch_sizes < int(n_splits)]
    if not too_small.empty:
        raise ValueError(
            "Every batch must have at least n_splits batteries for strict "
            f"stratification; too_small={too_small.to_dict()}"
        )

    frame = frame.sort_values(["batch_id", "battery_id"]).reset_index(drop=True)
    frame["outer_fold"] = -1
    frame["repeat_seed"] = int(seed)
    frame["observed_length_stratum"] = -1
    rng = np.random.default_rng(int(seed))
    total_fold_load = np.zeros(int(n_splits), dtype=int)

    for _, group in frame.groupby("batch_id", sort=True):
        ordered = group.copy()
        ordered["_seeded_tie_break"] = rng.random(len(ordered))
        ordered = ordered.sort_values(
            ["observed_cycle_end", "_seeded_tie_break", "battery_id"]
        )
        ordered_indices = ordered.index.to_numpy(dtype=int)
        batch_fold_load = np.zeros(int(n_splits), dtype=int)

        for stratum, start in enumerate(range(0, len(ordered_indices), int(n_splits))):
            block = ordered_indices[start : start + int(n_splits)]
            if len(block) == int(n_splits):
                fold_order = rng.permutation(int(n_splits)).astype(int).tolist()
            else:
                tie_break = rng.random(int(n_splits))
                fold_order = sorted(
                    range(int(n_splits)),
                    key=lambda fold: (
                        int(batch_fold_load[fold]),
                        int(total_fold_load[fold]),
                        float(tie_break[fold]),
                    ),
                )[: len(block)]

            # Randomize which adjacent rank goes to which selected fold while
            # retaining one-per-fold membership within this local life block.
            block_order = block[rng.permutation(len(block))]
            for index, fold in zip(block_order, fold_order):
                frame.loc[int(index), "outer_fold"] = int(fold)
                frame.loc[int(index), "observed_length_stratum"] = int(stratum)
                batch_fold_load[int(fold)] += 1
                total_fold_load[int(fold)] += 1

    if frame["outer_fold"].lt(0).any():
        raise RuntimeError("Some batteries did not receive an outer fold")

    per_batch = frame.groupby(["batch_id", "outer_fold"]).size().unstack(fill_value=0)
    if (per_batch.max(axis=1) - per_batch.min(axis=1)).gt(1).any():
        raise RuntimeError("Batch-stratified fold counts differ by more than one")
    stratum_fold_counts = frame.groupby(
        ["batch_id", "observed_length_stratum", "outer_fold"]
    ).size()
    if stratum_fold_counts.gt(1).any():
        raise RuntimeError(
            "An observed-length rank stratum assigned multiple batteries to one fold"
        )
    for _, group in frame.groupby(
        ["batch_id", "observed_length_stratum"], sort=True
    ):
        if len(group) == int(n_splits) and group["outer_fold"].nunique() != int(n_splits):
            raise RuntimeError(
                "A complete observed-length stratum does not cover every fold"
            )
    if frame.groupby("battery_id")["outer_fold"].nunique().ne(1).any():
        raise RuntimeError("A battery was assigned to more than one outer fold")

    return frame.sort_values("battery_id").reset_index(drop=True)


def split_inner_train_validation(
    fold_assignments: pd.DataFrame,
    *,
    outer_fold: int,
    validation_fraction: float = 0.20,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create batch-and-observed-length-balanced inner/outer battery roles."""

    if not 0.0 < float(validation_fraction) < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    _require_columns(
        fold_assignments,
        ["battery_id", "batch_id", "observed_cycle_end", "outer_fold"],
        "outer fold assignments",
    )
    frame = fold_assignments.copy()
    frame["battery_id"] = frame["battery_id"].astype(str)
    frame["batch_id"] = frame["batch_id"].astype(str)
    frame["observed_cycle_end"] = pd.to_numeric(
        frame["observed_cycle_end"], errors="coerce"
    )
    if not np.isfinite(frame["observed_cycle_end"].to_numpy(dtype=float)).all():
        raise ValueError("observed_cycle_end must be finite for inner splitting")
    if frame["battery_id"].duplicated().any():
        raise ValueError("fold_assignments must contain one row per battery")
    observed_folds = sorted(frame["outer_fold"].astype(int).unique())
    if int(outer_fold) not in observed_folds:
        raise ValueError(
            f"outer_fold={outer_fold} is absent; observed folds={observed_folds}"
        )

    outer_ids = sorted(
        frame.loc[frame["outer_fold"].astype(int) == int(outer_fold), "battery_id"]
        .astype(str)
        .tolist()
    )
    pool = frame[frame["outer_fold"].astype(int) != int(outer_fold)].copy()
    rng = np.random.default_rng(int(seed))
    validation_ids: list[str] = []

    for batch_id, group in pool.groupby("batch_id", sort=True):
        ordered = group.sort_values(
            ["observed_cycle_end", "battery_id"]
        ).reset_index(drop=True)
        if len(ordered) < 2:
            raise ValueError(
                f"Batch {batch_id} has fewer than two non-outer batteries"
            )
        n_validation = int(round(len(ordered) * float(validation_fraction)))
        n_validation = min(max(n_validation, 1), len(ordered) - 1)

        # Split the within-batch observed-length ranks into equal contiguous
        # bins and sample one battery per bin.  Early stopping/calibration
        # therefore sees short, middle, and long-observed cells instead of a
        # purely random subset whose coverage can drift in a small cohort.
        rank_bins = np.array_split(np.arange(len(ordered)), n_validation)
        selected_indices = [
            int(rank_bin[int(rng.integers(0, len(rank_bin)))])
            for rank_bin in rank_bins
        ]
        validation_ids.extend(
            ordered.iloc[selected_indices]["battery_id"].astype(str).tolist()
        )

    validation_ids = sorted(validation_ids)
    pool_ids = set(pool["battery_id"].astype(str))
    train_ids = sorted(pool_ids - set(validation_ids))
    roles = {
        ROLE_TRAIN: train_ids,
        ROLE_VALIDATION: validation_ids,
        ROLE_OUTER: outer_ids,
    }
    validate_role_partition(frame["battery_id"].astype(str), roles)

    # The outer and inner validation roles must preserve every represented batch.
    id_to_batch = frame.set_index("battery_id")["batch_id"].astype(str)
    all_batches = set(frame["batch_id"].astype(str))
    for role in [ROLE_VALIDATION, ROLE_OUTER]:
        represented = set(id_to_batch.reindex(roles[role]).dropna().astype(str))
        if represented != all_batches:
            raise RuntimeError(
                f"Role {role} does not preserve all batches: {sorted(represented)}"
            )
    return roles


def validate_role_partition(
    all_battery_ids: Sequence[str] | pd.Series,
    roles: Mapping[str, Sequence[str]],
) -> None:
    """Validate that the three battery roles are exhaustive and disjoint."""

    required_roles = {ROLE_TRAIN, ROLE_VALIDATION, ROLE_OUTER}
    if set(roles) != required_roles:
        raise ValueError(
            f"Expected roles={sorted(required_roles)}, got={sorted(roles)}"
        )
    all_ids = {str(value) for value in all_battery_ids}
    role_sets = {key: {str(value) for value in values} for key, values in roles.items()}
    if any(len(values) == 0 for values in role_sets.values()):
        raise ValueError("Every battery role must be non-empty")
    for left, right in [
        (ROLE_TRAIN, ROLE_VALIDATION),
        (ROLE_TRAIN, ROLE_OUTER),
        (ROLE_VALIDATION, ROLE_OUTER),
    ]:
        overlap = sorted(role_sets[left] & role_sets[right])
        if overlap:
            raise ValueError(f"Battery overlap between {left}/{right}: {overlap[:10]}")
    assigned = set().union(*role_sets.values())
    if assigned != all_ids:
        raise ValueError(
            "Battery roles are not exhaustive; "
            f"missing={sorted(all_ids - assigned)[:10]}, "
            f"unexpected={sorted(assigned - all_ids)[:10]}"
        )


def derive_seed(repeat_seed: int, outer_fold: int, horizon: int, *, offset: int = 0) -> int:
    """Derive a deterministic positive 31-bit seed for one training job."""

    sequence = np.random.SeedSequence(
        [int(repeat_seed), int(outer_fold), int(horizon), int(offset)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0] % np.uint32(2**31 - 1))


def _resolve_runtime_config(
    locked_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if bool(locked_config.get("test_metrics_used", False)):
        raise ValueError("The source config says test metrics were used for selection")
    best = locked_config.get("best", {})
    if not isinstance(best, dict):
        best = {}
    hyper = (
        locked_config.get("hyperparameters", {})
        or locked_config.get("selected_hyperparameters", {})
        or best
    )
    selected_model = str(
        locked_config.get("selected_model") or best.get("model") or ""
    ).lower()
    if not selected_model:
        raise ValueError("Locked config must define selected_model or best.model")
    if selected_model not in step7.DEFAULT_MODELS:
        raise ValueError(
            f"Unsupported selected model={selected_model!r}; allowed={step7.DEFAULT_MODELS}"
        )

    batches = step7.parse_batch_filter(locked_config.get("batches"))
    if batches:
        raise ValueError(
            "Full OOF analysis cannot use a batch-filtered locked config; "
            f"configured batches={batches}"
        )
    sample_mode = str(
        args.sample_mode
        or locked_config.get("sample_mode")
        or step7.SAMPLE_MODE_FIRST_WINDOW
    )
    if sample_mode not in step7.SAMPLE_MODES:
        raise ValueError(f"Unsupported sample_mode={sample_mode!r}")
    horizons = [
        int(value)
        for value in (
            args.horizons
            or locked_eval.config_sequence(
                locked_config,
                ["horizons", "confirm_horizons", "search_horizons"],
                [10, 50, 100],
            )
        )
    ]
    if len(horizons) == 0 or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be non-empty and distinct")

    runtime = {
        "selected_model": selected_model,
        "lookback": int(
            args.lookback
            or locked_config.get("lookback_cycles")
            or locked_config.get("lookback")
            or 20
        ),
        "sample_mode": sample_mode,
        "horizons": horizons,
        "target_scale": float(
            args.target_scale
            or locked_eval.config_value(locked_config, "target_scale", 1.0)
        ),
        "fixed_len": int(
            args.fixed_len
            or locked_eval.config_value(locked_config, "fixed_len", 100)
        ),
        "batch_size": int(hyper.get("batch_size", 16)),
        "dropout": float(hyper.get("dropout", 0.1)),
        "dsconv_channels": int(hyper.get("dsconv_channels", 64)),
        "epochs": int(hyper.get("epochs", 100)),
        "gru_embed_dim": int(hyper.get("gru_embed_dim", 64)),
        "gru_hidden": int(hyper.get("gru_hidden", 64)),
        "lr": float(hyper.get("lr", 3e-4)),
        "mlp_embed_dim": int(hyper.get("mlp_embed_dim", 64)),
        "model_hidden": int(hyper.get("model_hidden", 256)),
        "patience": int(hyper.get("patience", 12)),
        "weight_decay": float(hyper.get("weight_decay", 1e-5)),
        "clip_grad_norm": float(hyper.get("clip_grad_norm", 1.0)),
    }
    if runtime["lookback"] < 1 or runtime["fixed_len"] < 5:
        raise ValueError("lookback/fixed_len are invalid")
    if runtime["target_scale"] <= 0:
        raise ValueError("target_scale must be positive")
    if args.debug:
        runtime["epochs"] = min(runtime["epochs"], 3)
        runtime["patience"] = min(runtime["patience"], 2)
        runtime["fixed_len"] = min(runtime["fixed_len"], 40)
        runtime["batch_size"] = min(runtime["batch_size"], 16)
        runtime["mlp_embed_dim"] = min(runtime["mlp_embed_dim"], 16)
        runtime["gru_embed_dim"] = min(runtime["gru_embed_dim"], 16)
        runtime["model_hidden"] = min(runtime["model_hidden"], 64)
        runtime["gru_hidden"] = min(runtime["gru_hidden"], 16)
        runtime["dsconv_channels"] = min(runtime["dsconv_channels"], 16)
    return runtime


def _role_rows(
    assignments: pd.DataFrame,
    roles_by_fold: Mapping[int, Mapping[str, Sequence[str]]],
) -> pd.DataFrame:
    metadata = assignments.set_index("battery_id")
    rows: list[dict[str, Any]] = []
    repeat_seed = int(assignments["repeat_seed"].iloc[0])
    for outer_fold, roles in sorted(roles_by_fold.items()):
        for role, battery_ids in roles.items():
            for battery_id in battery_ids:
                row = metadata.loc[str(battery_id)]
                rows.append(
                    {
                        "repeat_seed": repeat_seed,
                        "outer_fold": int(outer_fold),
                        "battery_id": str(battery_id),
                        "cell_id": str(row.get("cell_id", battery_id)),
                        "batch_id": str(row["batch_id"]),
                        "observed_cycle_end": int(row["observed_cycle_end"]),
                        "observed_length_stratum": int(
                            row["observed_length_stratum"]
                        ),
                        "role": str(role),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["repeat_seed", "outer_fold", "role", "batch_id", "battery_id"]
    ).reset_index(drop=True)


def _observed_length_summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
    values = pd.to_numeric(frame["observed_cycle_end"], errors="coerce").dropna()
    if values.empty:
        return {
            "n_batteries": 0,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "mean": None,
        }
    return {
        "n_batteries": int(len(values)),
        "min": int(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def _build_repeat_manifest(
    assignments: pd.DataFrame,
    roles_by_fold: Mapping[int, Mapping[str, Sequence[str]]],
    *,
    n_splits: int,
    validation_fraction: float,
) -> dict[str, Any]:
    repeat_seed = int(assignments["repeat_seed"].iloc[0])
    role_frame = _role_rows(assignments, roles_by_fold)
    observed_length_by_batch = {
        str(batch_id): _observed_length_summary(group)
        for batch_id, group in assignments.groupby("batch_id", sort=True)
    }
    outer_length_audit = [
        {
            "outer_fold": int(outer_fold),
            "batch_id": str(batch_id),
            **_observed_length_summary(group),
        }
        for (outer_fold, batch_id), group in assignments.groupby(
            ["outer_fold", "batch_id"], sort=True
        )
    ]
    folds: list[dict[str, Any]] = []
    for outer_fold, roles in sorted(roles_by_fold.items()):
        one = role_frame[role_frame["outer_fold"] == int(outer_fold)]
        batch_counts = {
            role: dict(
                sorted(
                    one[one["role"] == role]["batch_id"]
                    .value_counts()
                    .astype(int)
                    .to_dict()
                    .items()
                )
            )
            for role in [ROLE_TRAIN, ROLE_VALIDATION, ROLE_OUTER]
        }
        folds.append(
            {
                "outer_fold": int(outer_fold),
                "inner_split_seed": derive_seed(
                    repeat_seed, int(outer_fold), 0, offset=17
                ),
                "inner_train_battery_ids": list(roles[ROLE_TRAIN]),
                "inner_validation_battery_ids": list(roles[ROLE_VALIDATION]),
                "outer_oof_battery_ids": list(roles[ROLE_OUTER]),
                "counts": {role: len(roles[role]) for role in roles},
                "batch_counts_by_role": batch_counts,
                "outer_observed_cycle_end_by_batch": {
                    str(batch_id): _observed_length_summary(group)
                    for batch_id, group in assignments[
                        assignments["outer_fold"].astype(int) == int(outer_fold)
                    ].groupby("batch_id", sort=True)
                },
                "inner_validation_observed_cycle_end_by_batch": {
                    str(batch_id): _observed_length_summary(group)
                    for batch_id, group in one[
                        one["role"] == ROLE_VALIDATION
                    ].groupby("batch_id", sort=True)
                },
            }
        )
    return {
        "dataset": step7.DATASET,
        "stage": STAGE,
        "repeat_seed": repeat_seed,
        "n_splits": int(n_splits),
        "split_level": "battery",
        "outer_stratification": {
            "categorical": "MATR batch_id",
            "continuous": "observed_cycle_end (max available cycle)",
            "method": (
                "within each batch, sort by observed_cycle_end, form adjacent "
                "rank blocks of n_splits, and allocate at most one cell per fold "
                "from every block"
            ),
            "batch_count_balance_guarantee": "per-batch fold counts differ by at most 1",
        },
        "inner_validation_fraction_of_non_outer_pool": float(validation_fraction),
        "inner_validation_stratification": {
            "categorical": "MATR batch_id",
            "continuous": "observed_cycle_end (max available cycle)",
            "method": (
                "within each batch, divide observed-length ranks into equal "
                "contiguous bins and sample one battery per bin"
            ),
            "reuse_note": (
                "used for early stopping and then reused for score-transform "
                "calibration; outer OOF batteries remain untouched"
            ),
        },
        "all_battery_ids": sorted(assignments["battery_id"].astype(str)),
        "batch_counts": dict(
            sorted(assignments["batch_id"].value_counts().astype(int).to_dict().items())
        ),
        "observed_cycle_end_by_batch": observed_length_by_batch,
        "observed_cycle_end_by_outer_fold_and_batch": outer_length_audit,
        "outer_assignments": assignments[
            [
                "battery_id",
                "cell_id",
                "batch_id",
                "observed_cycle_end",
                "observed_length_stratum",
                "outer_fold",
            ]
        ].to_dict("records"),
        "folds": folds,
    }


def _prediction_frame(
    split: step7.HorizonSplit,
    *,
    repeat_seed: int,
    outer_fold: int,
    training_seed: int,
    model_name: str,
    pred_soh: np.ndarray,
    pred_delta: np.ndarray,
    best_epoch: int,
    run_id: str,
) -> pd.DataFrame:
    frame = locked_eval.prediction_frame(
        split,
        seed=int(repeat_seed),
        model=model_name,
        pred_soh=pred_soh,
        pred_delta=pred_delta,
    )
    frame["stage"] = STAGE
    frame["repeat_seed"] = int(repeat_seed)
    frame["outer_fold"] = int(outer_fold)
    frame["training_seed"] = int(training_seed)
    frame["best_epoch"] = int(best_epoch)
    frame["normalizer_fit_role"] = ROLE_TRAIN
    frame["run_id"] = str(run_id)
    return frame


def _cell_macro_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    persistence_rows: list[dict[str, float]] = []
    for _, group in predictions.groupby("battery_id", sort=True):
        actual = group["actual_soh"].to_numpy(dtype=float)
        predicted = group["pred_soh"].to_numpy(dtype=float)
        persistence = group["current_soh"].to_numpy(dtype=float)
        rows.append(step7.compute_metrics(actual, predicted))
        persistence_rows.append(step7.compute_metrics(actual, persistence))
    metrics = pd.DataFrame(rows)
    persistence_metrics = pd.DataFrame(persistence_rows)
    macro_mae = float(metrics["MAE"].mean())
    persistence_macro_mae = float(persistence_metrics["MAE"].mean())
    skill = (
        float(1.0 - macro_mae / persistence_macro_mae)
        if persistence_macro_mae > 0
        else float("nan")
    )
    return {
        "macro_cell_MAE": macro_mae,
        "macro_cell_RMSE": float(metrics["RMSE"].mean()),
        "macro_cell_MAPE_percent": float(metrics["MAPE_percent"].mean()),
        "macro_cell_persistence_MAE": persistence_macro_mae,
        "macro_cell_Skill_MAE_vs_persistence": skill,
    }


def prediction_metrics_row(
    predictions: pd.DataFrame,
    *,
    best_epoch: int | float | None = None,
    checkpoint_path: str = "",
) -> dict[str, Any]:
    """Compute both window-weighted and equal-cell metrics."""

    _require_columns(
        predictions,
        [
            "repeat_seed",
            "outer_fold",
            "split",
            "horizon",
            "model",
            "battery_id",
            "actual_soh",
            "pred_soh",
            "current_soh",
        ],
        "OOF prediction metrics",
    )
    identifying = ["repeat_seed", "outer_fold", "split", "horizon", "model"]
    identity: dict[str, Any] = {}
    for column in identifying:
        values = predictions[column].drop_duplicates()
        if len(values) != 1:
            raise ValueError(
                f"prediction_metrics_row expected one {column}, got {values.tolist()}"
            )
        identity[column] = values.iloc[0]
    actual = predictions["actual_soh"].to_numpy(dtype=float)
    predicted = predictions["pred_soh"].to_numpy(dtype=float)
    persistence = predictions["current_soh"].to_numpy(dtype=float)
    persistence_mae = step7.compute_metrics(actual, persistence)["MAE"]
    window_metrics = step7.metric_row_with_skill(
        actual, predicted, persistence_mae
    )
    return {
        "dataset": step7.DATASET,
        "stage": STAGE,
        **identity,
        "n_batteries": int(predictions["battery_id"].nunique()),
        "n_windows": int(len(predictions)),
        **window_metrics,
        **_cell_macro_metrics(predictions),
        "best_epoch": (
            int(best_epoch)
            if best_epoch is not None and math.isfinite(float(best_epoch))
            else None
        ),
        "checkpoint_path": str(checkpoint_path),
    }


def _checkpoint_payload(
    *,
    model: Any,
    cfg: dict[str, Any],
    locked_config: dict[str, Any],
    repeat_seed: int,
    outer_fold: int,
    horizon: int,
    training_seed: int,
    roles: Mapping[str, Sequence[str]],
    mean: np.ndarray,
    std: np.ndarray,
    best_epoch: int,
    best_validation_mae: float,
    run_id: str,
) -> dict[str, Any]:
    return {
        "dataset": step7.DATASET,
        "stage": STAGE,
        "run_id": run_id,
        "test_metrics_used_for_selection": False,
        "model_selection_performed": False,
        "hyperparameter_tuning_performed": False,
        "model": cfg["selected_model"],
        "repeat_seed": int(repeat_seed),
        "outer_fold": int(outer_fold),
        "horizon": int(horizon),
        "training_seed": int(training_seed),
        "runtime_config": cfg,
        "locked_config": locked_config,
        "inner_train_battery_ids": list(roles[ROLE_TRAIN]),
        "inner_validation_battery_ids": list(roles[ROLE_VALIDATION]),
        "outer_oof_battery_ids": list(roles[ROLE_OUTER]),
        "normalization_fitted_on": ROLE_TRAIN,
        "normalization_mean": mean.astype(np.float32),
        "normalization_std": std.astype(np.float32),
        "best_epoch": int(best_epoch),
        "best_inner_validation_mae_reconstructed_soh": float(best_validation_mae),
        "model_state_dict": model.state_dict(),
    }


def _save_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _completed_chunk_is_valid(
    chunk_dir: Path,
    *,
    run_id: str,
    repeat_seed: int,
    outer_fold: int,
    horizon: int,
) -> bool:
    completion_path = chunk_dir / "complete.json"
    if not completion_path.is_file():
        return False
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected = {
        "run_id": str(run_id),
        "repeat_seed": int(repeat_seed),
        "outer_fold": int(outer_fold),
        "horizon": int(horizon),
    }
    mismatches = {
        key: (completion.get(key), value)
        for key, value in expected.items()
        if completion.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Completed chunk metadata mismatch in {completion_path}: {mismatches}"
        )
    outer_path = chunk_dir / "oof_predictions.csv"
    inner_path = chunk_dir / "inner_validation_predictions.csv"
    metrics_path = chunk_dir / "metrics.csv"
    checkpoint_path = chunk_dir / str(completion.get("checkpoint_file", ""))
    history_path = chunk_dir / str(completion.get("history_file", ""))
    required = [
        outer_path,
        inner_path,
        metrics_path,
        checkpoint_path,
        history_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print(
            "[resume] completion marker has missing artifacts; retraining chunk: "
            + ", ".join(missing),
            flush=True,
        )
        return False
    required_prediction_columns = {
        "run_id",
        "stage",
        "repeat_seed",
        "outer_fold",
        "normalizer_fit_role",
    }
    for prediction_path in [outer_path, inner_path]:
        header = pd.read_csv(prediction_path, nrows=1)
        missing_columns = required_prediction_columns - set(header.columns)
        if missing_columns:
            print(
                "[resume] prediction schema is stale; retraining chunk: "
                f"{prediction_path} missing={sorted(missing_columns)}",
                flush=True,
            )
            return False
        if set(header["run_id"].astype(str)) != {str(run_id)}:
            raise RuntimeError(
                f"Prediction run_id mismatch in {prediction_path}"
            )
    return True


def _run_training_chunk(
    *,
    records_by_id: dict[str, step7.BatteryRecord],
    roles: Mapping[str, Sequence[str]],
    cfg: dict[str, Any],
    locked_config: dict[str, Any],
    repeat_seed: int,
    outer_fold: int,
    horizon: int,
    device: str,
    run_id: str,
    chunk_dir: Path,
) -> None:
    training_seed = derive_seed(repeat_seed, outer_fold, horizon, offset=101)
    step7.pipe.set_seed(training_seed)
    train_split = step7.build_horizon_split(
        records_by_id,
        roles[ROLE_TRAIN],
        ROLE_TRAIN,
        int(horizon),
        cfg["lookback"],
        cfg["sample_mode"],
    )
    validation_split = step7.build_horizon_split(
        records_by_id,
        roles[ROLE_VALIDATION],
        ROLE_VALIDATION,
        int(horizon),
        cfg["lookback"],
        cfg["sample_mode"],
    )
    outer_split = step7.build_horizon_split(
        records_by_id,
        roles[ROLE_OUTER],
        "oof",
        int(horizon),
        cfg["lookback"],
        cfg["sample_mode"],
    )
    step7.ensure_non_empty_split(train_split, ROLE_TRAIN, repeat_seed, horizon)
    step7.ensure_non_empty_split(
        validation_split, ROLE_VALIDATION, repeat_seed, horizon
    )
    step7.ensure_non_empty_split(outer_split, "oof", repeat_seed, horizon)

    mean, std = step7.fit_train_normalizer(train_split.X)
    X_train = step7.normalize(train_split.X, mean, std)
    X_validation = step7.normalize(validation_split.X, mean, std)
    X_outer = step7.normalize(outer_split.X, mean, std)
    model_name = str(cfg["selected_model"])
    checkpoint_path = chunk_dir / f"{model_name}.pt"
    history_path = chunk_dir / f"{model_name}_history.csv"

    if model_name == "persistence":
        best_epoch = 0
        best_validation_mae = step7.compute_metrics(
            validation_split.y_soh_target, validation_split.current_soh
        )["MAE"]
        validation_pred_soh = validation_split.current_soh.astype(np.float32)
        validation_pred_delta = np.zeros_like(validation_split.y_delta)
        outer_pred_soh = outer_split.current_soh.astype(np.float32)
        outer_pred_delta = np.zeros_like(outer_split.y_delta)
        _atomic_json(
            checkpoint_path,
            {
                "dataset": step7.DATASET,
                "stage": STAGE,
                "run_id": run_id,
                "model": "persistence",
                "repeat_seed": repeat_seed,
                "outer_fold": outer_fold,
                "horizon": horizon,
            },
        )
        _atomic_csv(history_path, pd.DataFrame(columns=["epoch"]))
    else:
        model = locked_eval.make_model(model_name, cfg)
        train_loader = locked_eval.scaled_delta_loader(
            train_split,
            X_train,
            cfg["target_scale"],
            cfg["batch_size"],
            shuffle=True,
        )
        validation_loader = locked_eval.scaled_delta_loader(
            validation_split,
            X_validation,
            cfg["target_scale"],
            cfg["batch_size"],
            shuffle=False,
        )
        print(
            f"[train OOF] repeat={repeat_seed} fold={outer_fold} "
            f"horizon={horizon} model={model_name} device={device}",
            flush=True,
        )
        model, history, best_epoch, best_validation_mae = step7.train_delta_model(
            model,
            train_loader,
            validation_loader,
            val_target_soh=validation_split.y_soh_target,
            val_current_soh=validation_split.current_soh,
            target_scale=cfg["target_scale"],
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            patience=cfg["patience"],
            min_delta=0.0,
            clip_grad_norm=cfg["clip_grad_norm"],
            device=device,
        )
        validation_pred_soh, validation_pred_delta = locked_eval.evaluate_model_on_split(
            model,
            validation_split,
            X_validation,
            cfg["target_scale"],
            cfg["batch_size"],
            device,
        )
        outer_pred_soh, outer_pred_delta = locked_eval.evaluate_model_on_split(
            model,
            outer_split,
            X_outer,
            cfg["target_scale"],
            cfg["batch_size"],
            device,
        )
        _atomic_csv(history_path, history)
        _save_torch_atomic(
            checkpoint_path,
            _checkpoint_payload(
                model=model,
                cfg=cfg,
                locked_config=locked_config,
                repeat_seed=repeat_seed,
                outer_fold=outer_fold,
                horizon=horizon,
                training_seed=training_seed,
                roles=roles,
                mean=mean,
                std=std,
                best_epoch=int(best_epoch),
                best_validation_mae=float(best_validation_mae),
                run_id=run_id,
            ),
        )
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    inner_predictions = _prediction_frame(
        validation_split,
        repeat_seed=repeat_seed,
        outer_fold=outer_fold,
        training_seed=training_seed,
        model_name=model_name,
        pred_soh=validation_pred_soh,
        pred_delta=validation_pred_delta,
        best_epoch=int(best_epoch),
        run_id=run_id,
    )
    outer_predictions = _prediction_frame(
        outer_split,
        repeat_seed=repeat_seed,
        outer_fold=outer_fold,
        training_seed=training_seed,
        model_name=model_name,
        pred_soh=outer_pred_soh,
        pred_delta=outer_pred_delta,
        best_epoch=int(best_epoch),
        run_id=run_id,
    )
    metrics = pd.DataFrame(
        [
            prediction_metrics_row(
                inner_predictions,
                best_epoch=int(best_epoch),
                checkpoint_path=str(checkpoint_path),
            ),
            prediction_metrics_row(
                outer_predictions,
                best_epoch=int(best_epoch),
                checkpoint_path=str(checkpoint_path),
            ),
        ]
    )
    _atomic_csv(chunk_dir / "inner_validation_predictions.csv", inner_predictions)
    _atomic_csv(chunk_dir / "oof_predictions.csv", outer_predictions)
    _atomic_csv(chunk_dir / "metrics.csv", metrics)
    _atomic_json(
        chunk_dir / "complete.json",
        {
            "run_id": run_id,
            "repeat_seed": int(repeat_seed),
            "outer_fold": int(outer_fold),
            "horizon": int(horizon),
            "model": model_name,
            "checkpoint_file": checkpoint_path.name,
            "history_file": history_path.name,
            "n_inner_validation_windows": int(len(inner_predictions)),
            "n_outer_oof_windows": int(len(outer_predictions)),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return None


def validate_oof_prediction_coverage(
    predictions: pd.DataFrame,
    *,
    eligible_ids_by_horizon: Mapping[int, Sequence[str]],
    repeat_seeds: Sequence[int],
    model_name: str,
) -> None:
    """Assert exact once-per-repeat battery coverage and window-key uniqueness."""

    _require_columns(
        predictions,
        [
            "repeat_seed",
            "outer_fold",
            "model",
            "battery_id",
            "horizon",
            "target_cycle",
        ],
        "OOF predictions",
    )
    models = set(predictions["model"].astype(str).unique())
    if models != {str(model_name)}:
        raise ValueError(f"Expected selected model only, observed models={models}")
    duplicate_keys = [
        "repeat_seed",
        "model",
        "battery_id",
        "horizon",
        "target_cycle",
    ]
    duplicated = predictions.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        examples = predictions.loc[duplicated, duplicate_keys].head(10)
        raise ValueError(
            "Duplicate OOF prediction keys: " + repr(examples.to_dict("records"))
        )

    expected_repeats = {int(value) for value in repeat_seeds}
    observed_repeats = set(predictions["repeat_seed"].astype(int).unique())
    if observed_repeats != expected_repeats:
        raise ValueError(
            f"Repeat coverage mismatch: expected={expected_repeats}, observed={observed_repeats}"
        )
    for repeat_seed in sorted(expected_repeats):
        one_repeat = predictions[
            predictions["repeat_seed"].astype(int) == int(repeat_seed)
        ]
        per_battery_fold_count = one_repeat.groupby("battery_id")["outer_fold"].nunique()
        if per_battery_fold_count.gt(1).any():
            bad = per_battery_fold_count[per_battery_fold_count.gt(1)].index.tolist()
            raise ValueError(
                f"Repeat {repeat_seed} has batteries in multiple outer folds: {bad[:10]}"
            )
        for horizon, expected_values in eligible_ids_by_horizon.items():
            expected = {str(value) for value in expected_values}
            observed = set(
                one_repeat.loc[
                    one_repeat["horizon"].astype(int) == int(horizon), "battery_id"
                ].astype(str)
            )
            if observed != expected:
                raise ValueError(
                    f"OOF battery coverage mismatch repeat={repeat_seed}, horizon={horizon}; "
                    f"missing={sorted(expected - observed)[:10]}, "
                    f"unexpected={sorted(observed - expected)[:10]}"
                )


def _repeat_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["repeat_seed", "split", "horizon", "model"]
    for _, group in predictions.groupby(group_columns, sort=True):
        # Fold is deliberately collapsed here; mark it as -1 so the public
        # metric helper can still validate a single identity value.
        combined = group.copy()
        combined["outer_fold"] = -1
        row = prediction_metrics_row(combined)
        row["outer_fold"] = "all_folds"
        row["best_epoch"] = float(group["best_epoch"].median())
        row["best_epoch_min"] = int(group["best_epoch"].min())
        row["best_epoch_max"] = int(group["best_epoch"].max())
        row["checkpoint_path"] = "multiple_fold_checkpoints"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["split", "horizon", "repeat_seed"]
    ).reset_index(drop=True)


def _metric_summary(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "MAE",
        "RMSE",
        "MAPE_percent",
        "R2",
        "Skill_MAE_vs_persistence",
        "macro_cell_MAE",
        "macro_cell_RMSE",
        "macro_cell_MAPE_percent",
        "macro_cell_persistence_MAE",
        "macro_cell_Skill_MAE_vs_persistence",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in repeat_metrics.groupby(
        ["split", "horizon", "model"], sort=True
    ):
        split_name, horizon, model = keys
        row: dict[str, Any] = {
            "dataset": step7.DATASET,
            "stage": STAGE,
            "split": split_name,
            "horizon": int(horizon),
            "model": model,
            "repeats_evaluated": int(group["repeat_seed"].nunique()),
            "n_batteries_mean": float(group["n_batteries"].mean()),
            "n_windows_mean": float(group["n_windows"].mean()),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "horizon"]).reset_index(drop=True)


def _write_readme(
    output_dir: Path,
    *,
    cfg: dict[str, Any],
    repeat_seeds: Sequence[int],
    n_splits: int,
    validation_fraction: float,
) -> None:
    text = f"""# MATR Repeated Battery-Level OOF Predictions

This directory contains cross-fitted predictions from the already selected
`{cfg['selected_model']}` architecture and locked hyperparameters. No model or
hyperparameter selection is performed here.

- Outer folds: {int(n_splits)}, stratified by MATR batch and blocked by within-batch observed-cycle-end rank
- Repeat seeds: {list(int(value) for value in repeat_seeds)}
- Inner early-stop validation fraction: {float(validation_fraction):.3f} of the non-outer pool
- Inner validation sampling: balanced across batch and observed-cycle-end rank
- Lookback: {cfg['lookback']}
- Sample mode: {cfg['sample_mode']}
- Horizons: {cfg['horizons']}
- Normalization: fit separately on each inner-training split only
- OOF rule: an outer battery is absent from training, normalization, and early stopping
- Calibration note: inner-validation predictions are reused after early stopping
  to set the residual score transform; they are not an independent conformal set

Important files:

- `oof_predictions.csv`: selected-model predictions for every eligible battery
- `inner_validation_predictions.csv`: fold-local early-stopping predictions
- `oof_fold_assignments.csv`: one outer fold per battery and repeat
- `oof_split_roles.csv`: train/inner-validation/outer role audit
- `oof_metrics_by_fold_horizon.csv`: window and macro-cell metrics by fold
- `oof_metrics_by_repeat_horizon.csv`: all-fold metrics by repeat
- `oof_metrics_summary.csv`: repeat-aggregated metrics
- `oof_fold_manifest_seed<seed>.json`: exact battery IDs and batch counts
- `fold_artifacts/`: checkpoint, history, predictions, metrics, and completion marker per job

These are out-of-fold predictions for a fixed previously selected model, not a
fully nested estimate of the complete model-selection procedure. A strict
nested-CV claim would require repeating architecture and hyperparameter
selection inside every outer fold.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if int(args.n_splits) < 3:
        raise ValueError("--n-splits must be at least 3")
    repeat_seeds = [int(value) for value in args.repeat_seeds]
    if not repeat_seeds or len(set(repeat_seeds)) != len(repeat_seeds):
        raise ValueError("--repeat-seeds must be non-empty and distinct")
    if not 0.0 < float(args.inner_validation_fraction) < 0.5:
        raise ValueError("--inner-validation-fraction must be in (0, 0.5)")

    config_path = Path(args.config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Locked config does not exist: {config_path}")
    locked_config = locked_eval.load_locked_config(config_path)
    cfg = _resolve_runtime_config(locked_config, args)

    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    data_root = Path(args.data_root).expanduser().resolve()
    all_matr_files, excluded_files = step7.find_matr_files(data_root)
    records, dataset_manifest = step7.load_dataset_manifest(
        all_matr_files,
        excluded_files,
        data_root=data_root,
        lookback=cfg["lookback"],
        horizons=cfg["horizons"],
        fixed_len=cfg["fixed_len"],
        sample_mode=cfg["sample_mode"],
        requested_batches=None,
        excluded_batch_files=None,
    )
    records_by_id = {record.battery_id: record for record in records}
    battery_table = battery_table_from_records(records)
    if battery_table["batch_id"].nunique() < 2:
        raise ValueError("Full MATR OOF analysis expected multiple protocol batches")
    eligible_ids_by_horizon = {
        int(horizon): [
            record.battery_id
            for record in records
            if step7.count_horizon_samples(
                record, int(horizon), cfg["lookback"], cfg["sample_mode"]
            )
            > 0
        ]
        for horizon in cfg["horizons"]
    }
    if args.expected_battery_count is not None:
        expected_battery_count = int(args.expected_battery_count)
        if expected_battery_count < 1:
            raise ValueError("--expected-battery-count must be positive")
        if len(battery_table) != expected_battery_count:
            raise RuntimeError(
                "Usable battery count mismatch: "
                f"expected={expected_battery_count}, observed={len(battery_table)}"
            )
        horizon_count_mismatch = {
            int(horizon): len(ids)
            for horizon, ids in eligible_ids_by_horizon.items()
            if len(ids) != expected_battery_count
        }
        if horizon_count_mismatch:
            raise RuntimeError(
                "Every requested horizon must cover the full expected cohort; "
                f"mismatch={horizon_count_mismatch}"
            )

    dataset_identity = battery_table[
        ["battery_id", "cell_id", "batch_id", "observed_cycle_end"]
    ].to_dict("records")
    dataset_file_identity = [
        {
            "battery_id": str(record.battery_id),
            "file": str(record.file_path.resolve()),
            "size_bytes": int(record.file_path.stat().st_size),
            "modified_time_ns": int(record.file_path.stat().st_mtime_ns),
        }
        for record in sorted(records, key=lambda item: item.battery_id)
    ]
    source_config_text = config_path.read_text(encoding="utf-8")
    run_signature_payload = {
        "stage": STAGE,
        "pipeline_schema_version": 2,
        "data_root": str(data_root),
        "dataset_identity": dataset_identity,
        "dataset_file_identity": dataset_file_identity,
        "source_config_sha256": _sha256_text(source_config_text),
        "resolved_runtime_config": cfg,
        "n_splits": int(args.n_splits),
        "repeat_seeds": repeat_seeds,
        "inner_validation_fraction": float(args.inner_validation_fraction),
        "expected_battery_count": (
            int(args.expected_battery_count)
            if args.expected_battery_count is not None
            else None
        ),
        "device": device,
        "debug": bool(args.debug),
    }
    run_id = _sha256_text(_stable_json(run_signature_payload))[:20]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "oof_run_config.json"
    existing_files = [path for path in output_dir.iterdir()]
    if existing_files and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory "
            "or pass --resume for the exact same run configuration."
        )
    if args.resume and existing_files:
        if not run_config_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume without {run_config_path}; use a new output directory"
            )
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_config.get("run_id") != run_id:
            raise ValueError(
                "Resume configuration does not match this output directory: "
                f"existing run_id={existing_config.get('run_id')}, requested={run_id}"
            )

    _atomic_json(
        run_config_path,
        {
            "run_id": run_id,
            "created_or_resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_mode": "repeated_battery_level_oof_training",
            "model_selection_performed": False,
            "hyperparameter_tuning_performed": False,
            "test_metrics_used_for_selection": False,
            "source_locked_config_path": str(config_path),
            "source_locked_config": locked_config,
            **run_signature_payload,
        },
    )
    _atomic_json(output_dir / "dataset_manifest.json", dataset_manifest)

    all_assignment_frames: list[pd.DataFrame] = []
    all_role_frames: list[pd.DataFrame] = []
    roles_by_repeat_fold: dict[tuple[int, int], dict[str, list[str]]] = {}
    for repeat_seed in repeat_seeds:
        assignments = assign_stratified_folds(
            battery_table,
            n_splits=int(args.n_splits),
            seed=int(repeat_seed),
        )
        all_assignment_frames.append(assignments)
        roles_by_fold: dict[int, dict[str, list[str]]] = {}
        for outer_fold in range(int(args.n_splits)):
            inner_seed = derive_seed(repeat_seed, outer_fold, 0, offset=17)
            roles = split_inner_train_validation(
                assignments,
                outer_fold=outer_fold,
                validation_fraction=float(args.inner_validation_fraction),
                seed=inner_seed,
            )
            roles_by_fold[outer_fold] = roles
            roles_by_repeat_fold[(repeat_seed, outer_fold)] = roles
        role_frame = _role_rows(assignments, roles_by_fold)
        all_role_frames.append(role_frame)
        manifest = _build_repeat_manifest(
            assignments,
            roles_by_fold,
            n_splits=int(args.n_splits),
            validation_fraction=float(args.inner_validation_fraction),
        )
        _atomic_json(
            output_dir / f"oof_fold_manifest_seed{repeat_seed}.json", manifest
        )

    assignments_all = pd.concat(all_assignment_frames, ignore_index=True).sort_values(
        ["repeat_seed", "outer_fold", "batch_id", "battery_id"]
    )
    roles_all = pd.concat(all_role_frames, ignore_index=True)
    _atomic_csv(output_dir / "oof_fold_assignments.csv", assignments_all)
    _atomic_csv(output_dir / "oof_split_roles.csv", roles_all)

    completed_chunk_dirs: list[Path] = []
    artifact_root = output_dir / "fold_artifacts"

    for repeat_seed in repeat_seeds:
        for outer_fold in range(int(args.n_splits)):
            roles = roles_by_repeat_fold[(repeat_seed, outer_fold)]
            for horizon in cfg["horizons"]:
                chunk_dir = (
                    artifact_root
                    / f"seed{repeat_seed}"
                    / f"fold{outer_fold}"
                    / f"horizon{horizon}"
                )
                resumed = False
                if args.resume:
                    resumed = _completed_chunk_is_valid(
                        chunk_dir,
                        run_id=run_id,
                        repeat_seed=repeat_seed,
                        outer_fold=outer_fold,
                        horizon=int(horizon),
                    )
                if resumed:
                    print(
                        f"[resume] repeat={repeat_seed} fold={outer_fold} "
                        f"horizon={horizon}",
                        flush=True,
                    )
                else:
                    chunk_dir.mkdir(parents=True, exist_ok=True)
                    _run_training_chunk(
                        records_by_id=records_by_id,
                        roles=roles,
                        cfg=cfg,
                        locked_config=locked_config,
                        repeat_seed=repeat_seed,
                        outer_fold=outer_fold,
                        horizon=int(horizon),
                        device=device,
                        run_id=run_id,
                        chunk_dir=chunk_dir,
                    )
                completed_chunk_dirs.append(chunk_dir)

    # BatteryRecord objects retain all resampled cycles. Release them before
    # assembling the potentially million-row repeated prediction tables.
    records_by_id.clear()
    records.clear()
    del records_by_id, records
    gc.collect()

    outer_predictions = pd.concat(
        (
            pd.read_csv(chunk_dir / "oof_predictions.csv")
            for chunk_dir in completed_chunk_dirs
        ),
        ignore_index=True,
    ).sort_values(
        ["repeat_seed", "outer_fold", "horizon", "battery_id", "target_cycle"]
    )
    validate_oof_prediction_coverage(
        outer_predictions,
        eligible_ids_by_horizon=eligible_ids_by_horizon,
        repeat_seeds=repeat_seeds,
        model_name=cfg["selected_model"],
    )
    _atomic_csv(output_dir / "oof_predictions.csv", outer_predictions)
    outer_repeat_metrics = _repeat_metrics(outer_predictions)
    oof_unique_batteries = int(outer_predictions["battery_id"].nunique())
    oof_prediction_rows = int(len(outer_predictions))
    del outer_predictions
    gc.collect()

    inner_predictions = pd.concat(
        (
            pd.read_csv(chunk_dir / "inner_validation_predictions.csv")
            for chunk_dir in completed_chunk_dirs
        ),
        ignore_index=True,
    ).sort_values(
        ["repeat_seed", "outer_fold", "horizon", "battery_id", "target_cycle"]
    )
    _atomic_csv(
        output_dir / "inner_validation_predictions.csv", inner_predictions
    )
    inner_repeat_metrics = _repeat_metrics(inner_predictions)
    inner_validation_prediction_rows = int(len(inner_predictions))
    del inner_predictions
    gc.collect()

    fold_metrics = pd.concat(
        (
            pd.read_csv(chunk_dir / "metrics.csv")
            for chunk_dir in completed_chunk_dirs
        ),
        ignore_index=True,
    ).sort_values(["split", "horizon", "repeat_seed", "outer_fold"])
    _atomic_csv(output_dir / "oof_metrics_by_fold_horizon.csv", fold_metrics)
    repeat_metrics = pd.concat(
        [outer_repeat_metrics, inner_repeat_metrics],
        ignore_index=True,
    )
    repeat_metrics = repeat_metrics.sort_values(
        ["split", "horizon", "repeat_seed"]
    ).reset_index(drop=True)
    metric_summary = _metric_summary(repeat_metrics)
    _atomic_csv(
        output_dir / "oof_metrics_by_repeat_horizon.csv", repeat_metrics
    )
    _atomic_csv(output_dir / "oof_metrics_summary.csv", metric_summary)
    _write_readme(
        output_dir,
        cfg=cfg,
        repeat_seeds=repeat_seeds,
        n_splits=int(args.n_splits),
        validation_fraction=float(args.inner_validation_fraction),
    )
    _atomic_json(
        output_dir / "oof_completion.json",
        {
            "run_id": run_id,
            "stage": STAGE,
            "status": "complete",
            "selected_model": cfg["selected_model"],
            "repeat_seeds": repeat_seeds,
            "n_splits": int(args.n_splits),
            "horizons": cfg["horizons"],
            "eligible_batteries_by_horizon": {
                str(horizon): len(ids)
                for horizon, ids in eligible_ids_by_horizon.items()
            },
            "oof_unique_batteries": oof_unique_batteries,
            "oof_prediction_rows": oof_prediction_rows,
            "inner_validation_prediction_rows": inner_validation_prediction_rows,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print("\n=== MATR repeated battery-level OOF complete ===", flush=True)
    print(metric_summary.to_string(index=False), flush=True)
    print(f"\noutput_dir={output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate repeated batch-stratified battery-level OOF predictions "
            "with within-batch observed-length blocking, using the fixed, "
            "previously selected MATR model."
        )
    )
    parser.add_argument("--data-root", default="MATR")
    parser.add_argument(
        "--config-path",
        required=True,
        help="Locked final config or best_optuna_config.json (sidecar is loaded when present).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument(
        "--repeat-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_REPEAT_SEEDS),
        help="Fold-allocation repeat seeds; each repeat evaluates every eligible battery once.",
    )
    parser.add_argument(
        "--inner-validation-fraction",
        type=float,
        default=0.20,
        help="Batch-stratified early-stop fraction of the non-outer battery pool.",
    )
    parser.add_argument(
        "--expected-battery-count",
        type=int,
        default=None,
        help=(
            "Fail before training unless the usable cohort and every requested "
            "horizon contain exactly this many batteries."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse fold chunks completed under the exact same run signature.",
    )
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument(
        "--sample-mode", choices=step7.SAMPLE_MODES, default=None
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--fixed-len", type=int, default=None)
    parser.add_argument("--target-scale", type=float, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Small CPU/GPU smoke configuration; not for scientific results.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
