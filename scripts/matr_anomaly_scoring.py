from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnomalyConfig:
    """Configuration for cell-level empirical conformal anomaly scoring."""

    target_model: str = "cpmlp_cpdsconv_fusion"
    expected_horizons: tuple[int, ...] = (10, 50, 100)
    alpha_selection_horizons: tuple[int, int] = (50, 100)
    alpha_grid: tuple[float, ...] = tuple(float(x) for x in np.linspace(0.0, 1.0, 21))
    validation_selection_fraction: float = 1.0 / 3.0
    alpha_cell_aggregation: str = "mean"
    min_common_windows: int = 5
    min_paired_cells: int = 5
    bootstrap_repeats: int = 500
    conformal_candidate_p: float = 0.10
    conformal_strong_p: float = 0.05
    random_state: int = 20260711
    threshold_method: str = "cell_empirical_conformal"


CALIBRATION_KEYS = ["seed", "horizon"]
COMPONENT_COLUMNS = ["degradation_residual", "degradation_slope_score"]
STANDARDIZED_COMPONENT_COLUMNS = ["degradation_residual_z", "degradation_slope_z"]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{context} is missing columns: {missing}")


def _finite_numeric(frame: pd.DataFrame, columns: Sequence[str], context: str) -> pd.DataFrame:
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    if not finite.all():
        bad_counts = (~finite).sum(axis=0)
        raise ValueError(
            f"{context} contains non-finite values: "
            f"{dict(zip(values.columns, bad_counts.tolist()))}"
        )
    return values


def prepare_residual_features(
    predictions: pd.DataFrame,
    split_name: str,
    *,
    target_model: str,
    expected_horizons: Sequence[int],
) -> pd.DataFrame:
    """Build one-sided degradation residual and trajectory-slope components."""

    _require_columns(
        predictions,
        [
            "model",
            "seed",
            "battery_id",
            "horizon",
            "input_end_cycle",
            "target_cycle",
            "actual_soh",
            "current_soh",
            "pred_soh",
        ],
        f"{split_name} predictions",
    )

    frame = predictions[predictions["model"] == target_model].copy()
    if frame.empty:
        raise RuntimeError(f"No {split_name} predictions found for model={target_model!r}")

    if "cell_id" not in frame.columns:
        frame["cell_id"] = frame["battery_id"].astype(str)

    if "sample_mode" in frame.columns:
        sample_modes = sorted(frame["sample_mode"].dropna().astype(str).unique())
        if sample_modes != ["sliding-window"]:
            raise RuntimeError(
                f"Expected sliding-window {split_name} predictions, got {sample_modes}"
            )

    expected = sorted(int(h) for h in expected_horizons)
    observed = sorted(frame["horizon"].dropna().astype(int).unique())
    if observed != expected:
        raise RuntimeError(
            f"Expected {split_name} horizons {expected}, got {observed}"
        )

    key_cols = ["seed", "model", "battery_id", "horizon", "target_cycle"]
    duplicated = frame.duplicated(key_cols, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, key_cols].head(10).to_dict("records")
        raise ValueError(f"Duplicate prediction keys in {split_name}: {examples}")

    frame = frame.sort_values(key_cols).copy()
    group_cols = ["seed", "model", "battery_id", "horizon"]

    # Horizon delta must mean current SOH minus the future target SOH.
    # Do not replace it with an adjacent-row diff when output columns are absent.
    if "actual_delta_soh" not in frame.columns:
        frame["actual_delta_soh"] = frame["current_soh"] - frame["actual_soh"]
    if "pred_delta_soh" not in frame.columns:
        frame["pred_delta_soh"] = frame["current_soh"] - frame["pred_soh"]

    numeric_cols = [
        "actual_delta_soh",
        "pred_delta_soh",
        "actual_soh",
        "pred_soh",
        "target_cycle",
        "input_end_cycle",
    ]
    frame[numeric_cols] = _finite_numeric(frame, numeric_cols, split_name)

    frame["residual_score"] = (
        frame["actual_delta_soh"] - frame["pred_delta_soh"]
    ).abs()
    frame["degradation_residual"] = (
        frame["actual_delta_soh"] - frame["pred_delta_soh"]
    ).clip(lower=0.0)

    cycle_gap = frame.groupby(group_cols)["target_cycle"].diff()
    actual_soh_change = frame.groupby(group_cols)["actual_soh"].diff()
    pred_soh_change = frame.groupby(group_cols)["pred_soh"].diff()
    valid_gap = cycle_gap.gt(0) & cycle_gap.notna()

    frame["actual_degradation_slope"] = (
        (-actual_soh_change).div(cycle_gap).where(valid_gap, 0.0).clip(lower=0.0)
    )
    frame["pred_degradation_slope"] = (
        (-pred_soh_change).div(cycle_gap).where(valid_gap, 0.0).clip(lower=0.0)
    )
    frame["degradation_slope_score"] = (
        frame["actual_degradation_slope"] - frame["pred_degradation_slope"]
    ).clip(lower=0.0)
    frame["score_split"] = str(split_name)

    _finite_numeric(frame, COMPONENT_COLUMNS, f"{split_name} anomaly components")
    return frame


def split_validation_roles(
    validation_df: pd.DataFrame,
    *,
    selection_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split validation batteries into alpha-development and conformal-reference roles."""

    if not 0.0 < float(selection_fraction) < 1.0:
        raise ValueError("selection_fraction must be between 0 and 1")
    _require_columns(
        validation_df,
        ["score_split", "seed", "battery_id", "cell_id", "batch_id"],
        "validation residual frame",
    )
    split_values = set(validation_df["score_split"].dropna().astype(str).unique())
    if split_values != {"validation"}:
        raise ValueError(f"Expected validation rows only, got score_split={split_values}")

    cell_table = validation_df[
        ["seed", "battery_id", "cell_id", "batch_id"]
    ].drop_duplicates()
    per_battery = cell_table.groupby(["seed", "battery_id"], as_index=False).size()
    if per_battery["size"].ne(1).any():
        raise ValueError("Each seed/battery must map to exactly one cell and batch")

    assignments: list[dict[str, object]] = []
    for seed, one_seed in cell_table.groupby("seed", sort=True):
        seed_value = int(seed)
        rng = np.random.default_rng(int(random_state) + seed_value)
        role_counts = {"alpha_selection": 0, "conformal_calibration": 0}

        for batch_id, one_batch in one_seed.groupby("batch_id", sort=True):
            battery_ids = sorted(one_batch["battery_id"].astype(str).unique())
            battery_ids = list(np.asarray(battery_ids)[rng.permutation(len(battery_ids))])

            if len(battery_ids) == 1:
                selection_count = int(
                    role_counts["alpha_selection"]
                    <= role_counts["conformal_calibration"]
                )
            else:
                selection_count = int(round(len(battery_ids) * selection_fraction))
                selection_count = min(max(selection_count, 1), len(battery_ids) - 1)

            selection_ids = set(battery_ids[:selection_count])
            for battery_id in battery_ids:
                role = (
                    "alpha_selection"
                    if battery_id in selection_ids
                    else "conformal_calibration"
                )
                cell_id = one_batch.loc[
                    one_batch["battery_id"].astype(str) == battery_id, "cell_id"
                ].iloc[0]
                assignments.append(
                    {
                        "seed": seed,
                        "battery_id": battery_id,
                        "cell_id": cell_id,
                        "batch_id": batch_id,
                        "validation_role": role,
                    }
                )
                role_counts[role] += 1

        if min(role_counts.values()) == 0:
            raise RuntimeError(
                f"Seed {seed} could not be split into both validation roles: {role_counts}"
            )

    assignment_df = pd.DataFrame(assignments)
    result = validation_df.merge(
        assignment_df[["seed", "battery_id", "validation_role"]],
        on=["seed", "battery_id"],
        how="left",
        validate="many_to_one",
    )
    if result["validation_role"].isna().any():
        raise RuntimeError("Some validation rows were not assigned a role")
    return result, assignment_df


def assert_seed_split_isolation(
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Ensure each seed's validation batteries are absent from that seed's test split."""

    _require_columns(validation_df, ["seed", "battery_id"], "validation frame")
    _require_columns(test_df, ["seed", "battery_id"], "test frame")
    common_seeds = sorted(set(validation_df["seed"]) & set(test_df["seed"]))
    for seed in common_seeds:
        validation_ids = set(
            validation_df.loc[validation_df["seed"] == seed, "battery_id"].astype(str)
        )
        test_ids = set(test_df.loc[test_df["seed"] == seed, "battery_id"].astype(str))
        overlap = sorted(validation_ids & test_ids)
        if overlap:
            raise RuntimeError(
                f"Seed {seed} has validation/test battery overlap: {overlap[:10]}"
            )


def fit_component_calibration(calibration_df: pd.DataFrame) -> pd.DataFrame:
    """Fit component scales on alpha-development batteries only."""

    _require_columns(
        calibration_df,
        CALIBRATION_KEYS
        + ["battery_id", "target_cycle"]
        + COMPONENT_COLUMNS,
        "component calibration frame",
    )
    if "validation_role" in calibration_df.columns:
        roles = set(calibration_df["validation_role"].dropna().astype(str).unique())
        if roles != {"alpha_selection"}:
            raise ValueError(
                "Component statistics must use alpha_selection rows only; "
                f"got roles={roles}"
            )

    working = calibration_df.copy()
    working["_residual_square"] = working["degradation_residual"] ** 2
    working["_slope_square"] = working["degradation_slope_score"] ** 2
    cell_moments = (
        working.groupby(CALIBRATION_KEYS + ["battery_id"], as_index=False)
        .agg(
            residual_cell_mean=("degradation_residual", "mean"),
            residual_cell_second_moment=("_residual_square", "mean"),
            slope_cell_mean=("degradation_slope_score", "mean"),
            slope_cell_second_moment=("_slope_square", "mean"),
            cell_points=("target_cycle", "count"),
        )
    )
    stats = (
        cell_moments.groupby(CALIBRATION_KEYS, as_index=False)
        .agg(
            residual_mean=("residual_cell_mean", "mean"),
            residual_second_moment=("residual_cell_second_moment", "mean"),
            slope_mean=("slope_cell_mean", "mean"),
            slope_second_moment=("slope_cell_second_moment", "mean"),
            calibration_points=("cell_points", "sum"),
            calibration_batteries=("battery_id", "nunique"),
        )
    )
    stats["residual_std"] = np.sqrt(
        np.maximum(stats["residual_second_moment"] - stats["residual_mean"] ** 2, 0.0)
    )
    stats["slope_std"] = np.sqrt(
        np.maximum(stats["slope_second_moment"] - stats["slope_mean"] ** 2, 0.0)
    )
    stats["component_weighting"] = "equal_battery_window_moments"
    stats = stats.drop(
        columns=["residual_second_moment", "slope_second_moment"]
    )
    numeric = _finite_numeric(
        stats,
        ["residual_mean", "residual_std", "slope_mean", "slope_std"],
        "component calibration statistics",
    )
    stats[["residual_mean", "residual_std", "slope_mean", "slope_std"]] = numeric

    degenerate = stats[(stats["residual_std"] <= 0) | (stats["slope_std"] <= 0)]
    if not degenerate.empty:
        raise RuntimeError(
            "Degenerate validation component standard deviation: "
            + degenerate[CALIBRATION_KEYS + ["residual_std", "slope_std"]]
            .to_dict("records")
            .__repr__()
        )
    return stats


def apply_component_calibration(
    frame: pd.DataFrame,
    calibration_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Apply calibration statistics and create non-negative component z-scores."""

    calibrated = frame.merge(
        calibration_stats,
        on=CALIBRATION_KEYS,
        how="left",
        validate="many_to_one",
    )
    stat_cols = ["residual_mean", "residual_std", "slope_mean", "slope_std"]
    if calibrated[stat_cols].isna().any().any():
        missing = calibrated.loc[
            calibrated[stat_cols].isna().any(axis=1), CALIBRATION_KEYS
        ].drop_duplicates()
        raise RuntimeError(
            f"Missing component calibration for groups: {missing.to_dict('records')}"
        )

    calibrated["degradation_residual_z"] = (
        (calibrated["degradation_residual"] - calibrated["residual_mean"])
        / calibrated["residual_std"]
    ).clip(lower=0.0)
    calibrated["degradation_slope_z"] = (
        (calibrated["degradation_slope_score"] - calibrated["slope_mean"])
        / calibrated["slope_std"]
    ).clip(lower=0.0)
    _finite_numeric(
        calibrated,
        STANDARDIZED_COMPONENT_COLUMNS,
        "standardized anomaly components",
    )
    return calibrated


def _spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average")
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average")
    if x_rank.nunique() < 2 or y_rank.nunique() < 2:
        return float("nan")
    return float(x_rank.corr(y_rank))


def _paired_cell_severity(
    seed_frame: pd.DataFrame,
    *,
    alpha: float,
    horizons: tuple[int, int],
    severity_aggregation: str,
    min_common_windows: int,
) -> pd.DataFrame:
    scored = seed_frame.copy()
    scored["_candidate_score"] = (
        float(alpha) * scored["degradation_residual_z"]
        + (1.0 - float(alpha)) * scored["degradation_slope_z"]
    )
    row_keys = ["battery_id", "cell_id", "input_end_cycle", "horizon"]
    duplicated = scored.duplicated(row_keys, keep=False)
    if duplicated.any():
        raise ValueError(
            "Duplicate validation windows during alpha selection: "
            + scored.loc[duplicated, row_keys].head(10).to_dict("records").__repr__()
        )

    common = scored.pivot(
        index=["battery_id", "cell_id", "input_end_cycle"],
        columns="horizon",
        values="_candidate_score",
    ).reindex(columns=list(horizons))
    common = common.dropna()

    window_counts = common.groupby(level=["battery_id", "cell_id"]).size()
    eligible_cells = window_counts[window_counts >= int(min_common_windows)].index
    common = common.loc[common.index.droplevel("input_end_cycle").isin(eligible_cells)]
    if common.empty:
        return pd.DataFrame(columns=list(horizons), dtype=float)

    if str(severity_aggregation).lower() != "mean":
        raise ValueError("Only severity_aggregation='mean' is supported")
    severity = common.groupby(level=["battery_id", "cell_id"]).mean()
    return severity.dropna()


def select_alpha_by_seed(
    selection_df: pd.DataFrame,
    *,
    alpha_grid: Sequence[float],
    horizons: tuple[int, int],
    severity_aggregation: str,
    min_common_windows: int,
    min_paired_cells: int,
    bootstrap_repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one alpha per seed using H50/H100 cell-rank stability only."""

    _require_columns(
        selection_df,
        [
            "score_split",
            "seed",
            "battery_id",
            "cell_id",
            "horizon",
            "input_end_cycle",
        ]
        + STANDARDIZED_COMPONENT_COLUMNS,
        "alpha-selection frame",
    )
    if "validation_role" in selection_df.columns:
        roles = set(selection_df["validation_role"].dropna().astype(str).unique())
        if roles != {"alpha_selection"}:
            raise ValueError(f"Alpha selection received validation roles={roles}")
    split_values = set(selection_df["score_split"].dropna().astype(str).unique())
    if split_values != {"validation"}:
        raise ValueError(
            f"Alpha selection accepts validation rows only, got {split_values}"
        )

    alphas = np.unique(np.round(np.asarray(list(alpha_grid), dtype=float), 12))
    if len(alphas) == 0 or not np.isfinite(alphas).all():
        raise ValueError("alpha_grid must contain finite values")
    if (alphas < 0).any() or (alphas > 1).any():
        raise ValueError("alpha_grid values must be in [0, 1]")
    if len(horizons) != 2 or horizons[0] == horizons[1]:
        raise ValueError("Exactly two different horizons are required")
    if bootstrap_repeats < 100:
        raise ValueError("bootstrap_repeats must be at least 100")

    search_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for seed, seed_frame in selection_df.groupby("seed", sort=True):
        seed_frame = seed_frame[seed_frame["horizon"].isin(horizons)].copy()
        observed_horizons = set(seed_frame["horizon"].astype(int).unique())
        if observed_horizons != set(horizons):
            raise RuntimeError(
                f"Seed {seed} alpha selection has horizons={sorted(observed_horizons)}"
            )

        severity_by_alpha: dict[float, pd.DataFrame] = {}
        common_index: pd.Index | None = None
        for alpha in alphas:
            severity = _paired_cell_severity(
                seed_frame,
                alpha=float(alpha),
                horizons=horizons,
                severity_aggregation=severity_aggregation,
                min_common_windows=min_common_windows,
            )
            if len(severity) < int(min_paired_cells):
                raise RuntimeError(
                    f"Seed {seed} has only {len(severity)} paired validation cells "
                    f"for alpha={alpha:.2f}"
                )
            severity_by_alpha[float(alpha)] = severity
            common_index = severity.index if common_index is None else common_index.intersection(severity.index)

        if common_index is None or len(common_index) < int(min_paired_cells):
            raise RuntimeError(f"Seed {seed} has insufficient common cells across alpha grid")

        rng = np.random.default_rng(int(random_state) + int(seed) * 1009)
        n_cells = len(common_index)
        bootstrap_indices = rng.integers(
            0,
            n_cells,
            size=(int(bootstrap_repeats), n_cells),
        )

        seed_search_rows: list[dict[str, object]] = []
        for alpha in alphas:
            severity = severity_by_alpha[float(alpha)].reindex(common_index)
            x = severity[horizons[0]].to_numpy(dtype=float)
            y = severity[horizons[1]].to_numpy(dtype=float)
            observed_rho = _spearman_rank_corr(x, y)
            bootstrap_rho = np.asarray(
                [
                    _spearman_rank_corr(x[index], y[index])
                    for index in bootstrap_indices
                ],
                dtype=float,
            )
            finite_bootstrap = bootstrap_rho[np.isfinite(bootstrap_rho)]
            if len(finite_bootstrap) < max(50, int(bootstrap_repeats * 0.8)):
                seed_search_rows.append(
                    {
                        "seed": seed,
                        "alpha": float(alpha),
                        "beta": float(1.0 - alpha),
                        "rank_corr": observed_rho,
                        "bootstrap_std": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_paired_cells": int(n_cells),
                        "n_bootstrap_valid": int(len(finite_bootstrap)),
                        "search_status": "degenerate_bootstrap",
                    }
                )
                continue

            seed_search_rows.append(
                {
                    "seed": seed,
                    "alpha": float(alpha),
                    "beta": float(1.0 - alpha),
                    "rank_corr": observed_rho,
                    "bootstrap_std": float(np.std(finite_bootstrap, ddof=1)),
                    "ci_low": float(np.quantile(finite_bootstrap, 0.025)),
                    "ci_high": float(np.quantile(finite_bootstrap, 0.975)),
                    "n_paired_cells": int(n_cells),
                    "n_bootstrap_valid": int(len(finite_bootstrap)),
                    "search_status": "ok",
                }
            )

        seed_search = pd.DataFrame(seed_search_rows)
        valid = seed_search.dropna(subset=["rank_corr", "bootstrap_std"]).copy()
        if valid.empty:
            raise RuntimeError(f"Seed {seed} produced no finite alpha stability scores")

        best_position = valid["rank_corr"].idxmax()
        best_row = valid.loc[best_position]
        best_alpha = float(best_row["alpha"])
        best_se = float(best_row["bootstrap_std"])
        cutoff = float(best_row["rank_corr"] - best_se)
        seed_search["in_one_se_plateau"] = (
            seed_search["search_status"].eq("ok")
            & seed_search["rank_corr"].ge(cutoff)
        )

        plateau = seed_search[seed_search["in_one_se_plateau"]].copy()
        plateau["distance_from_balanced"] = (plateau["alpha"] - 0.5).abs()
        selected = (
            plateau.sort_values(
                ["distance_from_balanced", "rank_corr", "alpha"],
                ascending=[True, False, True],
            )
            .iloc[0]
        )
        selected_alpha = float(selected["alpha"])
        seed_search["one_se_cutoff"] = cutoff

        plateau_low = float(plateau["alpha"].min())
        plateau_high = float(plateau["alpha"].max())
        plateau_width = plateau_high - plateau_low
        status = (
            "weakly_identified_broad_plateau"
            if plateau_width >= 0.20
            else "stability_selected_one_se"
        )
        if float(best_row["rank_corr"]) <= 0:
            balanced = valid.assign(
                distance_from_balanced=(valid["alpha"] - 0.5).abs()
            ).sort_values(
                ["distance_from_balanced", "rank_corr", "alpha"],
                ascending=[True, False, True],
            )
            selected = balanced.iloc[0]
            selected_alpha = float(selected["alpha"])
            status = "nonpositive_stability_balanced_fallback"

        seed_search["selected"] = np.isclose(
            seed_search["alpha"], selected_alpha, atol=1e-12
        )
        search_rows.extend(seed_search.to_dict("records"))

        selected_rows.append(
            {
                "seed": seed,
                "alpha": selected_alpha,
                "beta": float(1.0 - selected_alpha),
                "selected_rank_corr": float(selected["rank_corr"]),
                "empirical_best_alpha": best_alpha,
                "empirical_best_rank_corr": float(best_row["rank_corr"]),
                "one_se_cutoff": cutoff,
                "plateau_alpha_low": plateau_low,
                "plateau_alpha_high": plateau_high,
                "n_paired_cells": int(n_cells),
                "selection_status": status,
            }
        )

    alpha_by_seed = pd.DataFrame(selected_rows).sort_values("seed").reset_index(drop=True)
    search_table = pd.DataFrame(search_rows).sort_values(["seed", "alpha"]).reset_index(drop=True)
    return alpha_by_seed, search_table


def add_scores_by_seed(
    frame: pd.DataFrame,
    alpha_by_seed: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the validation-selected alpha for each matching seed."""

    _require_columns(frame, ["seed"] + STANDARDIZED_COMPONENT_COLUMNS, "score frame")
    _require_columns(alpha_by_seed, ["seed", "alpha", "beta"], "alpha table")
    if alpha_by_seed["seed"].duplicated().any():
        raise ValueError("alpha_by_seed must contain exactly one row per seed")

    result = frame.drop(
        columns=["anomaly_alpha", "anomaly_beta"], errors="ignore"
    ).merge(
        alpha_by_seed[["seed", "alpha", "beta"]].rename(
            columns={"alpha": "anomaly_alpha", "beta": "anomaly_beta"}
        ),
        on="seed",
        how="left",
        validate="many_to_one",
    )
    if result[["anomaly_alpha", "anomaly_beta"]].isna().any().any():
        missing_seeds = sorted(result.loc[result["anomaly_alpha"].isna(), "seed"].unique())
        raise RuntimeError(f"Missing selected alpha for seeds={missing_seeds}")

    result["degradation_anomaly_score"] = (
        result["anomaly_alpha"] * result["degradation_residual_z"]
        + result["anomaly_beta"] * result["degradation_slope_z"]
    )
    return result


def summarize_common_horizon_scores(
    scored_frame: pd.DataFrame,
    *,
    expected_horizons: Sequence[int],
    min_common_windows: int,
) -> pd.DataFrame:
    """Create one continuous mean severity per cell and horizon.

    Only ``input_end_cycle`` values available for every requested horizon are
    retained. Sliding windows remain the model input, but the calibration unit
    is a physical battery rather than an individual, highly overlapping window.
    """

    horizons = tuple(sorted(int(h) for h in expected_horizons))
    if len(horizons) < 2 or len(set(horizons)) != len(horizons):
        raise ValueError("expected_horizons must contain distinct horizon values")
    if int(min_common_windows) < 1:
        raise ValueError("min_common_windows must be positive")

    required = [
        "seed",
        "battery_id",
        "cell_id",
        "horizon",
        "input_end_cycle",
        "target_cycle",
        "degradation_anomaly_score",
        "degradation_residual",
        "degradation_slope_score",
        "residual_score",
    ]
    _require_columns(scored_frame, required, "continuous anomaly-score frame")
    observed = tuple(sorted(scored_frame["horizon"].dropna().astype(int).unique()))
    if observed != horizons:
        raise ValueError(f"Expected scored horizons={horizons}, observed={observed}")

    cell_keys = ["seed", "battery_id", "cell_id"]
    duplicate_keys = cell_keys + ["input_end_cycle", "horizon"]
    duplicated = scored_frame.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        examples = scored_frame.loc[duplicated, duplicate_keys].head(10)
        raise ValueError(
            "Duplicate cell/horizon windows in continuous scoring: "
            + examples.to_dict("records").__repr__()
        )

    all_cells = scored_frame[cell_keys].drop_duplicates()
    horizon_counts = (
        scored_frame.groupby(cell_keys, as_index=False)["horizon"]
        .nunique()
        .rename(columns={"horizon": "n_observed_horizons"})
    )
    incomplete = horizon_counts[horizon_counts["n_observed_horizons"] != len(horizons)]
    if not incomplete.empty:
        raise RuntimeError(
            "Every scored cell must contain all requested horizons: "
            + incomplete.head(10).to_dict("records").__repr__()
        )

    wide = scored_frame.pivot(
        index=cell_keys + ["input_end_cycle"],
        columns="horizon",
        values="degradation_anomaly_score",
    ).reindex(columns=list(horizons))
    common = wide.dropna()
    common_counts = common.groupby(level=cell_keys).size().rename("n_common_windows")
    insufficient = common_counts[common_counts < int(min_common_windows)]
    missing_common = all_cells.merge(
        common_counts.reset_index(),
        on=cell_keys,
        how="left",
        validate="one_to_one",
    )
    missing_common["n_common_windows"] = missing_common["n_common_windows"].fillna(0)
    missing_common = missing_common[
        missing_common["n_common_windows"] < int(min_common_windows)
    ]
    if not missing_common.empty or not insufficient.empty:
        raise RuntimeError(
            f"Cells need at least {int(min_common_windows)} common windows: "
            + missing_common.head(10).to_dict("records").__repr__()
        )

    common_keys = common.reset_index()[cell_keys + ["input_end_cycle"]]
    common_rows = scored_frame.merge(
        common_keys,
        on=cell_keys + ["input_end_cycle"],
        how="inner",
        validate="many_to_one",
    )

    metadata_columns = [column for column in ["batch_id", "validation_role"] if column in common_rows]
    for column in metadata_columns:
        per_cell_unique = common_rows.groupby(cell_keys)[column].nunique(dropna=False)
        if per_cell_unique.gt(1).any():
            raise ValueError(f"Each scored cell must map to one {column}")

    horizon_summary = (
        common_rows.groupby(cell_keys + ["horizon"], as_index=False)
        .agg(
            horizon_mean_score=("degradation_anomaly_score", "mean"),
            horizon_p95_score=(
                "degradation_anomaly_score",
                lambda values: values.quantile(0.95),
            ),
            horizon_max_score=("degradation_anomaly_score", "max"),
            mean_degradation_residual=("degradation_residual", "mean"),
            mean_degradation_slope_score=("degradation_slope_score", "mean"),
            mean_absolute_residual=("residual_score", "mean"),
            n_common_windows=("input_end_cycle", "count"),
            common_input_cycle_start=("input_end_cycle", "min"),
            common_input_cycle_end=("input_end_cycle", "max"),
        )
    )

    metadata = common_rows[cell_keys + metadata_columns].drop_duplicates(cell_keys)
    horizon_summary = horizon_summary.merge(
        metadata,
        on=cell_keys,
        how="left",
        validate="many_to_one",
    )
    horizon_summary["horizon_aggregation"] = "mean_over_common_sliding_windows"
    _finite_numeric(
        horizon_summary,
        ["horizon_mean_score", "horizon_p95_score", "horizon_max_score"],
        "horizon continuous severity",
    )
    return horizon_summary.sort_values(cell_keys + ["horizon"]).reset_index(drop=True)


def fit_horizon_severity_calibration(
    alpha_horizon_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Fit robust horizon-level location/scale on alpha-development cells."""

    _require_columns(
        alpha_horizon_summary,
        ["seed", "horizon", "battery_id", "horizon_mean_score"],
        "alpha horizon summary",
    )
    if "validation_role" in alpha_horizon_summary.columns:
        roles = set(
            alpha_horizon_summary["validation_role"].dropna().astype(str).unique()
        )
        if roles != {"alpha_selection"}:
            raise ValueError(
                "Horizon severity calibration must use alpha_selection cells; "
                f"got roles={roles}"
            )

    rows: list[dict[str, object]] = []
    for (seed, horizon), group in alpha_horizon_summary.groupby(
        ["seed", "horizon"], sort=True
    ):
        values = pd.to_numeric(group["horizon_mean_score"], errors="coerce").to_numpy(
            dtype=float
        )
        if len(values) < 3 or not np.isfinite(values).all():
            raise ValueError(
                f"Invalid alpha-development horizon scores for seed={seed}, horizon={horizon}"
            )
        center = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        robust_scale = float((q75 - q25) / 1.349)
        standard_scale = float(np.std(values, ddof=0))
        if np.isfinite(robust_scale) and robust_scale > 1e-12:
            scale = robust_scale
            scale_method = "median_iqr"
        elif np.isfinite(standard_scale) and standard_scale > 1e-12:
            scale = standard_scale
            scale_method = "median_std_fallback"
        else:
            raise RuntimeError(
                f"Degenerate horizon severity scale for seed={seed}, horizon={horizon}"
            )
        rows.append(
            {
                "seed": seed,
                "horizon": int(horizon),
                "horizon_score_center": center,
                "horizon_score_scale": scale,
                "horizon_score_q25": float(q25),
                "horizon_score_q75": float(q75),
                "horizon_score_std": standard_scale,
                "horizon_scale_method": scale_method,
                "horizon_calibration_cells": int(group["battery_id"].nunique()),
                "horizon_calibration_source": "alpha_selection_only",
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "horizon"]).reset_index(drop=True)


def apply_horizon_severity_calibration(
    horizon_summary: pd.DataFrame,
    horizon_calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Put per-horizon cell means on a common alpha-development scale."""

    _require_columns(
        horizon_summary,
        ["seed", "horizon", "horizon_mean_score"],
        "horizon summary",
    )
    _require_columns(
        horizon_calibration,
        ["seed", "horizon", "horizon_score_center", "horizon_score_scale"],
        "horizon severity calibration",
    )
    result = horizon_summary.merge(
        horizon_calibration,
        on=["seed", "horizon"],
        how="left",
        validate="many_to_one",
    )
    if result[["horizon_score_center", "horizon_score_scale"]].isna().any().any():
        missing = result.loc[
            result["horizon_score_center"].isna(), ["seed", "horizon"]
        ].drop_duplicates()
        raise RuntimeError(
            "Missing horizon severity calibration: "
            + missing.to_dict("records").__repr__()
        )
    result["horizon_relative_severity"] = (
        result["horizon_mean_score"] - result["horizon_score_center"]
    ) / result["horizon_score_scale"]
    _finite_numeric(
        result,
        ["horizon_relative_severity"],
        "relative horizon severity",
    )
    return result


def aggregate_cell_nonconformity(
    calibrated_horizon_summary: pd.DataFrame,
    *,
    expected_horizons: Sequence[int],
) -> pd.DataFrame:
    """Aggregate horizon severities into one continuous score per cell.

    With H10/H50/H100, the median is the second-largest horizon severity. It
    continuously downweights a single-horizon spike without applying a binary
    horizon threshold or a 2-of-3 vote.
    """

    horizons = tuple(sorted(int(h) for h in expected_horizons))
    cell_keys = ["seed", "battery_id", "cell_id"]
    _require_columns(
        calibrated_horizon_summary,
        cell_keys
        + [
            "horizon",
            "horizon_mean_score",
            "horizon_relative_severity",
            "n_common_windows",
            "common_input_cycle_start",
            "common_input_cycle_end",
            "horizon_max_score",
        ],
        "calibrated horizon summary",
    )
    counts = calibrated_horizon_summary.groupby(cell_keys)["horizon"].nunique()
    incomplete = counts[counts != len(horizons)]
    if not incomplete.empty:
        raise RuntimeError(
            "Cell nonconformity requires complete horizon coverage: "
            + incomplete.head(10).to_dict().__repr__()
        )

    relative = calibrated_horizon_summary.pivot(
        index=cell_keys,
        columns="horizon",
        values="horizon_relative_severity",
    ).reindex(columns=list(horizons))
    raw_mean = calibrated_horizon_summary.pivot(
        index=cell_keys,
        columns="horizon",
        values="horizon_mean_score",
    ).reindex(columns=list(horizons))
    windows = calibrated_horizon_summary.pivot(
        index=cell_keys,
        columns="horizon",
        values="n_common_windows",
    ).reindex(columns=list(horizons))
    starts = calibrated_horizon_summary.pivot(
        index=cell_keys,
        columns="horizon",
        values="common_input_cycle_start",
    ).reindex(columns=list(horizons))
    ends = calibrated_horizon_summary.pivot(
        index=cell_keys,
        columns="horizon",
        values="common_input_cycle_end",
    ).reindex(columns=list(horizons))
    if (
        relative.isna().any().any()
        or raw_mean.isna().any().any()
        or windows.isna().any().any()
        or starts.isna().any().any()
        or ends.isna().any().any()
    ):
        raise RuntimeError("Missing horizon values while building cell nonconformity")
    if (
        windows.nunique(axis=1).gt(1).any()
        or starts.nunique(axis=1).gt(1).any()
        or ends.nunique(axis=1).gt(1).any()
    ):
        raise RuntimeError(
            "Common-window coverage must be identical across horizons within each cell"
        )

    result = pd.DataFrame(index=relative.index).reset_index()
    result["n_common_windows"] = windows.iloc[:, 0].to_numpy(dtype=int)
    result["common_input_cycle_start"] = starts.iloc[:, 0].to_numpy(dtype=float)
    result["common_input_cycle_end"] = ends.iloc[:, 0].to_numpy(dtype=float)
    result["common_input_cycle_span"] = (
        result["common_input_cycle_end"] - result["common_input_cycle_start"]
    )
    for horizon in horizons:
        result[f"h{horizon}_mean_score"] = raw_mean[horizon].to_numpy(dtype=float)
        result[f"h{horizon}_relative_severity"] = relative[horizon].to_numpy(
            dtype=float
        )
        result[f"n_common_windows_h{horizon}"] = windows[horizon].to_numpy(dtype=int)

    result["cell_nonconformity_score"] = np.median(
        relative.to_numpy(dtype=float), axis=1
    )
    result["n_total_horizons"] = len(horizons)
    result["has_full_horizon_coverage"] = True
    result["cell_horizon_aggregation"] = "median_of_relative_horizon_means"

    metadata_columns = [
        column
        for column in ["batch_id", "validation_role"]
        if column in calibrated_horizon_summary.columns
    ]
    for column in metadata_columns:
        unique_counts = calibrated_horizon_summary.groupby(cell_keys)[column].nunique(
            dropna=False
        )
        if unique_counts.gt(1).any():
            raise ValueError(
                f"Each cell must map to one {column} during aggregation"
            )
    metadata = calibrated_horizon_summary[cell_keys + metadata_columns].drop_duplicates(
        cell_keys
    )
    result = result.merge(metadata, on=cell_keys, how="left", validate="one_to_one")

    maxima = (
        calibrated_horizon_summary.groupby(cell_keys, as_index=False)[
            "horizon_max_score"
        ]
        .max()
        .rename(columns={"horizon_max_score": "max_window_anomaly_score"})
    )
    result = result.merge(maxima, on=cell_keys, how="left", validate="one_to_one")
    _finite_numeric(
        result,
        ["cell_nonconformity_score", "max_window_anomaly_score"],
        "cell nonconformity",
    )
    return result.sort_values(
        ["seed", "cell_nonconformity_score"], ascending=[True, False]
    ).reset_index(drop=True)


def audit_cell_score_coverage(
    cell_summary: pd.DataFrame,
    *,
    data_role: str,
    warning_abs_rank_corr: float = 0.50,
) -> pd.DataFrame:
    """Audit whether cell rarity is strongly associated with data coverage.

    Different cells can contribute very different life spans even though the
    same H10/H50/H100 ``input_end_cycle`` values are used within each cell.
    These diagnostics do not alter or validate a p-value; they reveal when the
    score may partly rank observation length/life stage instead of degradation.
    """

    required = [
        "seed",
        "battery_id",
        "cell_id",
        "cell_nonconformity_score",
        "n_common_windows",
        "common_input_cycle_start",
        "common_input_cycle_end",
        "common_input_cycle_span",
    ]
    _require_columns(cell_summary, required, "cell coverage audit")
    if not str(data_role).strip():
        raise ValueError("data_role must be a non-empty label")
    if not 0.0 < float(warning_abs_rank_corr) <= 1.0:
        raise ValueError("warning_abs_rank_corr must be in (0, 1]")
    duplicate_keys = ["seed", "battery_id", "cell_id"]
    duplicated = cell_summary.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        raise ValueError("Coverage audit requires one row per seed/physical cell")

    metrics = [
        "n_common_windows",
        "common_input_cycle_end",
        "common_input_cycle_span",
    ]
    rows: list[dict[str, object]] = []
    for seed, group in cell_summary.groupby("seed", sort=True):
        score = pd.to_numeric(
            group["cell_nonconformity_score"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(score).all():
            raise ValueError(f"Non-finite cell scores in coverage audit for seed={seed}")

        row: dict[str, object] = {
            "data_role": str(data_role),
            "seed": seed,
            "n_cells": int(len(group)),
            "common_windows_min": int(group["n_common_windows"].min()),
            "common_windows_median": float(group["n_common_windows"].median()),
            "common_windows_max": int(group["n_common_windows"].max()),
            "common_cycle_end_min": float(group["common_input_cycle_end"].min()),
            "common_cycle_end_median": float(
                group["common_input_cycle_end"].median()
            ),
            "common_cycle_end_max": float(group["common_input_cycle_end"].max()),
            "warning_abs_rank_corr_cutoff": float(warning_abs_rank_corr),
            "coverage_audit_note": (
                "diagnostic_only_large_correlation_can_confound_empirical_rarity"
            ),
        }
        correlations: list[float] = []
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                dtype=float
            )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Non-finite {metric} in coverage audit for seed={seed}"
                )
            correlation = _spearman_rank_corr(score, values)
            column = f"rank_corr_score_vs_{metric}"
            row[column] = correlation
            if np.isfinite(correlation):
                correlations.append(abs(float(correlation)))

        max_abs = max(correlations) if correlations else float("nan")
        row["max_abs_coverage_rank_corr"] = max_abs
        row["has_large_coverage_association"] = bool(
            np.isfinite(max_abs) and max_abs >= float(warning_abs_rank_corr)
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["data_role", "seed"]).reset_index(
        drop=True
    )


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if len(values) == 0:
        return values.copy()
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def apply_empirical_conformal_pvalues(
    test_cell_summary: pd.DataFrame,
    calibration_cell_summary: pd.DataFrame,
    *,
    candidate_p: float,
    strong_p: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare test cell scores with untouched validation-reference cells.

    The returned values are conservative rank p-values with a ``+1`` correction
    and ``>=`` tie handling. Because the underlying SOH predictor was previously
    selected using the broader validation split, they are deliberately labelled
    empirical conformal-style p-values rather than strict finite-sample claims.
    """

    if not 0.0 < float(strong_p) <= float(candidate_p) < 1.0:
        raise ValueError("Require 0 < strong_p <= candidate_p < 1")

    required = ["seed", "battery_id", "cell_id", "cell_nonconformity_score"]
    _require_columns(test_cell_summary, required, "test cell summary")
    _require_columns(calibration_cell_summary, required, "calibration cell summary")
    unique_keys = ["seed", "battery_id", "cell_id"]
    for name, frame in [
        ("test cell summary", test_cell_summary),
        ("calibration cell summary", calibration_cell_summary),
    ]:
        duplicated = frame.duplicated(unique_keys, keep=False)
        if duplicated.any():
            raise ValueError(
                f"{name} must contain one row per seed/physical cell: "
                + frame.loc[duplicated, unique_keys]
                .head(10)
                .to_dict("records")
                .__repr__()
            )
    if "validation_role" in calibration_cell_summary.columns:
        roles = set(
            calibration_cell_summary["validation_role"].dropna().astype(str).unique()
        )
        if roles != {"conformal_calibration"}:
            raise ValueError(
                "Empirical conformal reference must use conformal_calibration cells; "
                f"got roles={roles}"
            )
    assert_seed_split_isolation(calibration_cell_summary, test_cell_summary)

    test_seeds = set(test_cell_summary["seed"].unique())
    calibration_seeds = set(calibration_cell_summary["seed"].unique())
    if test_seeds != calibration_seeds:
        raise ValueError(
            f"Test/calibration seed mismatch: test={test_seeds}, calibration={calibration_seeds}"
        )

    result_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for seed, test_group in test_cell_summary.groupby("seed", sort=True):
        calibration_group = calibration_cell_summary[
            calibration_cell_summary["seed"] == seed
        ]
        calibration_scores = pd.to_numeric(
            calibration_group["cell_nonconformity_score"], errors="coerce"
        ).to_numpy(dtype=float)
        test_scores = pd.to_numeric(
            test_group["cell_nonconformity_score"], errors="coerce"
        ).to_numpy(dtype=float)
        if len(calibration_scores) < 2 or not np.isfinite(calibration_scores).all():
            raise ValueError(f"Invalid conformal calibration scores for seed={seed}")
        if not np.isfinite(test_scores).all():
            raise ValueError(f"Invalid test cell scores for seed={seed}")

        n_calibration = len(calibration_scores)
        tail_counts = (calibration_scores[:, None] >= test_scores[None, :]).sum(axis=0)
        p_values = (1.0 + tail_counts) / (n_calibration + 1.0)
        tail_rank = 1 + tail_counts

        # Deterministic leave-one-calibration-cell-out sensitivity audit. Using
        # n-1 references preserves the p<=.05 resolution available at n=22/23;
        # an 80% subsample would make a strong flag mathematically impossible.
        leave_one_out_scores = np.stack(
            [np.delete(calibration_scores, index) for index in range(n_calibration)]
        )
        leave_one_out_p = (
            1.0
            + (
                leave_one_out_scores[:, :, None] >= test_scores[None, None, :]
            ).sum(axis=1)
        ) / n_calibration

        one = test_group.copy().reset_index(drop=True)
        one["empirical_conformal_p_value"] = p_values
        one["bh_adjusted_empirical_tail_p"] = _benjamini_hochberg(p_values)
        one["calibration_tail_count"] = tail_counts.astype(int)
        one["calibration_tail_rank"] = tail_rank.astype(int)
        one["conformal_calibration_cells"] = int(n_calibration)
        one["minimum_attainable_p_value"] = float(1.0 / (n_calibration + 1.0))
        one["empirical_rarity_percentile"] = 100.0 * (1.0 - p_values)
        one["leave_one_out_median_p_value"] = np.median(
            leave_one_out_p, axis=0
        )
        one["leave_one_out_min_p_value"] = np.min(leave_one_out_p, axis=0)
        one["leave_one_out_max_p_value"] = np.max(leave_one_out_p, axis=0)
        one["leave_one_out_minimum_attainable_p_value"] = float(
            1.0 / n_calibration
        )
        one["leave_one_out_candidate_resolution_available"] = bool(
            (1.0 / n_calibration) <= float(candidate_p)
        )
        one["leave_one_out_strong_resolution_available"] = bool(
            (1.0 / n_calibration) <= float(strong_p)
        )
        one["leave_one_out_candidate_frequency"] = np.mean(
            leave_one_out_p <= float(candidate_p), axis=0
        )
        one["leave_one_out_strong_frequency"] = np.mean(
            leave_one_out_p <= float(strong_p), axis=0
        )
        one["is_conformal_candidate"] = p_values <= float(candidate_p)
        one["is_strong_candidate"] = p_values <= float(strong_p)
        one["is_exploratory_bh_flag"] = (
            one["bh_adjusted_empirical_tail_p"] <= float(candidate_p)
        )
        one["cell_status"] = "no_strong_tail_evidence"
        one.loc[one["is_conformal_candidate"], "cell_status"] = (
            "empirical_tail_candidate"
        )
        one.loc[one["is_strong_candidate"], "cell_status"] = (
            "strong_empirical_tail_candidate"
        )
        one["candidate_p_cutoff"] = float(candidate_p)
        one["strong_p_cutoff"] = float(strong_p)
        one["p_value_method"] = "cell_level_empirical_conformal_rank"
        one["calibration_scope"] = "seed_level_batch_pooled_marginal"
        one["stability_method"] = "leave_one_calibration_cell_out"
        one["bh_validity_note"] = "exploratory_only_no_formal_fdr_claim"
        result_parts.append(one)

        attainable_candidate = (
            np.floor(float(candidate_p) * (n_calibration + 1))
            / (n_calibration + 1)
        )
        attainable_strong = (
            np.floor(float(strong_p) * (n_calibration + 1))
            / (n_calibration + 1)
        )
        audit_rows.append(
            {
                "seed": seed,
                "calibration_cells": int(n_calibration),
                "calibration_batches": int(
                    calibration_group["batch_id"].nunique()
                    if "batch_id" in calibration_group
                    else 0
                ),
                "calibration_score_min": float(np.min(calibration_scores)),
                "calibration_score_median": float(np.median(calibration_scores)),
                "calibration_score_max": float(np.max(calibration_scores)),
                "minimum_attainable_p_value": float(
                    1.0 / (n_calibration + 1.0)
                ),
                "leave_one_out_minimum_attainable_p_value": float(
                    1.0 / n_calibration
                ),
                "leave_one_out_candidate_resolution_available": bool(
                    (1.0 / n_calibration) <= float(candidate_p)
                ),
                "leave_one_out_strong_resolution_available": bool(
                    (1.0 / n_calibration) <= float(strong_p)
                ),
                "requested_candidate_p": float(candidate_p),
                "largest_attainable_p_at_candidate_cutoff": float(
                    attainable_candidate
                ),
                "requested_strong_p": float(strong_p),
                "largest_attainable_p_at_strong_cutoff": float(attainable_strong),
                "candidate_resolution_available": bool(attainable_candidate > 0),
                "strong_resolution_available": bool(attainable_strong > 0),
                "calibration_scope": "seed_level_batch_pooled_marginal",
                "formal_validity_note": (
                    "exploratory_empirical_p_due_to_prior_validation_model_selection"
                ),
            }
        )

    result = pd.concat(result_parts, ignore_index=True)
    result = result.sort_values(
        ["is_strong_candidate", "is_conformal_candidate", "empirical_conformal_p_value", "cell_nonconformity_score"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    return result, pd.DataFrame(audit_rows).sort_values("seed").reset_index(drop=True)


def aggregate_physical_cell_evidence(
    scored_cell_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Merge repeated seed-test results into one conservative physical-cell row."""

    required = [
        "seed",
        "battery_id",
        "cell_id",
        "cell_nonconformity_score",
        "empirical_conformal_p_value",
        "leave_one_out_candidate_frequency",
        "is_conformal_candidate",
        "is_strong_candidate",
    ]
    _require_columns(scored_cell_summary, required, "seed-level conformal cells")
    unique_keys = ["seed", "battery_id", "cell_id"]
    duplicated = scored_cell_summary.duplicated(unique_keys, keep=False)
    if duplicated.any():
        raise ValueError(
            "Seed-level conformal cells must contain one row per seed/physical cell: "
            + scored_cell_summary.loc[duplicated, unique_keys]
            .head(10)
            .to_dict("records")
            .__repr__()
        )
    if "batch_id" in scored_cell_summary.columns:
        batch_counts = scored_cell_summary.groupby(
            ["battery_id", "cell_id"]
        )["batch_id"].nunique(dropna=False)
        if batch_counts.gt(1).any():
            raise ValueError("Each physical cell must map to exactly one batch_id")
    group_keys = ["battery_id", "cell_id"]
    if "batch_id" in scored_cell_summary.columns:
        group_keys.append("batch_id")

    severity_columns = [
        column
        for column in scored_cell_summary.columns
        if column.startswith("h") and column.endswith("_relative_severity")
    ]
    rows: list[dict[str, object]] = []
    for keys, group in scored_cell_summary.groupby(group_keys, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_keys, key_values))
        n_seeds = int(group["seed"].nunique())
        candidate_count = int(group["is_conformal_candidate"].sum())
        strong_count = int(group["is_strong_candidate"].sum())
        row.update(
            {
                "test_seeds": ",".join(
                    str(int(value)) for value in sorted(group["seed"].unique())
                ),
                "n_test_seeds": n_seeds,
                "median_cell_nonconformity_score": float(
                    group["cell_nonconformity_score"].median()
                ),
                "median_empirical_conformal_p_value": float(
                    group["empirical_conformal_p_value"].median()
                ),
                "worst_seed_empirical_p_value": float(
                    group["empirical_conformal_p_value"].max()
                ),
                "best_seed_empirical_p_value": float(
                    group["empirical_conformal_p_value"].min()
                ),
                "median_candidate_selection_frequency": float(
                    group["leave_one_out_candidate_frequency"].median()
                ),
                "candidate_seed_count": candidate_count,
                "strong_seed_count": strong_count,
                "is_physical_candidate": bool(candidate_count == n_seeds),
                "is_physical_strong_candidate": bool(strong_count == n_seeds),
                "seed_confirmation_rule": (
                    "all_available_out_of_sample_test_seeds_correlated_views"
                ),
            }
        )
        for column in severity_columns:
            row[f"median_{column}"] = float(group[column].median())
        rows.append(row)

    result = pd.DataFrame(rows)
    result["is_single_seed_candidate"] = (
        (result["n_test_seeds"] == 1) & result["is_physical_candidate"]
    )
    result["is_single_seed_strong_candidate"] = (
        (result["n_test_seeds"] == 1) & result["is_physical_strong_candidate"]
    )
    result["is_repeated_seed_candidate"] = (
        (result["n_test_seeds"] >= 2) & result["is_physical_candidate"]
    )
    result["is_repeated_seed_strong_candidate"] = (
        (result["n_test_seeds"] >= 2) & result["is_physical_strong_candidate"]
    )
    result["physical_cell_status"] = "no_strong_tail_evidence"
    result.loc[result["is_single_seed_candidate"], "physical_cell_status"] = (
        "single_seed_empirical_tail_candidate"
    )
    result.loc[
        result["is_single_seed_strong_candidate"], "physical_cell_status"
    ] = "single_seed_strong_empirical_tail_candidate"
    result.loc[result["is_repeated_seed_candidate"], "physical_cell_status"] = (
        "repeated_seed_empirical_tail_candidate"
    )
    result.loc[
        result["is_repeated_seed_strong_candidate"], "physical_cell_status"
    ] = "repeated_seed_strong_empirical_tail_candidate"
    result["seed_evidence_status"] = np.where(
        result["n_test_seeds"] > 1,
        "repeated_seed_evidence",
        "single_test_seed_only",
    )
    return result.sort_values(
        [
            "is_repeated_seed_strong_candidate",
            "is_repeated_seed_candidate",
            "is_single_seed_strong_candidate",
            "is_single_seed_candidate",
            "worst_seed_empirical_p_value",
            "median_cell_nonconformity_score",
        ],
        ascending=[False, False, False, False, True, False],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Legacy Top-5% / anomalous-window-ratio helpers
# ---------------------------------------------------------------------------
# These functions are retained only so older notebooks can still import them.
# The current notebook does not call this path; its final cell decision uses
# ``apply_empirical_conformal_pvalues`` and ``aggregate_physical_cell_evidence``.


def fit_score_thresholds(
    calibration_scored: pd.DataFrame,
    *,
    quantile: float,
    threshold_method: str = "top5",
) -> pd.DataFrame:
    """Fit validation Top-5% thresholds per seed/horizon.

    ``method='higher'`` and a strict ``score > threshold`` comparison keep
    tied scores at the cutoff from inflating the realized calibration alert rate.
    """

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if str(threshold_method).lower() != "top5":
        raise ValueError("Only threshold_method='top5' is supported")
    if not np.isclose(float(quantile), 0.95, atol=1e-12):
        raise ValueError(
            f"Top 5% threshold requires quantile=0.95, got {quantile}"
        )
    _require_columns(
        calibration_scored,
        CALIBRATION_KEYS + ["degradation_anomaly_score"],
        "score calibration frame",
    )
    if "validation_role" in calibration_scored.columns:
        roles = set(
            calibration_scored["validation_role"].dropna().astype(str).unique()
        )
        if roles != {"threshold_calibration"}:
            raise ValueError(f"Threshold calibration received roles={roles}")

    threshold_rows: list[dict[str, object]] = []
    for keys, one_group in calibration_scored.groupby(CALIBRATION_KEYS, sort=True):
        scores = pd.to_numeric(
            one_group["degradation_anomaly_score"], errors="coerce"
        ).to_numpy(dtype=float)
        if len(scores) == 0 or not np.isfinite(scores).all():
            raise ValueError(f"Invalid calibration scores for group={keys}")

        threshold = float(
            np.quantile(scores, float(quantile), method="higher")
        )
        n_above = int(np.sum(scores > threshold))
        row = {
            key: value for key, value in zip(CALIBRATION_KEYS, keys)
        }
        row.update(
            {
                "threshold_top5": threshold,
                "threshold_method": "top5_validation_quantile",
                "threshold_quantile": float(quantile),
                "threshold_quantile_method": "higher",
                "threshold_comparison": ">",
                "target_tail_fraction": float(1.0 - quantile),
                "calibration_points_for_threshold": int(len(scores)),
                "calibration_above_threshold": n_above,
                "calibration_above_rate": float(n_above / len(scores)),
            }
        )
        threshold_rows.append(row)

    thresholds = pd.DataFrame(threshold_rows)
    threshold_values = _finite_numeric(
        thresholds, ["threshold_top5"], "score thresholds"
    )["threshold_top5"]
    if threshold_values.le(0).any():
        bad = thresholds.loc[threshold_values.le(0), CALIBRATION_KEYS + ["threshold_top5"]]
        raise RuntimeError(
            "Non-positive anomaly threshold; calibration is degenerate: "
            + bad.to_dict("records").__repr__()
        )
    return thresholds


def apply_score_thresholds(
    scored_frame: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """Apply calibrated thresholds with a strict comparison to avoid zero-score ties."""

    result = scored_frame.drop(
        columns=[
            "threshold_top5",
            "threshold_method",
            "threshold_quantile",
            "threshold_quantile_method",
            "threshold_comparison",
            "target_tail_fraction",
            "calibration_points_for_threshold",
            "calibration_above_threshold",
            "calibration_above_rate",
            "threshold_source",
        ],
        errors="ignore",
    ).merge(
        thresholds,
        on=CALIBRATION_KEYS,
        how="left",
        validate="many_to_one",
    )
    if result["threshold_top5"].isna().any():
        missing = result.loc[
            result["threshold_top5"].isna(), CALIBRATION_KEYS
        ].drop_duplicates()
        raise RuntimeError(f"Missing thresholds for groups: {missing.to_dict('records')}")

    methods = set(result["threshold_method"].dropna().astype(str).unique())
    quantiles = set(result["threshold_quantile"].dropna().astype(float).unique())
    comparisons = set(result["threshold_comparison"].dropna().astype(str).unique())
    if methods != {"top5_validation_quantile"} or comparisons != {">"}:
        raise RuntimeError(
            f"Unexpected threshold metadata: methods={methods}, comparisons={comparisons}"
        )
    if len(quantiles) != 1 or not np.isclose(next(iter(quantiles)), 0.95):
        raise RuntimeError(f"Expected a single q95 threshold, got {quantiles}")

    result["threshold_source"] = "validation_top5_threshold_calibration"
    result["is_cycle_anomaly_top5"] = (
        result["degradation_anomaly_score"] > result["threshold_top5"]
    )
    return result


def summarize_horizon_and_cell_anomalies(
    scored_test: pd.DataFrame,
    *,
    expected_horizons: Sequence[int],
    cell_ratio_threshold: float,
    min_windows_per_horizon: int,
    min_anomalous_horizons: int,
    early_warning_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate windows by horizon first, then apply an equal-horizon cell consensus."""

    if not 0.0 < float(cell_ratio_threshold) <= 1.0:
        raise ValueError("cell_ratio_threshold must be in (0, 1]")
    if int(min_windows_per_horizon) < 1:
        raise ValueError("min_windows_per_horizon must be positive")
    horizons = tuple(sorted(int(h) for h in expected_horizons))
    if len(horizons) < 2 or len(set(horizons)) != len(horizons):
        raise ValueError("expected_horizons must contain distinct horizon values")
    if int(early_warning_horizon) not in horizons:
        raise ValueError("early_warning_horizon must be in expected_horizons")
    if not 1 <= int(min_anomalous_horizons) <= len(horizons):
        raise ValueError("min_anomalous_horizons is outside the horizon count")

    required = [
        "seed",
        "battery_id",
        "cell_id",
        "horizon",
        "input_end_cycle",
        "target_cycle",
        "is_cycle_anomaly_top5",
        "degradation_anomaly_score",
        "degradation_residual",
        "residual_score",
    ]
    _require_columns(scored_test, required, "scored test frame")
    observed = tuple(sorted(scored_test["horizon"].dropna().astype(int).unique()))
    if observed != horizons:
        raise ValueError(f"Expected scored horizons={horizons}, observed={observed}")

    cell_keys = ["seed", "battery_id", "cell_id"]
    horizon_keys = cell_keys + ["horizon"]
    horizon_summary = (
        scored_test.groupby(horizon_keys, as_index=False)
        .agg(
            n_points=("target_cycle", "count"),
            n_anomalies=("is_cycle_anomaly_top5", "sum"),
            mean_anomaly_score=("degradation_anomaly_score", "mean"),
            p95_anomaly_score=(
                "degradation_anomaly_score",
                lambda values: values.quantile(0.95),
            ),
            max_anomaly_score=("degradation_anomaly_score", "max"),
            mean_degradation_residual=("degradation_residual", "mean"),
            max_degradation_residual=("degradation_residual", "max"),
            mean_residual_score=("residual_score", "mean"),
            max_residual_score=("residual_score", "max"),
        )
    )
    horizon_summary["anomaly_ratio"] = (
        horizon_summary["n_anomalies"] / horizon_summary["n_points"]
    )
    horizon_summary["n_windows"] = horizon_summary["n_points"]
    horizon_summary["n_anomalous_windows"] = horizon_summary["n_anomalies"]
    horizon_summary["has_sufficient_windows"] = (
        horizon_summary["n_windows"] >= int(min_windows_per_horizon)
    )
    horizon_summary["is_horizon_anomaly"] = (
        horizon_summary["has_sufficient_windows"]
        & (horizon_summary["anomaly_ratio"] >= float(cell_ratio_threshold))
    )

    first_events = (
        scored_test[scored_test["is_cycle_anomaly_top5"]]
        .groupby(horizon_keys, as_index=False)
        .agg(
            first_anomaly_input_end_cycle=("input_end_cycle", "min"),
            first_anomaly_target_cycle=("target_cycle", "min"),
        )
    )
    horizon_summary = horizon_summary.merge(
        first_events,
        on=horizon_keys,
        how="left",
        validate="one_to_one",
    )

    # Equal-weight horizon aggregation prevents H10 (which has more windows)
    # from dominating the cell score merely because it contributes more rows.
    cell_summary = (
        horizon_summary.groupby(cell_keys, as_index=False)
        .agg(
            n_points=("n_points", "sum"),
            n_anomalies=("n_anomalies", "sum"),
            mean_horizon_anomaly_ratio=("anomaly_ratio", "mean"),
            max_horizon_anomaly_ratio=("anomaly_ratio", "max"),
            mean_anomaly_score=("mean_anomaly_score", "mean"),
            max_anomaly_score=("max_anomaly_score", "max"),
            mean_degradation_residual=("mean_degradation_residual", "mean"),
            max_degradation_residual=("max_degradation_residual", "max"),
            mean_residual_score=("mean_residual_score", "mean"),
            max_residual_score=("max_residual_score", "max"),
            n_anomaly_horizons=("is_horizon_anomaly", "sum"),
            n_total_horizons=("horizon", "nunique"),
            n_evaluated_horizons=("has_sufficient_windows", "sum"),
        )
    )
    cell_summary["pooled_window_anomaly_ratio"] = (
        cell_summary["n_anomalies"] / cell_summary["n_points"]
    )
    # Backward-compatible name, now explicitly equal-weighted by horizon.
    cell_summary["anomaly_ratio"] = cell_summary["mean_horizon_anomaly_ratio"]

    ratio_wide = horizon_summary.pivot(
        index=cell_keys,
        columns="horizon",
        values="anomaly_ratio",
    ).rename(columns={h: f"h{h}_anomaly_ratio" for h in horizons})
    flag_wide = horizon_summary.pivot(
        index=cell_keys,
        columns="horizon",
        values="is_horizon_anomaly",
    ).rename(columns={h: f"h{h}_is_horizon_anomaly" for h in horizons})
    window_wide = horizon_summary.pivot(
        index=cell_keys,
        columns="horizon",
        values="n_windows",
    ).rename(columns={h: f"n_windows_h{h}" for h in horizons})
    anomaly_count_wide = horizon_summary.pivot(
        index=cell_keys,
        columns="horizon",
        values="n_anomalous_windows",
    ).rename(columns={h: f"n_anomalies_h{h}" for h in horizons})
    cell_summary = (
        cell_summary.set_index(cell_keys)
        .join(ratio_wide, validate="one_to_one")
        .join(flag_wide, validate="one_to_one")
        .join(window_wide, validate="one_to_one")
        .join(anomaly_count_wide, validate="one_to_one")
        .reset_index()
    )

    warning_col = f"h{int(early_warning_horizon)}_is_horizon_anomaly"
    long_horizon_cols = [
        f"h{h}_is_horizon_anomaly"
        for h in horizons
        if h != int(early_warning_horizon)
    ]
    flag_cols = [f"h{h}_is_horizon_anomaly" for h in horizons]
    cell_summary[flag_cols] = cell_summary[flag_cols].fillna(False).astype(bool)
    cell_summary["has_full_horizon_coverage"] = (
        (cell_summary["n_total_horizons"] == len(horizons))
        & (cell_summary["n_evaluated_horizons"] == len(horizons))
    )
    cell_summary["is_h10_warning"] = cell_summary[warning_col]
    cell_summary["is_long_horizon_anomaly"] = cell_summary[long_horizon_cols].any(axis=1)
    cell_summary["is_cell_anomaly"] = (
        cell_summary["has_full_horizon_coverage"]
        & (cell_summary["n_anomaly_horizons"] >= int(min_anomalous_horizons))
    )
    cell_summary["confirmed_h10_and_long_horizon"] = (
        cell_summary["is_h10_warning"]
        & cell_summary["is_long_horizon_anomaly"]
    )
    cell_summary["is_h10_only_warning"] = (
        cell_summary["has_full_horizon_coverage"]
        & cell_summary["is_h10_warning"]
        & (cell_summary["n_anomaly_horizons"] == 1)
    )

    ratio_cols = [f"h{h}_anomaly_ratio" for h in horizons]
    ratio_values = cell_summary[ratio_cols].to_numpy(dtype=float)
    full_rows = cell_summary["has_full_horizon_coverage"].to_numpy(dtype=bool)
    consensus_ratio = np.full(len(cell_summary), np.nan, dtype=float)
    consensus_ratio[full_rows] = np.sort(ratio_values[full_rows], axis=1)[:, -2]
    cell_summary["consensus_ratio_score"] = consensus_ratio

    single_watch_status = {
        int(early_warning_horizon): "h10_only_early_warning",
        **{
            h: f"single_h{h}_watch"
            for h in horizons
            if h != int(early_warning_horizon)
        },
    }
    cell_summary["cell_status"] = "no_horizon_anomaly"
    cell_summary.loc[
        ~cell_summary["has_full_horizon_coverage"], "cell_status"
    ] = "insufficient_horizon_coverage"
    for horizon, status in single_watch_status.items():
        only_this_horizon = (
            cell_summary["has_full_horizon_coverage"]
            & cell_summary[f"h{horizon}_is_horizon_anomaly"]
            & (cell_summary["n_anomaly_horizons"] == 1)
        )
        cell_summary.loc[only_this_horizon, "cell_status"] = status
    cell_summary.loc[cell_summary["is_cell_anomaly"], "cell_status"] = "consensus_anomaly"

    anomalous_horizons = (
        horizon_summary[horizon_summary["is_horizon_anomaly"]]
        .groupby(cell_keys)["horizon"]
        .agg(lambda values: ",".join(str(int(v)) for v in sorted(values)))
        .rename("anomalous_horizons")
        .reset_index()
    )
    cell_summary = cell_summary.merge(
        anomalous_horizons,
        on=cell_keys,
        how="left",
        validate="one_to_one",
    )
    cell_summary["anomalous_horizons"] = cell_summary["anomalous_horizons"].fillna("")
    cell_summary["cell_consensus_rule"] = (
        f"at_least_{int(min_anomalous_horizons)}_of_{len(horizons)}_horizons"
    )
    cell_summary = cell_summary.sort_values(
        [
            "is_cell_anomaly",
            "n_anomaly_horizons",
            "anomaly_ratio",
            "max_anomaly_score",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return horizon_summary, cell_summary


def plot_alpha_search(
    search_table: pd.DataFrame,
    alpha_by_seed: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
):
    """Plot validation-only alpha stability curves and bootstrap intervals."""

    import matplotlib.pyplot as plt

    seeds = sorted(search_table["seed"].unique())
    fig, axes = plt.subplots(1, len(seeds), figsize=(6 * len(seeds), 4.8), squeeze=False)
    for ax, seed in zip(axes[0], seeds):
        one = search_table[search_table["seed"] == seed].sort_values("alpha")
        selected = alpha_by_seed.loc[alpha_by_seed["seed"] == seed].iloc[0]

        ax.fill_between(
            one["alpha"].to_numpy(dtype=float),
            one["ci_low"].to_numpy(dtype=float),
            one["ci_high"].to_numpy(dtype=float),
            color="#93C5FD",
            alpha=0.35,
            label="Battery bootstrap 95% interval",
        )
        ax.plot(
            one["alpha"],
            one["rank_corr"],
            marker="o",
            color="#2563EB",
            label="H50/H100 rank stability",
        )
        ax.axhline(
            selected["one_se_cutoff"],
            color="#6B7280",
            linestyle="--",
            linewidth=1.2,
            label="One-SE cutoff",
        )
        ax.axvspan(
            selected["plateau_alpha_low"],
            selected["plateau_alpha_high"],
            color="#FDE68A",
            alpha=0.28,
            label="One-SE plateau",
        )
        ax.axvline(
            selected["alpha"],
            color="#DC2626",
            linestyle=":",
            linewidth=2,
            label=f"Selected alpha={selected['alpha']:.2f}",
        )
        ax.set(
            title=f"Seed {seed}",
            xlabel="Alpha: degradation residual weight",
            ylabel="Validation H50/H100 cell-rank correlation",
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Validation-only Alpha Selection (stability heuristic, not anomaly accuracy)",
        fontsize=14,
    )
    fig.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    return fig
