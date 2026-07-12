from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import math
import pickle
import re

import numpy as np
import pandas as pd


TREND_FEATURE_COLUMNS = [
    "log_observed_cycle_span",
    "total_soh_drop",
    "mean_fade_rate_per_100",
    "early_fade_rate_per_100",
    "middle_fade_rate_per_100",
    "late_fade_rate_per_100",
    "knee_phase",
    "knee_strength",
    "normalized_auc",
    "relative_soh_phase_25",
    "relative_soh_phase_50",
    "relative_soh_phase_90",
]

TRANSIENT_FEATURE_COLUMNS = [
    "transient_mad",
    "down_spike_q95",
    "max_down_spike",
    "recovery_total_per_100",
    "roughness_mad",
]


@dataclass(frozen=True)
class CurvePatternConfig:
    """Configuration for retrospective, cell-level SOH curve screening."""

    normalized_grid_points: int = 64
    hampel_window: int = 11
    hampel_n_sigma: float = 4.0
    trend_window_fraction: float = 0.05
    trend_window_min: int = 15
    trend_window_max: int = 101
    enforce_nonincreasing_trend: bool = True
    crossfit_folds: int = 5
    crossfit_repeats: int = 20
    isolation_trees: int = 300
    isolation_max_samples: int = 64
    candidate_p: float = 0.10
    strong_p: float = 0.05
    required_selection_frequency: float = 0.80
    embedding_components: int = 3
    nearest_pattern_peers: int = 3
    cluster_k_max: int = 4
    cluster_min_silhouette: float = 0.25
    random_state: int = 20260712
    verbose: bool = False


@dataclass
class CurvePatternResult:
    raw_curves: pd.DataFrame
    processed_curves: pd.DataFrame
    feature_table: pd.DataFrame
    curve_vectors: pd.DataFrame
    pattern_summary: pd.DataFrame
    embedding_table: pd.DataFrame
    cluster_summary: pd.DataFrame
    crossfit_audit: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{context} is missing columns: {missing}")


def _odd_window(value: int, *, minimum: int = 3) -> int:
    window = max(int(value), int(minimum))
    return window if window % 2 == 1 else window + 1


def _robust_scale(values: np.ndarray) -> tuple[float, float, str]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        raise ValueError("Cannot fit a robust scale to an empty array")
    center = float(np.median(array))
    q25, q75 = np.quantile(array, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349)
    if np.isfinite(iqr_scale) and iqr_scale > 1e-12:
        return center, iqr_scale, "median_iqr"
    mad_scale = float(1.4826 * np.median(np.abs(array - center)))
    if np.isfinite(mad_scale) and mad_scale > 1e-12:
        return center, mad_scale, "median_mad_fallback"
    std_scale = float(np.std(array, ddof=0))
    if np.isfinite(std_scale) and std_scale > 1e-12:
        return center, std_scale, "median_std_fallback"
    return center, 1.0, "constant_feature_unit_fallback"


def _isotonic_increasing(values: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators fit with equal weights."""

    y = np.asarray(values, dtype=float)
    block_values: list[float] = []
    block_weights: list[int] = []
    for value in y:
        block_values.append(float(value))
        block_weights.append(1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            total_weight = block_weights[-2] + block_weights[-1]
            pooled = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / total_weight
            block_values[-2:] = [float(pooled)]
            block_weights[-2:] = [int(total_weight)]
    return np.concatenate(
        [np.full(weight, value, dtype=float) for value, weight in zip(block_values, block_weights)]
    )


def _nonincreasing_fit(values: np.ndarray) -> np.ndarray:
    return -_isotonic_increasing(-np.asarray(values, dtype=float))


def _hampel_filter(
    values: np.ndarray,
    *,
    window: int,
    n_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    series = pd.Series(np.asarray(values, dtype=float))
    rolling_median = series.rolling(window, center=True, min_periods=1).median()
    absolute_deviation = (series - rolling_median).abs()
    rolling_mad = absolute_deviation.rolling(
        window, center=True, min_periods=1
    ).median()
    global_mad = float(np.median(np.abs(series.to_numpy() - np.median(series))))
    fallback = max(1.4826 * global_mad, 1e-8)
    local_sigma = (1.4826 * rolling_mad).where(rolling_mad > 1e-12, fallback)
    is_outlier = absolute_deviation > float(n_sigma) * local_sigma
    filtered = series.where(~is_outlier, rolling_median)
    return filtered.to_numpy(dtype=float), is_outlier.to_numpy(dtype=bool)


def _linear_fade_rate(cycle: np.ndarray, soh: np.ndarray) -> float:
    x = np.asarray(cycle, dtype=float)
    y = np.asarray(soh, dtype=float)
    if len(x) < 2 or np.ptp(x) <= 0:
        return 0.0
    centered_x = x - np.mean(x)
    denominator = float(np.sum(centered_x**2))
    if denominator <= 0:
        return 0.0
    slope = float(np.sum(centered_x * (y - np.mean(y))) / denominator)
    return float(-100.0 * slope)


def _first_threshold_cycle(
    cycles: np.ndarray,
    trend: np.ndarray,
    threshold: float,
) -> tuple[float, bool]:
    matches = np.flatnonzero(np.asarray(trend, dtype=float) <= float(threshold))
    if len(matches) == 0:
        return float("nan"), True
    return float(np.asarray(cycles, dtype=float)[matches[0]]), False


def build_curve_pattern_features(
    raw_curves: pd.DataFrame,
    *,
    config: CurvePatternConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create robust trend, transient, and normalized-curve representations.

    The returned scalar trend features deliberately exclude transient features.
    This prevents a short down-up measurement spike from becoming a persistent
    degradation candidate. The analysis is retrospective because it uses the
    complete observed SOH curve of each cell.
    """

    required = ["battery_id", "cell_id", "batch_id", "cycle", "soh"]
    _require_columns(raw_curves, required, "raw MATR SOH curves")
    if int(config.normalized_grid_points) < 16:
        raise ValueError("normalized_grid_points must be at least 16")
    if not 0.0 < float(config.trend_window_fraction) < 0.5:
        raise ValueError("trend_window_fraction must be in (0, 0.5)")

    duplicated = raw_curves.duplicated(["battery_id", "cycle"], keep=False)
    if duplicated.any():
        raise ValueError("raw curves must contain one SOH value per battery/cycle")

    feature_rows: list[dict[str, object]] = []
    processed_parts: list[pd.DataFrame] = []
    vector_rows: list[dict[str, object]] = []
    grid_phase = np.linspace(0.0, 1.0, int(config.normalized_grid_points))

    for battery_id, group in raw_curves.groupby("battery_id", sort=True):
        metadata = group[["cell_id", "batch_id"]].drop_duplicates()
        if len(metadata) != 1:
            raise ValueError(f"Battery {battery_id} maps to multiple cell/batch values")
        group = group.sort_values("cycle").copy()
        cycles = pd.to_numeric(group["cycle"], errors="coerce").to_numpy(dtype=float)
        soh = pd.to_numeric(group["soh"], errors="coerce").to_numpy(dtype=float)
        if len(cycles) < 20:
            raise ValueError(f"Battery {battery_id} has fewer than 20 valid SOH points")
        if not np.isfinite(cycles).all() or not np.isfinite(soh).all():
            raise ValueError(f"Battery {battery_id} contains non-finite curve values")
        if not np.all(np.diff(cycles) > 0):
            raise ValueError(f"Battery {battery_id} cycles are not strictly increasing")

        hampel_window = min(_odd_window(config.hampel_window), _odd_window(len(soh)))
        if hampel_window > len(soh):
            hampel_window = len(soh) if len(soh) % 2 else len(soh) - 1
        filtered, hampel_outlier = _hampel_filter(
            soh,
            window=hampel_window,
            n_sigma=config.hampel_n_sigma,
        )

        requested_trend_window = int(round(len(soh) * config.trend_window_fraction))
        trend_window = _odd_window(
            min(
                max(requested_trend_window, config.trend_window_min),
                config.trend_window_max,
            )
        )
        if trend_window > len(soh):
            trend_window = len(soh) if len(soh) % 2 else len(soh) - 1
        local_smooth = (
            pd.Series(filtered)
            .rolling(trend_window, center=True, min_periods=1)
            .median()
            .rolling(3, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )
        trend = (
            _nonincreasing_fit(local_smooth)
            if config.enforce_nonincreasing_trend
            else local_smooth.copy()
        )

        span = float(cycles[-1] - cycles[0])
        if span <= 0:
            raise ValueError(f"Battery {battery_id} has non-positive observed span")
        phase = (cycles - cycles[0]) / span
        trend_grid = np.interp(grid_phase, phase, trend)
        initial_soh = float(trend_grid[0])
        final_soh = float(trend_grid[-1])
        relative_grid = trend_grid / max(abs(initial_soh), 1e-8)
        fade_profile = initial_soh - trend_grid

        segment_masks = [
            grid_phase <= 1.0 / 3.0,
            (grid_phase > 1.0 / 3.0) & (grid_phase <= 2.0 / 3.0),
            grid_phase > 2.0 / 3.0,
        ]
        segment_rates = [
            _linear_fade_rate(
                cycles[(phase >= bounds[0]) & (phase <= bounds[1])],
                trend[(phase >= bounds[0]) & (phase <= bounds[1])],
            )
            for bounds in [(0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0)]
        ]
        # If sparse/gapped cycles leave too few points in a segment, fall back
        # to the normalized interpolation for a finite, non-extrapolated slope.
        for index, mask in enumerate(segment_masks):
            if not np.isfinite(segment_rates[index]) or np.sum(mask) < 2:
                segment_rates[index] = _linear_fade_rate(
                    grid_phase[mask] * span + cycles[0], trend_grid[mask]
                )

        chord = trend_grid[0] + (trend_grid[-1] - trend_grid[0]) * grid_phase
        chord_deviation = trend_grid - chord
        knee_index = int(np.argmax(np.abs(chord_deviation)))
        knee_phase = float(grid_phase[knee_index])
        knee_strength = float(abs(chord_deviation[knee_index]))

        transient_residual = soh - local_smooth
        down_spike = np.maximum(local_smooth - soh, 0.0)
        recovery = np.maximum(np.diff(soh), 0.0)
        second_difference = np.diff(soh, n=2)
        transient_mad = float(
            1.4826
            * np.median(
                np.abs(transient_residual - np.median(transient_residual))
            )
        )
        roughness_mad = float(
            1.4826
            * np.median(
                np.abs(second_difference - np.median(second_difference))
            )
            if len(second_difference)
            else 0.0
        )
        transient_threshold = max(3.0 * transient_mad, 0.002)

        threshold_values: dict[str, object] = {}
        for threshold in [0.95, 0.90, 0.85, 0.80]:
            cycle_value, censored = _first_threshold_cycle(cycles, trend, threshold)
            label = f"{int(round(threshold * 100))}"
            threshold_values[f"cycle_to_soh_{label}"] = cycle_value
            threshold_values[f"soh_{label}_censored"] = bool(censored)

        metadata_row = metadata.iloc[0]
        feature_row: dict[str, object] = {
            "battery_id": str(battery_id),
            "cell_id": str(metadata_row["cell_id"]),
            "batch_id": str(metadata_row["batch_id"]),
            "n_soh_points": int(len(soh)),
            "cycle_start": float(cycles[0]),
            "cycle_end": float(cycles[-1]),
            "observed_cycle_span": span,
            "log_observed_cycle_span": float(np.log1p(span)),
            "initial_soh": initial_soh,
            "final_soh": final_soh,
            "total_soh_drop": float(initial_soh - final_soh),
            "mean_fade_rate_per_100": float(100.0 * (initial_soh - final_soh) / span),
            "early_fade_rate_per_100": float(segment_rates[0]),
            "middle_fade_rate_per_100": float(segment_rates[1]),
            "late_fade_rate_per_100": float(segment_rates[2]),
            "knee_phase": knee_phase,
            "knee_strength": knee_strength,
            "normalized_auc": float(np.trapezoid(relative_grid, grid_phase)),
            "relative_soh_phase_25": float(np.interp(0.25, grid_phase, relative_grid)),
            "relative_soh_phase_50": float(np.interp(0.50, grid_phase, relative_grid)),
            "relative_soh_phase_90": float(np.interp(0.90, grid_phase, relative_grid)),
            "transient_mad": transient_mad,
            "down_spike_q95": float(np.quantile(down_spike, 0.95)),
            "max_down_spike": float(np.max(down_spike)),
            "recovery_total_per_100": float(100.0 * np.sum(recovery) / span),
            "roughness_mad": roughness_mad,
            "hampel_outlier_count": int(np.sum(hampel_outlier)),
            "large_down_spike_count": int(np.sum(down_spike > transient_threshold)),
            "cycle_gap_fraction": float(
                np.mean(np.diff(cycles) > 1.0) if len(cycles) > 1 else 0.0
            ),
            **threshold_values,
        }
        feature_rows.append(feature_row)

        processed = group[["battery_id", "cell_id", "batch_id", "cycle", "soh"]].copy()
        processed["hampel_filtered_soh"] = filtered
        processed["local_smooth_soh"] = local_smooth
        processed["trend_soh"] = trend
        processed["transient_residual"] = transient_residual
        processed["down_spike_magnitude"] = down_spike
        processed["is_hampel_outlier"] = hampel_outlier
        processed_parts.append(processed)

        vector_row: dict[str, object] = {
            "battery_id": str(battery_id),
            "cell_id": str(metadata_row["cell_id"]),
            "batch_id": str(metadata_row["batch_id"]),
        }
        for index, value in enumerate(fade_profile):
            vector_row[f"curve_drop_phase_{index:03d}"] = float(value)
        vector_rows.append(vector_row)

    features = pd.DataFrame(feature_rows).sort_values("battery_id").reset_index(drop=True)
    processed_curves = pd.concat(processed_parts, ignore_index=True).sort_values(
        ["battery_id", "cycle"]
    ).reset_index(drop=True)
    vectors = pd.DataFrame(vector_rows).sort_values("battery_id").reset_index(drop=True)
    numeric_required = TREND_FEATURE_COLUMNS + TRANSIENT_FEATURE_COLUMNS
    if not np.isfinite(features[numeric_required].to_numpy(dtype=float)).all():
        raise RuntimeError("Curve feature extraction produced non-finite detector inputs")
    return features, processed_curves, vectors


def _average_isolation_path_length(size: int | np.ndarray) -> np.ndarray:
    n = np.asarray(size, dtype=float)
    output = np.zeros_like(n, dtype=float)
    mask_two = n == 2
    output[mask_two] = 1.0
    mask = n > 2
    output[mask] = 2.0 * (np.log(n[mask] - 1.0) + np.euler_gamma) - 2.0 * (
        n[mask] - 1.0
    ) / n[mask]
    return output


class _NumpyIsolationForest:
    """Small, dependency-free Isolation Forest used when sklearn is unavailable."""

    def __init__(
        self,
        *,
        n_estimators: int,
        max_samples: int,
        random_state: int,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_samples = int(max_samples)
        self.random_state = int(random_state)
        self.trees: list[tuple] = []
        self.sample_size = 0

    def _build_tree(
        self,
        values: np.ndarray,
        *,
        depth: int,
        max_depth: int,
        rng: np.random.Generator,
    ) -> tuple:
        n_rows = len(values)
        if n_rows <= 1 or depth >= max_depth:
            return ("leaf", int(n_rows))
        minima = np.min(values, axis=0)
        maxima = np.max(values, axis=0)
        available = np.flatnonzero(maxima - minima > 1e-12)
        if len(available) == 0:
            return ("leaf", int(n_rows))
        feature = int(rng.choice(available))
        split = float(rng.uniform(minima[feature], maxima[feature]))
        left_mask = values[:, feature] < split
        if not left_mask.any() or left_mask.all():
            return ("leaf", int(n_rows))
        left = self._build_tree(
            values[left_mask], depth=depth + 1, max_depth=max_depth, rng=rng
        )
        right = self._build_tree(
            values[~left_mask], depth=depth + 1, max_depth=max_depth, rng=rng
        )
        return ("node", feature, split, left, right)

    def fit(self, values: np.ndarray) -> "_NumpyIsolationForest":
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or len(matrix) < 2:
            raise ValueError("Isolation Forest requires a 2D matrix with at least 2 rows")
        if not np.isfinite(matrix).all():
            raise ValueError("Isolation Forest input contains non-finite values")
        self.sample_size = min(max(2, self.max_samples), len(matrix))
        max_depth = int(math.ceil(math.log2(self.sample_size)))
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        for _ in range(self.n_estimators):
            indices = rng.choice(len(matrix), size=self.sample_size, replace=False)
            self.trees.append(
                self._build_tree(
                    matrix[indices], depth=0, max_depth=max_depth, rng=rng
                )
            )
        return self

    def _path_length(self, row: np.ndarray, tree: tuple, depth: int = 0) -> float:
        if tree[0] == "leaf":
            leaf_size = int(tree[1])
            return float(depth + _average_isolation_path_length(leaf_size))
        _, feature, split, left, right = tree
        branch = left if row[int(feature)] < float(split) else right
        return self._path_length(row, branch, depth + 1)

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Isolation Forest must be fit before scoring")
        matrix = np.asarray(values, dtype=float)
        paths = np.asarray(
            [
                np.mean([self._path_length(row, tree) for tree in self.trees])
                for row in matrix
            ],
            dtype=float,
        )
        normalizer = float(_average_isolation_path_length(self.sample_size))
        if normalizer <= 0:
            return np.full(len(matrix), 0.5, dtype=float)
        return np.power(2.0, -paths / normalizer)


def _fit_batch_scaler(
    frame: pd.DataFrame,
    fit_indices: np.ndarray,
    feature_columns: Sequence[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fit_frame = frame.iloc[np.asarray(fit_indices, dtype=int)]
    scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for batch_id, group in fit_frame.groupby("batch_id", sort=True):
        centers = []
        scales = []
        for column in feature_columns:
            center, scale, _ = _robust_scale(group[column].to_numpy(dtype=float))
            centers.append(center)
            scales.append(scale)
        scalers[str(batch_id)] = (
            np.asarray(centers, dtype=float),
            np.asarray(scales, dtype=float),
        )
    return scalers


def _transform_batch_scaled(
    frame: pd.DataFrame,
    indices: np.ndarray,
    feature_columns: Sequence[str],
    scalers: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    rows = frame.iloc[np.asarray(indices, dtype=int)]
    output = np.empty((len(rows), len(feature_columns)), dtype=float)
    for output_index, (_, row) in enumerate(rows.iterrows()):
        batch_id = str(row["batch_id"])
        if batch_id not in scalers:
            raise RuntimeError(f"No fit-fold scaler is available for batch={batch_id}")
        center, scale = scalers[batch_id]
        values = row[list(feature_columns)].to_numpy(dtype=float)
        output[output_index] = np.clip((values - center) / scale, -8.0, 8.0)
    return output


def _stratified_fold_assignments(
    frame: pd.DataFrame,
    *,
    n_folds: int,
    random_state: int,
) -> np.ndarray:
    folds = np.full(len(frame), -1, dtype=int)
    rng = np.random.default_rng(int(random_state))
    for _, group in frame.groupby("batch_id", sort=True):
        indices = group.sort_values("battery_id").index.to_numpy(dtype=int)
        shuffled = indices[rng.permutation(len(indices))]
        folds[shuffled] = np.arange(len(shuffled), dtype=int) % int(n_folds)
    if (folds < 0).any():
        raise RuntimeError("Some cells did not receive a cross-fit fold")
    return folds


def repeated_crossfit_isolation_scores(
    feature_table: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    prefix: str,
    config: CurvePatternConfig,
    random_state_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create held-out scores and empirical tail p-values for every cell."""

    _require_columns(
        feature_table,
        ["battery_id", "cell_id", "batch_id"] + list(feature_columns),
        f"{prefix} cross-fit features",
    )
    if int(config.crossfit_folds) < 3:
        raise ValueError("crossfit_folds must be at least 3")
    if int(config.crossfit_repeats) < 1:
        raise ValueError("crossfit_repeats must be positive")
    if not 0.0 < config.strong_p <= config.candidate_p < 1.0:
        raise ValueError("Require 0 < strong_p <= candidate_p < 1")
    if not 0.0 < config.required_selection_frequency <= 1.0:
        raise ValueError("required_selection_frequency must be in (0, 1]")

    frame = feature_table.sort_values("battery_id").reset_index(drop=True).copy()
    n_cells = len(frame)
    score_matrix = np.full((n_cells, config.crossfit_repeats), np.nan, dtype=float)
    p_matrix = np.full_like(score_matrix, np.nan)
    batch_p_matrix = np.full_like(score_matrix, np.nan)
    minimum_p_matrix = np.full_like(score_matrix, np.nan)
    audit_rows: list[dict[str, object]] = []

    for repeat in range(config.crossfit_repeats):
        if config.verbose and (
            repeat == 0
            or (repeat + 1) % 5 == 0
            or repeat + 1 == config.crossfit_repeats
        ):
            print(
                f"[{prefix}] cross-fit repeat {repeat + 1}/{config.crossfit_repeats}",
                flush=True,
            )
        repeat_seed = (
            int(config.random_state)
            + int(random_state_offset)
            + repeat * 100_003
        )
        folds = _stratified_fold_assignments(
            frame,
            n_folds=config.crossfit_folds,
            random_state=repeat_seed,
        )
        for score_fold in range(config.crossfit_folds):
            calibration_fold = (score_fold + 1) % config.crossfit_folds
            score_indices = np.flatnonzero(folds == score_fold)
            calibration_indices = np.flatnonzero(folds == calibration_fold)
            fit_indices = np.flatnonzero(
                (folds != score_fold) & (folds != calibration_fold)
            )
            if min(len(score_indices), len(calibration_indices), len(fit_indices)) < 2:
                raise RuntimeError("Cross-fit fold is too small")

            scalers = _fit_batch_scaler(frame, fit_indices, feature_columns)
            fit_values = _transform_batch_scaled(
                frame, fit_indices, feature_columns, scalers
            )
            calibration_values = _transform_batch_scaled(
                frame, calibration_indices, feature_columns, scalers
            )
            score_values = _transform_batch_scaled(
                frame, score_indices, feature_columns, scalers
            )
            model = _NumpyIsolationForest(
                n_estimators=config.isolation_trees,
                max_samples=config.isolation_max_samples,
                random_state=repeat_seed + score_fold * 7_919,
            ).fit(fit_values)
            calibration_scores = model.score_samples(calibration_values)
            held_out_scores = model.score_samples(score_values)
            tail_counts = (
                calibration_scores[:, None] >= held_out_scores[None, :]
            ).sum(axis=0)
            p_values = (1.0 + tail_counts) / (len(calibration_scores) + 1.0)

            score_matrix[score_indices, repeat] = held_out_scores
            p_matrix[score_indices, repeat] = p_values
            minimum_p_matrix[score_indices, repeat] = 1.0 / (
                len(calibration_scores) + 1.0
            )

            calibration_batches = frame.iloc[calibration_indices][
                "batch_id"
            ].astype(str).to_numpy()
            held_out_batches = frame.iloc[score_indices]["batch_id"].astype(
                str
            ).to_numpy()
            for local_index, batch_id in enumerate(held_out_batches):
                same_batch_scores = calibration_scores[
                    calibration_batches == batch_id
                ]
                if len(same_batch_scores) == 0:
                    batch_p = np.nan
                else:
                    batch_p = (
                        1.0
                        + np.sum(same_batch_scores >= held_out_scores[local_index])
                    ) / (len(same_batch_scores) + 1.0)
                batch_p_matrix[score_indices[local_index], repeat] = batch_p

            audit_rows.append(
                {
                    "score_channel": str(prefix),
                    "repeat": int(repeat),
                    "score_fold": int(score_fold),
                    "calibration_fold": int(calibration_fold),
                    "fit_cells": int(len(fit_indices)),
                    "calibration_cells": int(len(calibration_indices)),
                    "held_out_cells": int(len(score_indices)),
                    "minimum_attainable_p": float(
                        1.0 / (len(calibration_scores) + 1.0)
                    ),
                    "candidate_resolution_available": bool(
                        1.0 / (len(calibration_scores) + 1.0)
                        <= config.candidate_p
                    ),
                    "strong_resolution_available": bool(
                        1.0 / (len(calibration_scores) + 1.0)
                        <= config.strong_p
                    ),
                    "fit_batches": int(frame.iloc[fit_indices]["batch_id"].nunique()),
                }
            )

    if (
        np.isnan(score_matrix).any()
        or np.isnan(p_matrix).any()
        or np.isnan(minimum_p_matrix).any()
    ):
        raise RuntimeError("Cross-fitting did not score every cell in every repeat")

    result = frame[["battery_id", "cell_id", "batch_id"]].copy()
    result[f"{prefix}_isolation_score_median"] = np.median(score_matrix, axis=1)
    result[f"{prefix}_isolation_score_q10"] = np.quantile(
        score_matrix, 0.10, axis=1
    )
    result[f"{prefix}_isolation_score_q90"] = np.quantile(
        score_matrix, 0.90, axis=1
    )
    result[f"{prefix}_empirical_p_median"] = np.median(p_matrix, axis=1)
    result[f"{prefix}_empirical_p_q10"] = np.quantile(p_matrix, 0.10, axis=1)
    result[f"{prefix}_empirical_p_q90"] = np.quantile(p_matrix, 0.90, axis=1)
    result[f"{prefix}_within_batch_p_median"] = np.nanmedian(
        batch_p_matrix, axis=1
    )
    result[f"{prefix}_candidate_frequency"] = np.mean(
        p_matrix <= config.candidate_p, axis=1
    )
    result[f"{prefix}_strong_frequency"] = np.mean(
        p_matrix <= config.strong_p, axis=1
    )
    result[f"{prefix}_minimum_attainable_p_median"] = np.median(
        minimum_p_matrix, axis=1
    )
    result[f"is_{prefix}_candidate"] = (
        result[f"{prefix}_empirical_p_median"] <= config.candidate_p
    ) & (
        result[f"{prefix}_candidate_frequency"]
        >= config.required_selection_frequency
    )
    result[f"is_{prefix}_strong_candidate"] = (
        result[f"{prefix}_empirical_p_median"] <= config.strong_p
    ) & (
        result[f"{prefix}_strong_frequency"]
        >= config.required_selection_frequency
    )
    result[f"{prefix}_candidate_p_cutoff"] = float(config.candidate_p)
    result[f"{prefix}_strong_p_cutoff"] = float(config.strong_p)
    result[f"{prefix}_required_selection_frequency"] = float(
        config.required_selection_frequency
    )
    result[f"{prefix}_score_method"] = (
        "batch_robust_scaled_pooled_numpy_isolation_forest_crossfit"
    )
    return result, pd.DataFrame(audit_rows)


def _descriptive_batch_z(
    features: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    output = features[["battery_id", "batch_id"]].copy()
    for column in columns:
        values = np.zeros(len(features), dtype=float)
        for _, group in features.groupby("batch_id", sort=True):
            center, scale, _ = _robust_scale(group[column].to_numpy(dtype=float))
            values[group.index.to_numpy(dtype=int)] = (
                group[column].to_numpy(dtype=float) - center
            ) / scale
        output[f"{column}_batch_z"] = values
    return output


def _pca_embedding(matrix: np.ndarray, n_components: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    centered = values - np.mean(values, axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        return np.zeros((len(values), int(n_components)), dtype=float)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    available = min(int(n_components), u.shape[1])
    scores = u[:, :available] * singular_values[:available]
    if available < int(n_components):
        scores = np.pad(scores, ((0, 0), (0, int(n_components) - available)))
    return scores


def _kmeans(
    values: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    n_init: int = 20,
    max_iter: int = 200,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(random_state))
    best_labels: np.ndarray | None = None
    best_inertia = float("inf")
    for _ in range(int(n_init)):
        first = int(rng.integers(0, len(matrix)))
        centers = [matrix[first].copy()]
        while len(centers) < int(n_clusters):
            distances = np.min(
                np.stack(
                    [np.sum((matrix - center) ** 2, axis=1) for center in centers],
                    axis=1,
                ),
                axis=1,
            )
            if np.sum(distances) <= 1e-12:
                remaining = [
                    index
                    for index in range(len(matrix))
                    if not any(np.allclose(matrix[index], center) for center in centers)
                ]
                centers.append(matrix[remaining[0] if remaining else 0].copy())
            else:
                centers.append(
                    matrix[int(rng.choice(len(matrix), p=distances / np.sum(distances)))].copy()
                )
        centers_array = np.asarray(centers, dtype=float)
        labels = np.full(len(matrix), -1, dtype=int)
        for _ in range(int(max_iter)):
            distance_matrix = np.stack(
                [np.sum((matrix - center) ** 2, axis=1) for center in centers_array],
                axis=1,
            )
            new_labels = np.argmin(distance_matrix, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for cluster in range(int(n_clusters)):
                members = matrix[labels == cluster]
                if len(members):
                    centers_array[cluster] = np.mean(members, axis=0)
        inertia = float(
            np.sum(
                [
                    np.sum((matrix[labels == cluster] - centers_array[cluster]) ** 2)
                    for cluster in range(int(n_clusters))
                ]
            )
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    if best_labels is None:
        raise RuntimeError("K-means did not produce labels")
    return best_labels, best_inertia


def _silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    matrix = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    clusters = np.unique(labels)
    if len(clusters) < 2 or any(np.sum(labels == cluster) < 2 for cluster in clusters):
        return -1.0
    pairwise = np.sqrt(
        np.sum((matrix[:, None, :] - matrix[None, :, :]) ** 2, axis=2)
    )
    scores = []
    for index in range(len(matrix)):
        same = labels == labels[index]
        same[index] = False
        a = float(np.mean(pairwise[index, same])) if same.any() else 0.0
        b = min(
            float(np.mean(pairwise[index, labels == cluster]))
            for cluster in clusters
            if cluster != labels[index]
        )
        denominator = max(a, b)
        scores.append(0.0 if denominator <= 0 else (b - a) / denominator)
    return float(np.mean(scores))


def build_curve_embedding_and_clusters(
    feature_table: pd.DataFrame,
    curve_vectors: pd.DataFrame,
    *,
    config: CurvePatternConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build descriptive curve-shape embedding and auxiliary clusters."""

    keys = ["battery_id", "cell_id", "batch_id"]
    vector_columns = [
        column for column in curve_vectors.columns if column.startswith("curve_drop_phase_")
    ]
    merged = feature_table[keys].merge(
        curve_vectors[keys + vector_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
    ).sort_values("battery_id").reset_index(drop=True)
    scaled = np.zeros((len(merged), len(vector_columns)), dtype=float)
    for _, group in merged.groupby("batch_id", sort=True):
        indices = group.index.to_numpy(dtype=int)
        for column_index, column in enumerate(vector_columns):
            center, scale, _ = _robust_scale(group[column].to_numpy(dtype=float))
            scaled[indices, column_index] = np.clip(
                (group[column].to_numpy(dtype=float) - center) / scale,
                -8.0,
                8.0,
            )
    embedding = _pca_embedding(scaled, config.embedding_components)
    output = merged[keys].copy()
    for component in range(config.embedding_components):
        output[f"curve_pc{component + 1}"] = embedding[:, component]

    output["pattern_cluster_id"] = ""
    output["pattern_cluster_size"] = 0
    output["pattern_cluster_silhouette"] = np.nan
    output["nearest_pattern_peers"] = ""
    output["nearest_pattern_peer_distances"] = ""
    cluster_rows: list[dict[str, object]] = []

    pc_columns = [f"curve_pc{index + 1}" for index in range(config.embedding_components)]
    for batch_id, group in output.groupby("batch_id", sort=True):
        indices = group.index.to_numpy(dtype=int)
        values = group[pc_columns].to_numpy(dtype=float)
        selected_labels = np.zeros(len(group), dtype=int)
        selected_k = 1
        selected_silhouette = 0.0
        for k in range(2, min(config.cluster_k_max, len(group) - 1) + 1):
            labels, _ = _kmeans(
                values,
                n_clusters=k,
                random_state=config.random_state + int(k) * 101 + len(indices),
            )
            silhouette = _silhouette(values, labels)
            if silhouette > selected_silhouette:
                selected_labels = labels
                selected_k = k
                selected_silhouette = silhouette
        if selected_silhouette < config.cluster_min_silhouette:
            selected_labels = np.zeros(len(group), dtype=int)
            selected_k = 1
            selected_silhouette = 0.0

        pairwise = np.sqrt(
            np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=2)
        )
        np.fill_diagonal(pairwise, np.inf)
        battery_ids = group["battery_id"].astype(str).to_numpy()
        for local_index, global_index in enumerate(indices):
            cluster = int(selected_labels[local_index])
            cluster_id = f"{batch_id}_shape_cluster_{cluster}"
            cluster_size = int(np.sum(selected_labels == cluster))
            nearest = np.argsort(pairwise[local_index])[
                : min(config.nearest_pattern_peers, len(group) - 1)
            ]
            output.loc[global_index, "pattern_cluster_id"] = cluster_id
            output.loc[global_index, "pattern_cluster_size"] = cluster_size
            output.loc[global_index, "pattern_cluster_silhouette"] = float(
                selected_silhouette
            )
            output.loc[global_index, "nearest_pattern_peers"] = ",".join(
                battery_ids[nearest]
            )
            output.loc[global_index, "nearest_pattern_peer_distances"] = ",".join(
                f"{value:.6g}" for value in pairwise[local_index, nearest]
            )
        for cluster in sorted(np.unique(selected_labels)):
            members = battery_ids[selected_labels == cluster]
            cluster_rows.append(
                {
                    "batch_id": str(batch_id),
                    "selected_cluster_count": int(selected_k),
                    "cluster_id": f"{batch_id}_shape_cluster_{int(cluster)}",
                    "cluster_size": int(len(members)),
                    "cluster_fraction": float(len(members) / len(group)),
                    "silhouette": float(selected_silhouette),
                    "member_battery_ids": ",".join(members),
                    "cluster_role": "descriptive_only_not_anomaly_label",
                }
            )
    return output, pd.DataFrame(cluster_rows)


def _assign_pattern_explanation(summary: pd.DataFrame) -> pd.Series:
    labels = []
    for _, row in summary.iterrows():
        if bool(row["is_transient_candidate"]) and not bool(row["is_trend_candidate"]):
            labels.append("transient_or_data_quality_pattern")
        elif bool(row["is_trend_candidate"]):
            if (
                row["log_observed_cycle_span_batch_z"] <= -1.5
                and row["mean_fade_rate_per_100_batch_z"] >= 1.5
            ):
                labels.append("rapid_fade_pattern")
            elif row["knee_strength_batch_z"] >= 2.0:
                labels.append(
                    "early_knee_pattern"
                    if row["knee_phase"] < 0.65
                    else "late_knee_pattern"
                )
            elif row["log_observed_cycle_span_batch_z"] >= 2.0:
                labels.append("long_life_pattern")
            else:
                labels.append("atypical_persistent_curve_pattern")
        elif row["log_observed_cycle_span_batch_z"] >= 2.0:
            labels.append("long_life_pattern_no_stable_tail_evidence")
        else:
            labels.append("no_stable_pattern_tail_evidence")
    return pd.Series(labels, index=summary.index, dtype="object")


def analyze_curve_patterns(
    raw_curves: pd.DataFrame,
    *,
    config: CurvePatternConfig | None = None,
) -> CurvePatternResult:
    """Run the complete retrospective curve-pattern and transient analysis."""

    config = config or CurvePatternConfig()
    features, processed, vectors = build_curve_pattern_features(
        raw_curves, config=config
    )
    trend_scores, trend_audit = repeated_crossfit_isolation_scores(
        features,
        feature_columns=TREND_FEATURE_COLUMNS,
        prefix="trend",
        config=config,
        random_state_offset=0,
    )
    transient_scores, transient_audit = repeated_crossfit_isolation_scores(
        features,
        feature_columns=TRANSIENT_FEATURE_COLUMNS,
        prefix="transient",
        config=config,
        random_state_offset=50_000_003,
    )
    embedding, clusters = build_curve_embedding_and_clusters(
        features, vectors, config=config
    )
    descriptive_z = _descriptive_batch_z(
        features,
        [
            "log_observed_cycle_span",
            "mean_fade_rate_per_100",
            "knee_strength",
            "max_down_spike",
        ],
    )

    keys = ["battery_id", "cell_id", "batch_id"]
    summary = features.merge(trend_scores, on=keys, validate="one_to_one")
    summary = summary.merge(transient_scores, on=keys, validate="one_to_one")
    summary = summary.merge(embedding, on=keys, validate="one_to_one")
    summary = summary.merge(
        descriptive_z.drop(columns=["batch_id"]),
        on="battery_id",
        validate="one_to_one",
    )
    summary["review_status"] = "no_stable_pattern_tail_evidence"
    summary.loc[
        summary["is_transient_candidate"], "review_status"
    ] = "transient_or_data_quality_candidate"
    summary.loc[
        summary["is_trend_candidate"], "review_status"
    ] = "persistent_degradation_pattern_candidate"
    summary.loc[
        summary["is_trend_candidate"] & summary["is_transient_candidate"],
        "review_status",
    ] = "persistent_and_transient_pattern_candidate"
    summary["pattern_explanation"] = _assign_pattern_explanation(summary)
    summary["detector_scope"] = "retrospective_full_observed_curve"
    summary["formal_validity_note"] = (
        "exploratory_cross_fitted_empirical_rarity_no_fault_probability_claim"
    )
    summary = summary.sort_values(
        [
            "is_trend_strong_candidate",
            "is_trend_candidate",
            "trend_candidate_frequency",
            "trend_empirical_p_median",
            "is_transient_candidate",
            "transient_empirical_p_median",
        ],
        ascending=[False, False, False, True, False, True],
    ).reset_index(drop=True)
    crossfit_audit = pd.concat(
        [trend_audit, transient_audit], ignore_index=True
    )
    return CurvePatternResult(
        raw_curves=raw_curves.copy(),
        processed_curves=processed,
        feature_table=features,
        curve_vectors=vectors,
        pattern_summary=summary,
        embedding_table=embedding,
        cluster_summary=clusters,
        crossfit_audit=crossfit_audit,
    )


def load_matr_soh_curves(
    matr_dir: str | Path,
    *,
    expected_battery_count: int | None = None,
    expected_battery_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load trusted local MATR pickle files with the model's target formula.

    Pickle files can execute code while loading; this function must only be
    used with the trusted local MATR dataset supplied for this project.
    """

    try:
        import soh_gru_dsconv_pipeline as soh_pipe
    except ImportError as exc:
        raise ImportError(
            "soh_gru_dsconv_pipeline must be importable before loading MATR curves"
        ) from exc

    root = Path(matr_dir)
    paths = sorted(
        path
        for path in root.rglob("*.pkl")
        if re.fullmatch(r"MATR_b\d+c\d+", path.stem)
    )
    if not paths:
        raise FileNotFoundError(f"No MATR cell pickle files found under {root}")
    curve_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for path in paths:
        with path.open("rb") as handle:
            cell = pickle.load(handle)
        if not isinstance(cell, dict) or "cycle_data" not in cell:
            raise ValueError(f"Unsupported MATR pickle structure: {path}")
        cycles = sorted(
            list(cell["cycle_data"]),
            key=lambda item: int(item.get("cycle_number", 0)),
        )
        reference_capacity = float(soh_pipe.infer_reference_capacity(cell, cycles))
        battery_id = path.stem
        cell_id = str(cell.get("cell_id", battery_id))
        match = re.search(r"(b\d+)", battery_id)
        batch_id = match.group(1).lower() if match else "unknown"
        rows: list[dict[str, object]] = []
        skipped = 0
        for index, cycle in enumerate(cycles):
            cycle_number = int(cycle.get("cycle_number", index + 1))
            try:
                soh = float(soh_pipe.extract_soh_label(cycle, reference_capacity))
            except Exception:
                skipped += 1
                continue
            if not np.isfinite(soh):
                skipped += 1
                continue
            rows.append(
                {
                    "battery_id": battery_id,
                    "cell_id": cell_id,
                    "batch_id": batch_id,
                    "cycle": cycle_number,
                    "soh": soh,
                    "reference_capacity": reference_capacity,
                    "source_file": str(path),
                }
            )
        if not rows:
            raise ValueError(f"No valid SOH targets extracted from {path}")
        curve = (
            pd.DataFrame(rows)
            .drop_duplicates("cycle", keep="last")
            .sort_values("cycle")
            .reset_index(drop=True)
        )
        if batch_id == "unknown":
            raise ValueError(f"Could not infer MATR batch from {battery_id}")
        curve_parts.append(curve)
        cycle_values = curve["cycle"].to_numpy(dtype=float)
        audit_rows.append(
            {
                "battery_id": battery_id,
                "cell_id": cell_id,
                "batch_id": batch_id,
                "source_file": str(path),
                "raw_cycle_records": int(len(cycles)),
                "valid_unique_soh_points": int(len(curve)),
                "skipped_soh_records": int(skipped),
                "duplicate_cycle_records_removed": int(len(rows) - len(curve)),
                "cycle_min": int(curve["cycle"].min()),
                "cycle_max": int(curve["cycle"].max()),
                "cycle_gap_fraction": float(
                    np.mean(np.diff(cycle_values) > 1.0)
                    if len(cycle_values) > 1
                    else 0.0
                ),
                "reference_capacity": reference_capacity,
            }
        )
        # Release the potentially very large Python-list pickle payload before
        # the next cell is unpickled. Only the compact numeric curve is kept.
        del cell, cycles
    all_curves = pd.concat(curve_parts, ignore_index=True).sort_values(
        ["battery_id", "cycle"]
    ).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows).sort_values("battery_id").reset_index(drop=True)
    observed_ids = set(audit["battery_id"].astype(str))
    if expected_battery_count is not None and len(observed_ids) != int(
        expected_battery_count
    ):
        raise RuntimeError(
            f"Expected {expected_battery_count} MATR batteries, loaded {len(observed_ids)}"
        )
    if expected_battery_ids is not None:
        expected = set(str(value) for value in expected_battery_ids)
        if observed_ids != expected:
            raise RuntimeError(
                "Loaded MATR IDs differ from the expected manifest: "
                f"missing={sorted(expected - observed_ids)[:10]}, "
                f"extra={sorted(observed_ids - expected)[:10]}"
            )
    if all_curves.duplicated(["battery_id", "cycle"]).any():
        raise RuntimeError("Duplicate battery/cycle rows remain after loading")
    return all_curves, audit


def assert_curve_prediction_parity(
    raw_curves: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    *,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Assert that full-curve SOH equals the future-SOH model target definition."""

    _require_columns(raw_curves, ["battery_id", "cycle", "soh"], "raw curves")
    _require_columns(
        prediction_rows,
        ["battery_id", "target_cycle", "actual_soh"],
        "prediction parity rows",
    )
    left = prediction_rows[["battery_id", "target_cycle", "actual_soh"]].copy()
    spread = left.groupby(["battery_id", "target_cycle"])["actual_soh"].agg(
        lambda values: float(np.max(values) - np.min(values))
    )
    if spread.gt(float(tolerance)).any():
        raise RuntimeError("Prediction rows disagree on actual SOH for a battery/cycle")
    left = left.drop_duplicates(["battery_id", "target_cycle"])
    right = raw_curves[["battery_id", "cycle", "soh"]].drop_duplicates(
        ["battery_id", "cycle"]
    )
    parity = left.merge(
        right,
        left_on=["battery_id", "target_cycle"],
        right_on=["battery_id", "cycle"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing = parity[parity["_merge"] != "both"]
    if not missing.empty:
        raise RuntimeError(
            "Some model targets are missing from raw curves: "
            + missing[["battery_id", "target_cycle"]]
            .head(10)
            .to_dict("records")
            .__repr__()
        )
    parity["abs_diff"] = (parity["actual_soh"] - parity["soh"]).abs()
    max_difference = float(parity["abs_diff"].max())
    if max_difference > float(tolerance):
        raise RuntimeError(
            f"Raw SOH/model target mismatch: max_abs_diff={max_difference}"
        )
    return parity.drop(columns=["_merge"])
