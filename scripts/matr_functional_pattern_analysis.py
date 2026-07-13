from __future__ import annotations

"""Retrospective functional analysis of MATR SOH degradation curves.

This module deliberately separates four questions which should not share one
opaque anomaly score:

* Is a cell's persistent curve unusual at the same *absolute cycle* within its
  batch?
* Is its degradation *shape* unusual after removing horizontal T90 lifetime
  and vertical total-drop effects?
* Are its sustained SOH landmark times unusual, with censoring kept explicit?
* Does it mainly contain short down/up transients or data-quality artifacts?

The implementation uses only NumPy and pandas.  It does not use contamination,
top-k selection, or Isolation Forest; consequently every candidate set may be
empty.  ``rarity`` values are robust descriptive scores, not p-values and not
probabilities of physical failure.
"""

from dataclasses import asdict, dataclass
from typing import Sequence
import math

import numpy as np
import pandas as pd

try:  # Re-export the trusted data/parity helpers without duplicating them.
    from matr_curve_pattern_analysis import (
        assert_curve_prediction_parity,
        load_matr_soh_curves,
    )
except ImportError:  # Package-style import, useful for tests and IDEs.
    from .matr_curve_pattern_analysis import (  # type: ignore
        assert_curve_prediction_parity,
        load_matr_soh_curves,
    )


LANDMARK_THRESHOLDS = (0.95, 0.90, 0.85, 0.80)


@dataclass(frozen=True)
class FunctionalPatternConfig:
    """Configuration for label-free, retrospective functional screening.

    The default cutoffs are robust tolerance rules rather than empirical
    percentiles.  They can therefore select zero cells.  Any scientific report
    should include a sensitivity analysis over smoothing and rarity settings.
    """

    smooth_half_window_cycles: float = 20.0
    smooth_min_points: int = 7
    smooth_robust_iterations: int = 2
    smooth_huber_c: float = 1.5
    landmark_sustain_cycles: float = 20.0
    landmark_sustain_fraction: float = 0.80
    landmark_min_points: int = 3
    shape_landmark_soh: float = 0.90
    shape_min_span_cycles: float = 40.0
    shape_min_total_drop: float = 0.03
    shape_smooth_half_window_fraction: float = 0.05
    shape_smooth_min_points: int = 5
    shape_max_internal_gap_cycles: float = 25.0
    shape_max_internal_gap_fraction: float = 0.15
    shape_level_scale_floor: float = 0.005
    shape_derivative_scale_floor: float = 0.05
    shape_grid_points: int = 64
    absolute_grid_points: int = 64
    absolute_min_batch_cells: int = 6
    absolute_min_coverage: float = 0.75
    absolute_level_scale_floor: float = 0.002
    absolute_derivative_scale_floor: float = 0.00002
    fpca_variance_fraction: float = 0.90
    fpca_min_components: int = 1
    fpca_max_components: int = 6
    fpca_robust_iterations: int = 5
    knn_neighbors: int = 5
    level_distance_weight: float = 0.60
    derivative_distance_weight: float = 0.40
    rarity_cutoff: float = 3.5
    strong_rarity_cutoff: float = 5.0
    individual_method_consensus: int = 2
    minimum_adjusted_curve_rms: float = 0.75
    required_selection_frequency: float = 0.80
    stability_repeats: int = 30
    stability_grid_fraction: float = 0.80
    group_mutual_neighbors: int = 3
    group_edge_stability: float = 0.90
    group_min_size: int = 3
    group_max_size: int = 5
    group_max_fraction: float = 0.12
    group_separation_ratio: float = 2.00
    group_center_rarity_cutoff: float = 4.00
    group_center_min_effect: float = 1.00
    group_min_edge_density: float = 0.67
    group_min_member_degree: int = 2
    group_required_view_support: int = 1
    lifetime_group_mutual_neighbors: int = 2
    lifetime_group_min_observed_landmarks: int = 2
    lifetime_group_min_common_landmarks: int = 3
    lifetime_group_landmark_fraction: float = 0.75
    lifetime_log_duration_scale_floor: float = 0.05
    lifetime_group_min_size: int = 2
    lifetime_group_max_size: int = 5
    lifetime_group_max_fraction: float = 0.12
    lifetime_group_edge_stability: float = 0.90
    lifetime_group_min_pair_coverage: float = 0.25
    lifetime_group_separation_ratio: float = 2.00
    lifetime_group_center_rarity_cutoff: float = 4.00
    lifetime_group_acceleration_ratio: float = 0.80
    lifetime_group_pair_duration_ratio: float = 1.25
    lifetime_peer_count: int = 4
    transient_min_spike: float = 0.002
    transient_mad_effect_floor: float = 0.0005
    transient_q95_effect_floor: float = 0.001
    transient_max_spike_effect_floor: float = 0.003
    transient_recovery_effect_floor: float = 0.001
    transient_roughness_effect_floor: float = 0.0005
    random_state: int = 20260712
    verbose: bool = False


@dataclass
class FunctionalPatternResult:
    """Tables produced by :func:`analyze_functional_patterns`."""

    pattern_summary: pd.DataFrame
    processed_curves: pd.DataFrame
    shape_representation: pd.DataFrame
    absolute_representation: pd.DataFrame
    landmark_table: pd.DataFrame
    rare_group_summary: pd.DataFrame
    stability_audit: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{context} is missing columns: {missing}")


def _validate_config(config: FunctionalPatternConfig) -> None:
    if config.smooth_half_window_cycles <= 0:
        raise ValueError("smooth_half_window_cycles must be positive")
    if config.smooth_min_points < 3:
        raise ValueError("smooth_min_points must be at least 3")
    if config.smooth_robust_iterations < 1:
        raise ValueError("smooth_robust_iterations must be positive")
    if config.smooth_huber_c <= 0:
        raise ValueError("smooth_huber_c must be positive")
    if not 0.5 <= config.landmark_sustain_fraction <= 1.0:
        raise ValueError("landmark_sustain_fraction must be in [0.5, 1]")
    if config.landmark_min_points < 1:
        raise ValueError("landmark_min_points must be positive")
    if not math.isclose(config.shape_landmark_soh, 0.90, abs_tol=1e-12):
        raise ValueError("this implementation requires shape_landmark_soh=0.90")
    if min(config.shape_grid_points, config.absolute_grid_points) < 16:
        raise ValueError("functional grids must have at least 16 points")
    if not 0 < config.shape_smooth_half_window_fraction < 0.25:
        raise ValueError("shape_smooth_half_window_fraction must be in (0, 0.25)")
    if config.shape_smooth_min_points < 3:
        raise ValueError("shape_smooth_min_points must be at least 3")
    if config.shape_max_internal_gap_cycles <= 0:
        raise ValueError("shape_max_internal_gap_cycles must be positive")
    if not 0 < config.shape_max_internal_gap_fraction < 1:
        raise ValueError("shape_max_internal_gap_fraction must be in (0, 1)")
    if min(
        config.shape_level_scale_floor,
        config.shape_derivative_scale_floor,
        config.absolute_level_scale_floor,
        config.absolute_derivative_scale_floor,
    ) <= 0:
        raise ValueError("functional physical scale floors must be positive")
    if not 0.5 <= config.absolute_min_coverage <= 1.0:
        raise ValueError("absolute_min_coverage must be in [0.5, 1]")
    if config.absolute_min_batch_cells < 4:
        raise ValueError("absolute_min_batch_cells must be at least 4")
    if not 0.5 <= config.fpca_variance_fraction < 1.0:
        raise ValueError("fpca_variance_fraction must be in [0.5, 1)")
    if not 1 <= config.fpca_min_components <= config.fpca_max_components:
        raise ValueError("invalid FPCA component bounds")
    if (
        config.knn_neighbors < 1
        or config.group_mutual_neighbors < 1
        or config.lifetime_group_mutual_neighbors < 1
    ):
        raise ValueError("neighbor counts must be positive")
    if not math.isclose(
        config.level_distance_weight + config.derivative_distance_weight,
        1.0,
        rel_tol=1e-8,
        abs_tol=1e-8,
    ):
        raise ValueError("level and derivative distance weights must sum to one")
    if config.rarity_cutoff <= 0 or config.strong_rarity_cutoff < config.rarity_cutoff:
        raise ValueError("invalid rarity cutoffs")
    if config.individual_method_consensus not in (1, 2, 3):
        raise ValueError("individual_method_consensus must be 1, 2, or 3")
    if config.minimum_adjusted_curve_rms <= 0:
        raise ValueError("minimum_adjusted_curve_rms must be positive")
    if not 0 < config.required_selection_frequency <= 1:
        raise ValueError("required_selection_frequency must be in (0, 1]")
    if config.stability_repeats < 1:
        raise ValueError("stability_repeats must be positive")
    if not 0.5 <= config.stability_grid_fraction <= 1.0:
        raise ValueError("stability_grid_fraction must be in [0.5, 1]")
    if not 0 < config.group_edge_stability <= 1:
        raise ValueError("group_edge_stability must be in (0, 1]")
    if config.group_min_size < 2 or config.group_max_size < config.group_min_size:
        raise ValueError("invalid group size bounds")
    if not 0 < config.group_max_fraction < 1:
        raise ValueError("group_max_fraction must be in (0, 1)")
    if config.group_center_rarity_cutoff <= 0 or config.group_center_min_effect <= 0:
        raise ValueError("group central-separation cutoffs must be positive")
    if not 0 < config.group_min_edge_density <= 1:
        raise ValueError("group_min_edge_density must be in (0, 1]")
    if config.group_min_member_degree < 1:
        raise ValueError("group_min_member_degree must be positive")
    if not 1 <= config.lifetime_group_min_observed_landmarks <= len(
        LANDMARK_THRESHOLDS
    ):
        raise ValueError("invalid lifetime_group_min_observed_landmarks")
    if not 1 <= config.lifetime_group_min_common_landmarks <= len(
        LANDMARK_THRESHOLDS
    ):
        raise ValueError("invalid lifetime_group_min_common_landmarks")
    if not 0.5 <= config.lifetime_group_landmark_fraction <= 1.0:
        raise ValueError("lifetime_group_landmark_fraction must be in [0.5, 1]")
    if config.lifetime_log_duration_scale_floor <= 0:
        raise ValueError("lifetime_log_duration_scale_floor must be positive")
    if (
        config.lifetime_group_min_size < 2
        or config.lifetime_group_max_size < config.lifetime_group_min_size
    ):
        raise ValueError("invalid lifetime group size bounds")
    if not 0 < config.lifetime_group_max_fraction < 1:
        raise ValueError("lifetime_group_max_fraction must be in (0, 1)")
    if not 0 < config.lifetime_group_edge_stability <= 1:
        raise ValueError("lifetime_group_edge_stability must be in (0, 1]")
    if not 0 < config.lifetime_group_min_pair_coverage <= 1:
        raise ValueError("lifetime_group_min_pair_coverage must be in (0, 1]")
    if min(
        config.lifetime_group_separation_ratio,
        config.lifetime_group_center_rarity_cutoff,
    ) <= 0:
        raise ValueError("lifetime group separation cutoffs must be positive")
    if not 0 < config.lifetime_group_acceleration_ratio < 1:
        raise ValueError("lifetime_group_acceleration_ratio must be in (0, 1)")
    if config.lifetime_group_pair_duration_ratio <= 1:
        raise ValueError("lifetime_group_pair_duration_ratio must exceed one")
    if config.lifetime_peer_count < 1:
        raise ValueError("lifetime_peer_count must be positive")
    if min(
        config.transient_mad_effect_floor,
        config.transient_q95_effect_floor,
        config.transient_max_spike_effect_floor,
        config.transient_recovery_effect_floor,
        config.transient_roughness_effect_floor,
    ) <= 0:
        raise ValueError("transient physical effect floors must be positive")


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return 0.0, 1.0
    center = float(np.median(array))
    scale = float(1.4826 * np.median(np.abs(array - center)))
    if not np.isfinite(scale) or scale <= 1e-12:
        q25, q75 = np.quantile(array, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(array))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def _robust_z(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    center, scale = _robust_location_scale(array)
    return (array - center) / scale


def _positive_rarity(values: np.ndarray) -> np.ndarray:
    """Return non-negative robust distance above the typical score center."""

    return np.maximum(_robust_z(np.asarray(values, dtype=float)), 0.0)


def _isotonic_increasing(values: np.ndarray) -> np.ndarray:
    block_values: list[float] = []
    block_weights: list[int] = []
    for value in np.asarray(values, dtype=float):
        block_values.append(float(value))
        block_weights.append(1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            weight = block_weights[-2] + block_weights[-1]
            pooled = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / weight
            block_values[-2:] = [float(pooled)]
            block_weights[-2:] = [int(weight)]
    return np.concatenate(
        [np.full(weight, value) for value, weight in zip(block_values, block_weights)]
    )


def _nonincreasing_fit(values: np.ndarray) -> np.ndarray:
    return -_isotonic_increasing(-np.asarray(values, dtype=float))


def _weighted_local_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Closed-form weighted intercept/slope for x centered at the target."""

    local_x = np.asarray(x, dtype=float)
    local_y = np.asarray(y, dtype=float)
    weight = np.asarray(weights, dtype=float)
    sw = float(np.sum(weight))
    sx = float(np.sum(weight * local_x))
    sxx = float(np.sum(weight * local_x * local_x))
    sy = float(np.sum(weight * local_y))
    sxy = float(np.sum(weight * local_x * local_y))
    denominator = sw * sxx - sx * sx
    if sw <= 1e-12:
        return float(np.median(local_y)), 0.0
    if abs(denominator) <= 1e-12:
        return sy / sw, 0.0
    slope = (sw * sxy - sx * sy) / denominator
    intercept = (sy - slope * sx) / sw
    return float(intercept), float(slope)


def _robust_cycle_smooth(
    cycles: np.ndarray,
    soh: np.ndarray,
    *,
    config: FunctionalPatternConfig,
    half_window_override: float | None = None,
    min_points_override: int | None = None,
) -> np.ndarray:
    """Robust local-linear smoothing with a window measured in cycle units."""

    x = np.asarray(cycles, dtype=float)
    y = np.asarray(soh, dtype=float)
    output = np.empty(len(x), dtype=float)
    minimum = min(
        int(min_points_override or config.smooth_min_points),
        len(x),
    )
    half_window = float(
        config.smooth_half_window_cycles
        if half_window_override is None
        else half_window_override
    )
    for index, center in enumerate(x):
        left = int(np.searchsorted(x, center - half_window, side="left"))
        right = int(np.searchsorted(x, center + half_window, side="right"))
        if right - left < minimum:
            left = max(0, index - minimum // 2)
            right = min(len(x), left + minimum)
            left = max(0, right - minimum)
        local_x = x[left:right] - center
        local_y = y[left:right]
        radius = max(float(np.max(np.abs(local_x))), half_window, 1.0)
        relative_distance = np.minimum(np.abs(local_x) / radius, 1.0)
        distance_weights = (1.0 - relative_distance**3) ** 3
        robust_weights = np.ones(len(local_x), dtype=float)
        prediction = float(np.median(local_y))
        for _ in range(config.smooth_robust_iterations):
            weights = np.maximum(distance_weights * robust_weights, 1e-10)
            prediction, slope = _weighted_local_fit(local_x, local_y, weights)
            # Fit residuals around the current local line, not only its center.
            residual = local_y - (prediction + slope * local_x)
            _, scale = _robust_location_scale(residual)
            standardized = np.abs(residual) / max(config.smooth_huber_c * scale, 1e-10)
            robust_weights = np.ones_like(standardized)
            large = standardized > 1.0
            robust_weights[large] = 1.0 / standardized[large]
        output[index] = prediction
    return output


def _sustained_threshold_crossing(
    cycles: np.ndarray,
    trend: np.ndarray,
    threshold: float,
    *,
    config: FunctionalPatternConfig,
) -> tuple[float, bool, str]:
    """Find a sustained threshold crossing and keep censoring direction explicit.

    When the first available trend value is already at or below ``threshold``,
    the true crossing happened before observation began.  Recording the first
    cycle as an observed event would manufacture an extremely short lifetime,
    so such a landmark is returned as left-censored instead.
    """

    x = np.asarray(cycles, dtype=float)
    y = np.asarray(trend, dtype=float)
    if y[0] <= float(threshold):
        return float("nan"), True, "left_censored_at_observation_start"
    for index in np.flatnonzero(y <= float(threshold)):
        end_cycle = x[index] + float(config.landmark_sustain_cycles)
        followup_index = int(np.searchsorted(x, end_cycle, side="left"))
        right = min(len(x), followup_index + 1)
        window = y[index:right]
        has_span = right > index and x[right - 1] >= end_cycle
        if (
            has_span
            and len(window) >= config.landmark_min_points
            and np.mean(window <= float(threshold)) >= config.landmark_sustain_fraction
        ):
            if index > 0 and y[index - 1] > threshold and y[index] != y[index - 1]:
                fraction = (y[index - 1] - threshold) / (y[index - 1] - y[index])
                crossing = x[index - 1] + np.clip(fraction, 0.0, 1.0) * (
                    x[index] - x[index - 1]
                )
            else:
                crossing = x[index]
            return float(crossing), False, "observed_sustained_crossing"
    below = bool(np.any(y <= float(threshold)))
    reason = "insufficient_followup_after_crossing" if below else "not_reached"
    return float("nan"), True, reason


def _preprocess_curves(
    raw_curves: pd.DataFrame,
    config: FunctionalPatternConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["battery_id", "cell_id", "batch_id", "cycle", "soh"]
    _require_columns(raw_curves, required, "raw SOH curves")
    duplicated = raw_curves.duplicated(["battery_id", "cycle"], keep=False)
    if duplicated.any():
        raise ValueError("raw_curves must contain one row per battery/cycle")
    processed_parts: list[pd.DataFrame] = []
    landmark_rows: list[dict[str, object]] = []
    for battery_id, group in raw_curves.groupby("battery_id", sort=True):
        group = group.sort_values("cycle").copy()
        metadata = group[["cell_id", "batch_id"]].drop_duplicates()
        if len(metadata) != 1:
            raise ValueError(f"{battery_id} maps to multiple cell or batch values")
        cycles = pd.to_numeric(group["cycle"], errors="coerce").to_numpy(dtype=float)
        soh = pd.to_numeric(group["soh"], errors="coerce").to_numpy(dtype=float)
        if len(cycles) < max(12, config.smooth_min_points):
            raise ValueError(f"{battery_id} has too few SOH points")
        if not np.isfinite(cycles).all() or not np.isfinite(soh).all():
            raise ValueError(f"{battery_id} contains non-finite cycle/SOH values")
        if not np.all(np.diff(cycles) > 0):
            raise ValueError(f"{battery_id} cycles are not strictly increasing")
        unconstrained = _robust_cycle_smooth(cycles, soh, config=config)
        monotone = _nonincreasing_fit(unconstrained)
        residual = soh - unconstrained
        monotone_residual = soh - monotone
        residual_center, residual_scale = _robust_location_scale(residual)
        despike_threshold = max(
            config.transient_min_spike,
            4.0 * residual_scale,
        )
        despike_mask = np.abs(residual - residual_center) > despike_threshold
        despiked_soh = np.where(despike_mask, unconstrained, soh)
        metadata_row = metadata.iloc[0]
        row: dict[str, object] = {
            "battery_id": str(battery_id),
            "cell_id": str(metadata_row["cell_id"]),
            "batch_id": str(metadata_row["batch_id"]),
            "n_points": int(len(group)),
            "cycle_start": float(cycles[0]),
            "cycle_end": float(cycles[-1]),
            "observed_span_cycles": float(cycles[-1] - cycles[0]),
            "initial_trend_soh": float(unconstrained[0]),
            "final_trend_soh": float(unconstrained[-1]),
            "total_trend_drop": float(unconstrained[0] - unconstrained[-1]),
            "cycle_gap_fraction": float(np.mean(np.diff(cycles) > 1.0)),
            "max_cycle_gap": float(np.max(np.diff(cycles))),
        }
        for threshold in LANDMARK_THRESHOLDS:
            crossing, censored, reason = _sustained_threshold_crossing(
                cycles, unconstrained, threshold, config=config
            )
            (
                monotone_crossing,
                monotone_censored,
                monotone_reason,
            ) = _sustained_threshold_crossing(cycles, monotone, threshold, config=config)
            label = f"t{int(round(threshold * 100))}"
            censoring = (
                "observed"
                if not censored
                else "left"
                if reason == "left_censored_at_observation_start"
                else "right"
            )
            monotone_censoring = (
                "observed"
                if not monotone_censored
                else "left"
                if monotone_reason == "left_censored_at_observation_start"
                else "right"
            )
            row[label] = crossing
            row[f"{label}_censored"] = bool(censored)
            row[f"{label}_censoring"] = censoring
            row[f"{label}_left_censored"] = censoring == "left"
            row[f"{label}_right_censored"] = censoring == "right"
            row[f"{label}_status"] = reason
            row[f"{label}_monotone"] = monotone_crossing
            row[f"{label}_monotone_censored"] = bool(monotone_censored)
            row[f"{label}_monotone_censoring"] = monotone_censoring
            row[f"{label}_monotone_status"] = monotone_reason
            row[f"{label}_smoother_difference"] = (
                float(abs(crossing - monotone_crossing))
                if np.isfinite(crossing) and np.isfinite(monotone_crossing)
                else float("nan")
            )
        shape_t90, shape_t90_censored, shape_t90_status = _sustained_threshold_crossing(
            cycles,
            despiked_soh,
            config.shape_landmark_soh,
            config=config,
        )
        row["shape_t90"] = shape_t90
        row["shape_t90_censored"] = bool(shape_t90_censored)
        row["shape_t90_censoring"] = (
            "observed"
            if not shape_t90_censored
            else "left"
            if shape_t90_status == "left_censored_at_observation_start"
            else "right"
        )
        row["shape_t90_status"] = shape_t90_status
        transient_center, transient_scale = _robust_location_scale(residual)
        centered_residual = residual - transient_center
        down_spikes = np.maximum(-centered_residual, 0.0)
        recoveries = np.maximum(np.diff(soh), 0.0)
        second_difference = np.diff(soh, n=2)
        row.update(
            {
                "transient_mad": float(transient_scale),
                "down_spike_q95": float(np.quantile(down_spikes, 0.95)),
                "max_down_spike": float(np.max(down_spikes)),
                "large_down_spike_count": int(
                    np.sum(down_spikes > max(config.transient_min_spike, 3 * transient_scale))
                ),
                "recovery_total_per_100_cycles": float(
                    100.0 * np.sum(recoveries) / max(cycles[-1] - cycles[0], 1.0)
                ),
                "roughness_mad": float(
                    _robust_location_scale(second_difference)[1]
                    if len(second_difference)
                    else 0.0
                ),
                "monotone_sensitivity_rms": float(
                    np.sqrt(np.mean((unconstrained - monotone) ** 2))
                ),
            }
        )
        landmark_rows.append(row)
        processed = group[required].copy()
        processed["robust_unconstrained_soh"] = unconstrained
        processed["robust_monotone_soh"] = monotone
        processed["persistent_trend_soh"] = unconstrained
        processed["despiked_soh"] = despiked_soh
        processed["is_transient_despiked"] = despike_mask
        processed["transient_residual"] = residual
        processed["monotone_residual"] = monotone_residual
        processed["trend_sensitivity_gap"] = unconstrained - monotone
        processed_parts.append(processed)
    processed_curves = pd.concat(processed_parts, ignore_index=True).sort_values(
        ["battery_id", "cycle"]
    ).reset_index(drop=True)
    landmark_table = pd.DataFrame(landmark_rows).sort_values("battery_id").reset_index(
        drop=True
    )
    return processed_curves, landmark_table


def _pointwise_robust_standardize(
    matrix: np.ndarray,
    *,
    minimum_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    global_scale = _robust_location_scale((values - center).ravel())[1]
    scale = np.where(np.isfinite(scale) & (scale > 1e-10), scale, global_scale)
    scale = np.maximum(scale, float(minimum_scale))
    scale = np.where(scale > 1e-10, scale, 1.0)
    standardized = np.clip((values - center) / scale, -12.0, 12.0)
    return standardized, center, scale


def _representation_rows(
    keys: pd.DataFrame,
    *,
    view: str,
    coordinates: np.ndarray,
    level: np.ndarray,
    derivative: np.ndarray,
    adjusted_level: np.ndarray,
    adjusted_derivative: np.ndarray,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for row_index, key_row in keys.reset_index(drop=True).iterrows():
        part = pd.DataFrame(
            {
                "battery_id": str(key_row["battery_id"]),
                "cell_id": str(key_row["cell_id"]),
                "batch_id": str(key_row["batch_id"]),
                "view": str(view),
                "grid_index": np.arange(len(coordinates), dtype=int),
                "coordinate": np.asarray(coordinates, dtype=float),
                "level_value": level[row_index],
                "derivative_value": derivative[row_index],
                "adjusted_level": adjusted_level[row_index],
                "adjusted_derivative": adjusted_derivative[row_index],
            }
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(
            columns=[
                "battery_id",
                "cell_id",
                "batch_id",
                "view",
                "grid_index",
                "coordinate",
                "level_value",
                "derivative_value",
                "adjusted_level",
                "adjusted_derivative",
            ]
        )
    return pd.concat(parts, ignore_index=True)


def _build_absolute_representation(
    processed_curves: pd.DataFrame,
    landmark_table: pd.DataFrame,
    config: FunctionalPatternConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build batch-specific common-support absolute-cycle representations."""

    parts: list[pd.DataFrame] = []
    availability_rows: list[dict[str, object]] = []
    eligibility = pd.Series(False, index=landmark_table["battery_id"].astype(str), dtype=bool)
    lookup = {
        str(battery_id): group.sort_values("cycle")
        for battery_id, group in processed_curves.groupby("battery_id", sort=True)
    }
    for batch_id, batch_meta in landmark_table.groupby("batch_id", sort=True):
        batch_meta = batch_meta.sort_values("battery_id").reset_index(drop=True)
        n_batch = len(batch_meta)
        required = max(
            int(config.absolute_min_batch_cells),
            int(math.ceil(config.absolute_min_coverage * n_batch)),
        )
        if required > n_batch:
            availability_rows.append(
                {
                    "batch_id": str(batch_id),
                    "n_batch_cells": int(n_batch),
                    "required_coverage_cells": int(required),
                    "eligible_cells": 0,
                    "absolute_cycle_start": np.nan,
                    "absolute_cycle_end": np.nan,
                    "absolute_view_available": False,
                    "reason": "batch_smaller_than_required_minimum",
                }
            )
            continue
        starts = batch_meta["cycle_start"].to_numpy(dtype=float)
        ends = np.sort(batch_meta["cycle_end"].to_numpy(dtype=float))
        common_start = float(np.max(starts))
        # The order statistic gives the longest support covered by `required` cells.
        common_end = float(ends[n_batch - required])
        eligible_meta = batch_meta.loc[
            (batch_meta["cycle_start"] <= common_start)
            & (batch_meta["cycle_end"] >= common_end)
        ].copy()
        available = len(eligible_meta) >= required and common_end > common_start
        availability_rows.append(
            {
                "batch_id": str(batch_id),
                "n_batch_cells": int(n_batch),
                "required_coverage_cells": int(required),
                "eligible_cells": int(len(eligible_meta) if available else 0),
                "absolute_cycle_start": common_start if available else np.nan,
                "absolute_cycle_end": common_end if available else np.nan,
                "absolute_view_available": bool(available),
                "reason": "available" if available else "insufficient_common_support",
            }
        )
        if not available:
            continue
        coordinates = np.linspace(common_start, common_end, config.absolute_grid_points)
        level_rows = []
        for battery_id in eligible_meta["battery_id"].astype(str):
            curve = lookup[battery_id]
            level_rows.append(
                np.interp(
                    coordinates,
                    curve["cycle"].to_numpy(dtype=float),
                    curve["persistent_trend_soh"].to_numpy(dtype=float),
                )
            )
            eligibility.loc[battery_id] = True
        level = np.asarray(level_rows, dtype=float)
        derivative = np.gradient(level, coordinates, axis=1)
        adjusted_level, _, _ = _pointwise_robust_standardize(
            level, minimum_scale=config.absolute_level_scale_floor
        )
        adjusted_derivative, _, _ = _pointwise_robust_standardize(
            derivative, minimum_scale=config.absolute_derivative_scale_floor
        )
        parts.append(
            _representation_rows(
                eligible_meta[["battery_id", "cell_id", "batch_id"]],
                view="absolute_cycle_within_batch",
                coordinates=coordinates,
                level=level,
                derivative=derivative,
                adjusted_level=adjusted_level,
                adjusted_derivative=adjusted_derivative,
            )
        )
    representation = (
        pd.concat(parts, ignore_index=True)
        if parts
        else _representation_rows(
            pd.DataFrame(columns=["battery_id", "cell_id", "batch_id"]),
            view="absolute_cycle_within_batch",
            coordinates=np.asarray([], dtype=float),
            level=np.empty((0, 0)),
            derivative=np.empty((0, 0)),
            adjusted_level=np.empty((0, 0)),
            adjusted_derivative=np.empty((0, 0)),
        )
    )
    return representation, eligibility, pd.DataFrame(availability_rows)


def _build_shape_representation(
    processed_curves: pd.DataFrame,
    landmark_table: pd.DataFrame,
    config: FunctionalPatternConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build sustained-T90 normalized pure-shape representations.

    Horizontal time is divided by sustained T90 and the vertical drop is
    divided by the total trend drop at T90.  Cells without an observed,
    sufficiently supported T90 remain explicitly ineligible; their last
    observed cycle is never substituted for T90.
    """

    eligibility = pd.Series(False, index=landmark_table["battery_id"].astype(str), dtype=bool)
    reasons = pd.Series("", index=eligibility.index, dtype=object)
    phase = np.linspace(0.0, 1.0, config.shape_grid_points)
    curve_lookup = {
        str(battery_id): group.sort_values("cycle")
        for battery_id, group in processed_curves.groupby("battery_id", sort=True)
    }
    keys: list[dict[str, str]] = []
    levels: list[np.ndarray] = []
    for _, row in landmark_table.sort_values("battery_id").iterrows():
        battery_id = str(row["battery_id"])
        if bool(row["shape_t90_censored"]) or not np.isfinite(float(row["shape_t90"])):
            reasons.loc[battery_id] = str(row["shape_t90_status"])
            continue
        t90 = float(row["shape_t90"])
        start = float(row["cycle_start"])
        if t90 - start < config.shape_min_span_cycles:
            reasons.loc[battery_id] = "t90_span_too_short"
            continue
        curve = curve_lookup[battery_id]
        cycles = curve["cycle"].to_numpy(dtype=float)
        before_t90 = cycles <= t90
        internal_cycles = np.append(cycles[before_t90], t90)
        maximum_gap = float(np.max(np.diff(internal_cycles))) if len(internal_cycles) > 1 else np.inf
        allowed_gap = min(
            config.shape_max_internal_gap_cycles,
            config.shape_max_internal_gap_fraction * (t90 - start),
        )
        if maximum_gap > allowed_gap:
            reasons.loc[battery_id] = "internal_cycle_gap_too_large_for_shape"
            continue
        # The persistent T90 landmark is measured on the absolute axis, but
        # shape smoothing must have a fixed phase bandwidth.  Reusing a fixed
        # cycle-domain smoother here would distort short-life curves more than
        # long-life curves and leak lifetime into shape rarity.
        despiked = curve["despiked_soh"].to_numpy(dtype=float)
        sample_cycles = start + phase * (t90 - start)
        sampled_soh = np.interp(sample_cycles, cycles, despiked)
        baseline = float(sampled_soh[0])
        total_drop = float(baseline - sampled_soh[-1])
        if total_drop < config.shape_min_total_drop:
            reasons.loc[battery_id] = "vertical_drop_too_small"
            continue
        raw_shape = (baseline - sampled_soh) / total_drop
        smoothed_shape = _robust_cycle_smooth(
            phase,
            raw_shape,
            config=config,
            half_window_override=config.shape_smooth_half_window_fraction,
            min_points_override=config.shape_smooth_min_points,
        )
        shape_span = float(smoothed_shape[-1] - smoothed_shape[0])
        if shape_span <= 1e-8:
            reasons.loc[battery_id] = "phase_smoothed_vertical_drop_too_small"
            continue
        pure_shape = (smoothed_shape - smoothed_shape[0]) / shape_span
        keys.append(
            {
                "battery_id": battery_id,
                "cell_id": str(row["cell_id"]),
                "batch_id": str(row["batch_id"]),
            }
        )
        levels.append(pure_shape)
        eligibility.loc[battery_id] = True
        reasons.loc[battery_id] = "eligible_observed_sustained_t90"
    if not levels:
        empty = _representation_rows(
            pd.DataFrame(columns=["battery_id", "cell_id", "batch_id"]),
            view="sustained_t90_normalized_shape",
            coordinates=np.asarray([], dtype=float),
            level=np.empty((0, 0)),
            derivative=np.empty((0, 0)),
            adjusted_level=np.empty((0, 0)),
            adjusted_derivative=np.empty((0, 0)),
        )
        return empty, eligibility, reasons
    key_frame = pd.DataFrame(keys)
    level = np.asarray(levels, dtype=float)
    derivative = np.gradient(level, phase, axis=1)
    adjusted_level = np.zeros_like(level)
    adjusted_derivative = np.zeros_like(derivative)
    # Remove planned batch shifts while retaining the raw shape columns.
    for _, group in key_frame.groupby("batch_id", sort=True):
        indices = group.index.to_numpy(dtype=int)
        adjusted_level[indices], _, _ = _pointwise_robust_standardize(
            level[indices], minimum_scale=config.shape_level_scale_floor
        )
        adjusted_derivative[indices], _, _ = _pointwise_robust_standardize(
            derivative[indices], minimum_scale=config.shape_derivative_scale_floor
        )
    representation = _representation_rows(
        key_frame,
        view="sustained_t90_normalized_shape",
        coordinates=phase,
        level=level,
        derivative=derivative,
        adjusted_level=adjusted_level,
        adjusted_derivative=adjusted_derivative,
    )
    return representation, eligibility, reasons


def _matrix_from_representation(
    representation: pd.DataFrame,
    value_column: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if representation.empty:
        return (
            pd.DataFrame(columns=["battery_id", "cell_id", "batch_id"]),
            np.empty((0, 0), dtype=float),
        )
    keys = ["battery_id", "cell_id", "batch_id"]
    pivot = representation.pivot_table(
        index=keys,
        columns="grid_index",
        values=value_column,
        aggfunc="first",
        observed=False,
    ).sort_index()
    if pivot.isna().any().any():
        raise RuntimeError(f"{value_column} representation contains incomplete grids")
    return pivot.index.to_frame(index=False), pivot.to_numpy(dtype=float)


def _fpca_metrics(
    matrix: np.ndarray,
    config: FunctionalPatternConfig,
) -> dict[str, np.ndarray | float | int]:
    """Robustly weighted FPCA score/orthogonal distances and rarity ratios."""

    values = np.asarray(matrix, dtype=float)
    n_rows, n_columns = values.shape
    if n_rows < 4 or n_columns < 4:
        nan = np.full(n_rows, np.nan)
        return {
            "score_distance": nan,
            "orthogonal_distance": nan,
            "sd_ratio": nan,
            "od_ratio": nan,
            "rarity": nan,
            "components": 0,
            "sd_threshold": np.nan,
            "od_threshold": np.nan,
        }
    center = np.median(values, axis=0)
    weights = np.ones(n_rows, dtype=float)
    components = np.empty((0, n_columns), dtype=float)
    n_components = config.fpca_min_components
    for _ in range(config.fpca_robust_iterations):
        weights = np.maximum(weights, 1e-6)
        center = np.average(values, axis=0, weights=weights)
        centered = values - center
        weighted = centered * np.sqrt(weights)[:, None]
        try:
            _, singular_values, vt = np.linalg.svd(weighted, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("SVD failed during robust FPCA") from exc
        maximum = min(config.fpca_max_components, n_rows - 2, n_columns)
        minimum = min(config.fpca_min_components, maximum)
        variances = singular_values[:maximum] ** 2
        total = float(np.sum(singular_values**2))
        if total <= 1e-12:
            n_components = minimum
        else:
            cumulative = np.cumsum(variances) / total
            n_components = int(np.searchsorted(cumulative, config.fpca_variance_fraction) + 1)
            n_components = min(max(n_components, minimum), maximum)
        components = vt[:n_components]
        scores = centered @ components.T
        reconstruction = scores @ components
        residual_rms = np.sqrt(np.mean((centered - reconstruction) ** 2, axis=1))
        row_energy = np.sqrt(np.mean(centered**2, axis=1))
        residual_rarity = _positive_rarity(residual_rms)
        energy_rarity = _positive_rarity(row_energy)
        combined = np.maximum(residual_rarity, energy_rarity)
        weights = np.where(
            combined <= config.rarity_cutoff,
            1.0,
            config.rarity_cutoff / np.maximum(combined, 1e-12),
        )
    centered = values - center
    scores = centered @ components.T
    standardized_scores = np.empty_like(scores)
    for component in range(scores.shape[1]):
        component_center, component_scale = _robust_location_scale(scores[:, component])
        standardized_scores[:, component] = (
            scores[:, component] - component_center
        ) / component_scale
    score_distance = np.sqrt(np.sum(standardized_scores**2, axis=1))
    reconstruction = scores @ components
    orthogonal_distance = np.sqrt(np.mean((centered - reconstruction) ** 2, axis=1))
    sd_center, sd_scale = _robust_location_scale(score_distance)
    od_center, od_scale = _robust_location_scale(orthogonal_distance)
    sd_threshold = max(sd_center + config.rarity_cutoff * sd_scale, 1e-12)
    od_threshold = max(od_center + config.rarity_cutoff * od_scale, 1e-12)
    rarity = np.maximum(
        np.maximum((score_distance - sd_center) / sd_scale, 0.0),
        np.maximum((orthogonal_distance - od_center) / od_scale, 0.0),
    )
    return {
        "score_distance": score_distance,
        "orthogonal_distance": orthogonal_distance,
        "sd_ratio": score_distance / sd_threshold,
        "od_ratio": orthogonal_distance / od_threshold,
        "rarity": rarity,
        "components": int(n_components),
        "sd_threshold": float(sd_threshold),
        "od_threshold": float(od_threshold),
    }


def _functional_outlyingness(matrix: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    row_mean = np.mean(values, axis=1)
    magnitude = np.abs(row_mean)
    shape = np.sqrt(np.mean((values - row_mean[:, None]) ** 2, axis=1))
    magnitude_rarity = _positive_rarity(magnitude)
    shape_rarity = _positive_rarity(shape)
    return {
        "magnitude": magnitude,
        "shape": shape,
        "magnitude_rarity": magnitude_rarity,
        "shape_rarity": shape_rarity,
        "rarity": np.maximum(magnitude_rarity, shape_rarity),
    }


def _combined_curve_distance(
    level: np.ndarray,
    derivative: np.ndarray,
    config: FunctionalPatternConfig,
) -> np.ndarray:
    level_values = np.asarray(level, dtype=float)
    derivative_values = np.asarray(derivative, dtype=float)
    level_distance = np.sqrt(
        np.mean(
            (level_values[:, None, :] - level_values[None, :, :]) ** 2,
            axis=2,
        )
    )
    derivative_distance = np.sqrt(
        np.mean(
            (
                derivative_values[:, None, :]
                - derivative_values[None, :, :]
            )
            ** 2,
            axis=2,
        )
    )
    return np.sqrt(
        config.level_distance_weight * level_distance**2
        + config.derivative_distance_weight * derivative_distance**2
    )


def _distance_metrics(
    level: np.ndarray,
    derivative: np.ndarray,
    config: FunctionalPatternConfig,
) -> dict[str, np.ndarray]:
    distance = _combined_curve_distance(level, derivative, config)
    n_rows = len(distance)
    if n_rows < 2:
        nan = np.full(n_rows, np.nan)
        return {"distance": distance, "knn_distance": nan, "rarity": nan}
    masked = distance.copy()
    np.fill_diagonal(masked, np.inf)
    neighbors = min(config.knn_neighbors, n_rows - 1)
    nearest = np.partition(masked, neighbors - 1, axis=1)[:, :neighbors]
    knn_distance = np.mean(nearest, axis=1)
    return {
        "distance": distance,
        "knn_distance": knn_distance,
        "rarity": _positive_rarity(knn_distance),
    }


def _score_matrix(
    level: np.ndarray,
    derivative: np.ndarray,
    config: FunctionalPatternConfig,
) -> dict[str, np.ndarray | float | int]:
    fpca = _fpca_metrics(level, config)
    outlyingness = _functional_outlyingness(level)
    distance = _distance_metrics(level, derivative, config)
    method_rarities = np.column_stack(
        [
            np.asarray(fpca["rarity"], dtype=float),
            np.asarray(outlyingness["rarity"], dtype=float),
            np.asarray(distance["rarity"], dtype=float),
        ]
    )
    practical_effect_rms = np.sqrt(np.mean(np.asarray(level, dtype=float) ** 2, axis=1))
    has_practical_effect = practical_effect_rms >= config.minimum_adjusted_curve_rms
    # A curve can be statistically distinct in an almost noiseless synthetic
    # cohort yet differ by far less than the predeclared physical scale floor.
    # The practical-effect gate prevents that numerical rarity becoming a
    # degradation-pattern candidate.
    method_rarities = np.where(has_practical_effect[:, None], method_rarities, 0.0)
    method_count = np.sum(method_rarities >= config.rarity_cutoff, axis=1)
    ordered = np.sort(method_rarities, axis=1)[:, ::-1]
    consensus_index = min(config.individual_method_consensus - 1, 2)
    pattern_score = ordered[:, consensus_index]
    candidate = method_count >= config.individual_method_consensus
    return {
        **{f"fpca_{key}": value for key, value in fpca.items()},
        **{f"outlying_{key}": value for key, value in outlyingness.items()},
        **{f"distance_{key}": value for key, value in distance.items()},
        "method_count": method_count,
        "pattern_score": pattern_score,
        "candidate": candidate,
        "practical_effect_rms": practical_effect_rms,
        "has_practical_effect": has_practical_effect,
    }


def _score_representation(
    representation: pd.DataFrame,
    *,
    view_prefix: str,
    config: FunctionalPatternConfig,
    stratify_by_batch: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a representation and estimate grid-subsampling stability."""

    output_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    if representation.empty:
        return pd.DataFrame(), pd.DataFrame()
    strata = (
        representation.groupby("batch_id", sort=True)
        if stratify_by_batch
        else [("pooled", representation)]
    )
    rng = np.random.default_rng(config.random_state + (0 if view_prefix == "shape" else 701))
    for stratum, part in strata:
        keys, level = _matrix_from_representation(part, "adjusted_level")
        derivative_keys, derivative = _matrix_from_representation(
            part, "adjusted_derivative"
        )
        if not keys.equals(derivative_keys):
            raise RuntimeError(f"{view_prefix} level/derivative keys do not match")
        if len(keys) < 4:
            continue
        metrics = _score_matrix(level, derivative, config)
        result = keys.copy()
        result[f"{view_prefix}_score_distance"] = metrics["fpca_score_distance"]
        result[f"{view_prefix}_orthogonal_distance"] = metrics[
            "fpca_orthogonal_distance"
        ]
        result[f"{view_prefix}_sd_ratio"] = metrics["fpca_sd_ratio"]
        result[f"{view_prefix}_od_ratio"] = metrics["fpca_od_ratio"]
        result[f"{view_prefix}_fpca_rarity"] = metrics["fpca_rarity"]
        result[f"{view_prefix}_magnitude_outlyingness"] = metrics[
            "outlying_magnitude"
        ]
        result[f"{view_prefix}_shape_outlyingness"] = metrics["outlying_shape"]
        result[f"{view_prefix}_functional_rarity"] = metrics["outlying_rarity"]
        result[f"{view_prefix}_knn_distance"] = metrics["distance_knn_distance"]
        result[f"{view_prefix}_distance_rarity"] = metrics["distance_rarity"]
        result[f"{view_prefix}_method_count"] = metrics["method_count"]
        result[f"{view_prefix}_pattern_score"] = metrics["pattern_score"]
        result[f"{view_prefix}_base_candidate"] = metrics["candidate"]
        result[f"{view_prefix}_practical_effect_rms"] = metrics[
            "practical_effect_rms"
        ]
        result[f"{view_prefix}_has_practical_effect"] = metrics[
            "has_practical_effect"
        ]
        result[f"{view_prefix}_fpca_components"] = int(metrics["fpca_components"])
        result[f"{view_prefix}_sd_threshold"] = float(metrics["fpca_sd_threshold"])
        result[f"{view_prefix}_od_threshold"] = float(metrics["fpca_od_threshold"])

        candidate_matrix = np.zeros((len(keys), config.stability_repeats), dtype=bool)
        grid_count = level.shape[1]
        sampled_count = min(
            grid_count,
            max(8, int(math.ceil(config.stability_grid_fraction * grid_count))),
        )
        for repeat in range(config.stability_repeats):
            columns = np.sort(rng.choice(grid_count, size=sampled_count, replace=False))
            repeat_metrics = _score_matrix(
                level[:, columns], derivative[:, columns], config
            )
            candidate_matrix[:, repeat] = np.asarray(
                repeat_metrics["candidate"], dtype=bool
            )
            for index, battery_id in enumerate(keys["battery_id"].astype(str)):
                audit_rows.append(
                    {
                        "battery_id": battery_id,
                        "cell_id": str(keys.iloc[index]["cell_id"]),
                        "batch_id": str(keys.iloc[index]["batch_id"]),
                        "view": view_prefix,
                        "stratum": str(stratum),
                        "repeat": int(repeat),
                        "sampled_grid_points": int(sampled_count),
                        "fpca_rarity": float(
                            np.asarray(repeat_metrics["fpca_rarity"])[index]
                        ),
                        "functional_rarity": float(
                            np.asarray(repeat_metrics["outlying_rarity"])[index]
                        ),
                        "distance_rarity": float(
                            np.asarray(repeat_metrics["distance_rarity"])[index]
                        ),
                        "method_count": int(
                            np.asarray(repeat_metrics["method_count"])[index]
                        ),
                        "selected": bool(candidate_matrix[index, repeat]),
                    }
                )
        result[f"{view_prefix}_selection_frequency"] = np.mean(
            candidate_matrix, axis=1
        )
        result[f"is_{view_prefix}_candidate"] = (
            result[f"{view_prefix}_base_candidate"]
            & (
                result[f"{view_prefix}_selection_frequency"]
                >= config.required_selection_frequency
            )
        )
        output_rows.append(result)
    return (
        pd.concat(output_rows, ignore_index=True) if output_rows else pd.DataFrame(),
        pd.DataFrame(audit_rows),
    )


def _score_lifetime(
    landmark_table: pd.DataFrame,
    config: FunctionalPatternConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score landmark timing while distinguishing left and right censoring."""

    frame = landmark_table.sort_values("battery_id").reset_index(drop=True).copy()
    labels = [f"t{int(round(value * 100))}" for value in LANDMARK_THRESHOLDS]

    def score_with_reference(reference_indices: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        rarity_matrix = np.zeros((len(frame), len(labels)), dtype=float)
        evidence_matrix = np.zeros_like(rarity_matrix, dtype=bool)
        for batch_id, group in frame.groupby("batch_id", sort=True):
            targets = group.index.to_numpy(dtype=int)
            references = reference_indices[str(batch_id)]
            for column_index, label in enumerate(labels):
                censoring_column = f"{label}_censoring"
                ref_observed = references[
                    frame.loc[references, censoring_column]
                    .astype(str)
                    .eq("observed")
                    .to_numpy(dtype=bool)
                    & np.isfinite(frame.loc[references, label].to_numpy(dtype=float))
                ]
                if len(ref_observed) < 3:
                    continue
                ref_duration = (
                    frame.loc[ref_observed, label].to_numpy(dtype=float)
                    - frame.loc[ref_observed, "cycle_start"].to_numpy(dtype=float)
                )
                center, scale = _robust_location_scale(np.log1p(ref_duration))
                for target in targets:
                    censoring = str(frame.loc[target, censoring_column])
                    if censoring == "observed":
                        duration = float(frame.loc[target, label] - frame.loc[target, "cycle_start"])
                        rarity_matrix[target, column_index] = abs(
                            (math.log1p(max(duration, 0.0)) - center) / scale
                        )
                        evidence_matrix[target, column_index] = True
                    elif censoring == "right":
                        # A right-censored observation supplies a lower bound and
                        # can support unusually long, never unusually short, life.
                        lower_bound = float(
                            frame.loc[target, "cycle_end"]
                            - frame.loc[target, "cycle_start"]
                        )
                        rarity_matrix[target, column_index] = max(
                            (math.log1p(max(lower_bound, 0.0)) - center) / scale,
                            0.0,
                        )
                        evidence_matrix[target, column_index] = (
                            rarity_matrix[target, column_index] > 0
                        )
                    elif censoring == "left":
                        # The threshold was already crossed before measurement.
                        # Its unknown event time is neither an observed short life
                        # nor a right-censored lower bound, so it supplies no
                        # lifetime-speed evidence.
                        continue
                    else:
                        raise ValueError(
                            f"Unknown {censoring_column}={censoring!r} for "
                            f"battery_id={frame.loc[target, 'battery_id']!r}"
                        )
        ordered = np.sort(rarity_matrix, axis=1)[:, ::-1]
        score = ordered[:, 1]  # two-landmark consensus; zero when unavailable
        count = np.sum(rarity_matrix >= config.rarity_cutoff, axis=1)
        strong = np.max(rarity_matrix, axis=1) >= config.strong_rarity_cutoff
        candidate = (count >= 2) | strong
        return score, candidate

    full_references = {
        str(batch_id): group.index.to_numpy(dtype=int)
        for batch_id, group in frame.groupby("batch_id", sort=True)
    }
    score, base_candidate = score_with_reference(full_references)
    rng = np.random.default_rng(config.random_state + 1_409)
    selected = np.zeros((len(frame), config.stability_repeats), dtype=bool)
    audit_rows: list[dict[str, object]] = []
    for repeat in range(config.stability_repeats):
        references = {
            str(batch_id): rng.choice(
                group.index.to_numpy(dtype=int), size=len(group), replace=True
            )
            for batch_id, group in frame.groupby("batch_id", sort=True)
        }
        repeat_score, repeat_selected = score_with_reference(references)
        selected[:, repeat] = repeat_selected
        for index, row in frame.iterrows():
            audit_rows.append(
                {
                    "battery_id": str(row["battery_id"]),
                    "cell_id": str(row["cell_id"]),
                    "batch_id": str(row["batch_id"]),
                    "view": "lifetime",
                    "stratum": str(row["batch_id"]),
                    "repeat": int(repeat),
                    "sampled_grid_points": 4,
                    "fpca_rarity": np.nan,
                    "functional_rarity": np.nan,
                    "distance_rarity": float(repeat_score[index]),
                    "method_count": int(repeat_selected[index]),
                    "selected": bool(repeat_selected[index]),
                }
            )
    result = frame[["battery_id", "cell_id", "batch_id"]].copy()
    result["lifetime_score"] = score
    result["lifetime_base_candidate"] = base_candidate
    result["lifetime_selection_frequency"] = np.mean(selected, axis=1)
    result["is_lifetime_candidate"] = result["lifetime_base_candidate"] & (
        result["lifetime_selection_frequency"] >= config.required_selection_frequency
    )
    censoring_columns = [f"{label}_censoring" for label in labels]
    censoring_values = frame[censoring_columns].astype(str)
    result["lifetime_observed_landmarks"] = censoring_values.eq("observed").sum(axis=1)
    result["lifetime_left_censored_landmarks"] = censoring_values.eq("left").sum(axis=1)
    result["lifetime_right_censored_landmarks"] = censoring_values.eq("right").sum(axis=1)
    result["lifetime_evidence_note"] = (
        "observed_landmarks_are_two_sided;right_censored_only_support_long_life;"
        "left_censored_supply_no_lifetime_speed_evidence"
    )
    return result, pd.DataFrame(audit_rows)


def _score_transient(
    landmark_table: pd.DataFrame,
    config: FunctionalPatternConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        "transient_mad",
        "down_spike_q95",
        "max_down_spike",
        "recovery_total_per_100_cycles",
        "roughness_mad",
    ]
    effect_floors = np.asarray(
        [
            config.transient_mad_effect_floor,
            config.transient_q95_effect_floor,
            config.transient_max_spike_effect_floor,
            config.transient_recovery_effect_floor,
            config.transient_roughness_effect_floor,
        ],
        dtype=float,
    )
    frame = landmark_table.sort_values("battery_id").reset_index(drop=True).copy()

    def score_with_reference(reference_indices: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        rarity = np.zeros((len(frame), len(features)), dtype=float)
        for batch_id, group in frame.groupby("batch_id", sort=True):
            targets = group.index.to_numpy(dtype=int)
            references = reference_indices[str(batch_id)]
            for column_index, column in enumerate(features):
                center, scale = _robust_location_scale(
                    frame.loc[references, column].to_numpy(dtype=float)
                )
                rarity[targets, column_index] = np.maximum(
                    (
                        frame.loc[targets, column].to_numpy(dtype=float) - center
                    )
                    / scale,
                    0.0,
                )
                has_physical_effect = (
                    frame.loc[targets, column].to_numpy(dtype=float)
                    >= effect_floors[column_index]
                )
                rarity[targets, column_index] = np.where(
                    has_physical_effect,
                    rarity[targets, column_index],
                    0.0,
                )
        ordered = np.sort(rarity, axis=1)[:, ::-1]
        score = ordered[:, 1]
        count = np.sum(rarity >= config.rarity_cutoff, axis=1)
        strong = np.max(rarity, axis=1) >= config.strong_rarity_cutoff
        return score, (count >= 2) | strong

    full_references = {
        str(batch_id): group.index.to_numpy(dtype=int)
        for batch_id, group in frame.groupby("batch_id", sort=True)
    }
    score, base_candidate = score_with_reference(full_references)
    rng = np.random.default_rng(config.random_state + 2_809)
    selected = np.zeros((len(frame), config.stability_repeats), dtype=bool)
    audit_rows: list[dict[str, object]] = []
    for repeat in range(config.stability_repeats):
        references = {
            str(batch_id): rng.choice(
                group.index.to_numpy(dtype=int), size=len(group), replace=True
            )
            for batch_id, group in frame.groupby("batch_id", sort=True)
        }
        repeat_score, repeat_selected = score_with_reference(references)
        selected[:, repeat] = repeat_selected
        for index, row in frame.iterrows():
            audit_rows.append(
                {
                    "battery_id": str(row["battery_id"]),
                    "cell_id": str(row["cell_id"]),
                    "batch_id": str(row["batch_id"]),
                    "view": "transient",
                    "stratum": str(row["batch_id"]),
                    "repeat": int(repeat),
                    "sampled_grid_points": len(features),
                    "fpca_rarity": np.nan,
                    "functional_rarity": np.nan,
                    "distance_rarity": float(repeat_score[index]),
                    "method_count": int(repeat_selected[index]),
                    "selected": bool(repeat_selected[index]),
                }
            )
    result = frame[["battery_id", "cell_id", "batch_id"]].copy()
    result["transient_score"] = score
    result["transient_base_candidate"] = base_candidate
    result["transient_selection_frequency"] = np.mean(selected, axis=1)
    result["is_transient_candidate"] = result["transient_base_candidate"] & (
        result["transient_selection_frequency"]
        >= config.required_selection_frequency
    )
    return result, pd.DataFrame(audit_rows)


def _connected_components(adjacency: np.ndarray) -> list[np.ndarray]:
    matrix = np.asarray(adjacency, dtype=bool)
    visited = np.zeros(len(matrix), dtype=bool)
    components: list[np.ndarray] = []
    for start in range(len(matrix)):
        if visited[start] or not np.any(matrix[start]):
            continue
        stack = [start]
        visited[start] = True
        members = []
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbor in np.flatnonzero(matrix[node]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        components.append(np.asarray(sorted(members), dtype=int))
    return components


def _nearest_peer_map(
    representation: pd.DataFrame,
    config: FunctionalPatternConfig,
) -> dict[str, str]:
    peers: dict[str, str] = {}
    if representation.empty:
        return peers
    for _, part in representation.groupby("batch_id", sort=True):
        keys, level = _matrix_from_representation(part, "adjusted_level")
        derivative_keys, derivative = _matrix_from_representation(
            part, "adjusted_derivative"
        )
        if not keys.equals(derivative_keys) or len(keys) < 2:
            continue
        distance = _combined_curve_distance(level, derivative, config)
        np.fill_diagonal(distance, np.inf)
        count = min(config.knn_neighbors, len(keys) - 1)
        battery_ids = keys["battery_id"].astype(str).to_numpy()
        for index, battery_id in enumerate(battery_ids):
            nearest = np.argsort(distance[index])[:count]
            peers[battery_id] = ",".join(battery_ids[nearest])
    return peers


def _discover_rare_groups_for_view(
    representation: pd.DataFrame,
    *,
    view: str,
    config: FunctionalPatternConfig,
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find cohesive small groups from stable mutual-neighbor relations."""

    group_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    pair_audit_rows: list[dict[str, object]] = []
    if representation.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(config.random_state + seed_offset)
    group_number = 0
    for batch_id, part in representation.groupby("batch_id", sort=True):
        keys, level = _matrix_from_representation(part, "adjusted_level")
        derivative_keys, derivative = _matrix_from_representation(
            part, "adjusted_derivative"
        )
        if not keys.equals(derivative_keys) or len(keys) < 4:
            continue
        n_rows, grid_count = level.shape
        neighbor_count = min(config.group_mutual_neighbors, n_rows - 1)
        sampled_count = min(
            grid_count,
            max(8, int(math.ceil(config.stability_grid_fraction * grid_count))),
        )
        edge_counts = np.zeros((n_rows, n_rows), dtype=float)
        for _ in range(config.stability_repeats):
            columns = np.sort(rng.choice(grid_count, size=sampled_count, replace=False))
            distance = _combined_curve_distance(
                level[:, columns], derivative[:, columns], config
            )
            np.fill_diagonal(distance, np.inf)
            nearest = np.argsort(distance, axis=1)[:, :neighbor_count]
            directed = np.zeros((n_rows, n_rows), dtype=bool)
            directed[np.arange(n_rows)[:, None], nearest] = True
            mutual = directed & directed.T
            edge_counts += mutual
        edge_frequency = edge_counts / config.stability_repeats
        stable_adjacency = edge_frequency >= config.group_edge_stability
        np.fill_diagonal(stable_adjacency, False)
        full_distance = _combined_curve_distance(level, derivative, config)
        battery_ids = keys["battery_id"].astype(str).to_numpy()
        # Cohesion alone labels ordinary local neighborhoods.  The batch medoid
        # supplies a robust central reference so a group must also be far from
        # the dominant functional pattern.
        medoid_index = int(np.argmin(np.median(full_distance, axis=1)))
        cell_center_distance = full_distance[:, medoid_index]
        center_distance_location, center_distance_scale = _robust_location_scale(
            cell_center_distance
        )
        for left in range(n_rows):
            for right in range(left + 1, n_rows):
                if edge_frequency[left, right] > 0:
                    pair_audit_rows.append(
                        {
                            "battery_id": battery_ids[left],
                            "peer_battery_id": battery_ids[right],
                            "cell_id": str(keys.iloc[left]["cell_id"]),
                            "batch_id": str(batch_id),
                            "view": f"{view}_rare_group_pair",
                            "stratum": str(batch_id),
                            "repeat": -1,
                            "sampled_grid_points": int(sampled_count),
                            "pair_edge_frequency": float(edge_frequency[left, right]),
                            "selected": bool(stable_adjacency[left, right]),
                        }
                    )
        for component in _connected_components(stable_adjacency):
            size = len(component)
            if size < config.group_min_size:
                continue
            maximum_size = min(
                config.group_max_size,
                max(
                    config.group_min_size,
                    int(math.floor(config.group_max_fraction * n_rows)),
                ),
            )
            inside_mask = np.zeros(n_rows, dtype=bool)
            inside_mask[component] = True
            outside = np.flatnonzero(~inside_mask)
            upper = full_distance[np.ix_(component, component)][
                np.triu_indices(size, k=1)
            ]
            within_distance = float(np.median(upper)) if len(upper) else 0.0
            if len(outside):
                nearest_outside = np.min(
                    full_distance[np.ix_(component, outside)], axis=1
                )
                outside_distance = float(np.median(nearest_outside))
            else:
                outside_distance = 0.0
            separation_ratio = outside_distance / max(within_distance, 1e-12)
            component_edges = edge_frequency[np.ix_(component, component)][
                np.triu_indices(size, k=1)
            ]
            positive_edges = component_edges[component_edges > 0]
            stability = float(np.mean(positive_edges)) if len(positive_edges) else 0.0
            component_adjacency = stable_adjacency[np.ix_(component, component)]
            stable_edge_count = int(np.sum(np.triu(component_adjacency, k=1)))
            possible_edge_count = size * (size - 1) // 2
            edge_density = stable_edge_count / max(possible_edge_count, 1)
            minimum_member_degree = int(np.min(np.sum(component_adjacency, axis=1)))
            group_level_center = np.median(level[component], axis=0)
            group_derivative_center = np.median(derivative[component], axis=0)
            group_center_distance = float(
                np.sqrt(
                    config.level_distance_weight
                    * np.mean((group_level_center - level[medoid_index]) ** 2)
                    + config.derivative_distance_weight
                    * np.mean(
                        (group_derivative_center - derivative[medoid_index]) ** 2
                    )
                )
            )
            group_center_rarity = max(
                (
                    group_center_distance - center_distance_location
                )
                / center_distance_scale,
                0.0,
            )
            base_candidate = bool(
                size <= maximum_size
                and stability >= config.group_edge_stability
                and separation_ratio >= config.group_separation_ratio
                and group_center_distance >= config.group_center_min_effect
                and group_center_rarity >= config.group_center_rarity_cutoff
                and edge_density >= config.group_min_edge_density
                and minimum_member_degree >= config.group_min_member_degree
            )
            group_number += 1
            group_id = f"{view}_{batch_id}_rare_group_{group_number:02d}"
            members = battery_ids[component]
            group_rows.append(
                {
                    "rare_group_id": group_id,
                    "view": view,
                    "batch_id": str(batch_id),
                    "group_size": int(size),
                    "batch_fraction": float(size / n_rows),
                    "member_battery_ids": ",".join(members),
                    "within_group_distance": within_distance,
                    "nearest_outside_distance": outside_distance,
                    "separation_ratio": float(separation_ratio),
                    "central_reference_battery_id": battery_ids[medoid_index],
                    "group_center_distance": group_center_distance,
                    "group_center_rarity": float(group_center_rarity),
                    "group_center_rarity_cutoff": float(
                        config.group_center_rarity_cutoff
                    ),
                    "stable_edge_density": float(edge_density),
                    "minimum_member_degree": int(minimum_member_degree),
                    "rare_group_stability": stability,
                    "is_base_rare_group_candidate": base_candidate,
                    "interpretation": (
                        "stable_small_cohesive_centrally_rare_group_not_fault_label"
                        if base_candidate
                        else "descriptive_neighbor_component"
                    ),
                }
            )
            for member in members:
                membership_rows.append(
                    {
                        "battery_id": str(member),
                        "rare_group_id": group_id,
                        "view": view,
                        "rare_group_stability": stability,
                        "is_base_rare_group_candidate": base_candidate,
                    }
                )
    return (
        pd.DataFrame(group_rows),
        pd.DataFrame(membership_rows),
        pd.DataFrame(pair_audit_rows),
    )


def _masked_landmark_distance(
    values: np.ndarray,
    observed: np.ndarray,
    columns: np.ndarray,
    *,
    minimum_common: int,
) -> np.ndarray:
    """Pairwise RMS distance using observed landmarks shared by both cells."""

    matrix = np.asarray(values, dtype=float)
    mask = np.asarray(observed, dtype=bool)
    selected = np.asarray(columns, dtype=int)
    n_rows = len(matrix)
    distance = np.full((n_rows, n_rows), np.inf, dtype=float)
    np.fill_diagonal(distance, 0.0)
    for left in range(n_rows):
        for right in range(left + 1, n_rows):
            common = selected[mask[left, selected] & mask[right, selected]]
            if len(common) < minimum_common:
                continue
            value = float(
                np.sqrt(np.mean((matrix[left, common] - matrix[right, common]) ** 2))
            )
            distance[left, right] = value
            distance[right, left] = value
    return distance


def _standardized_lifetime_landmarks(
    batch: pd.DataFrame,
    config: FunctionalPatternConfig,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return robust log-duration vectors plus explicit observed-event masks."""

    labels = [f"t{int(round(value * 100))}" for value in LANDMARK_THRESHOLDS]
    n_rows = len(batch)
    values = np.full((n_rows, len(labels)), np.nan, dtype=float)
    raw_duration = np.full_like(values, np.nan)
    observed = np.zeros_like(values, dtype=bool)
    batch_median_duration = np.full(len(labels), np.nan, dtype=float)
    for column_index, label in enumerate(labels):
        observed[:, column_index] = (
            batch[f"{label}_censoring"].astype(str).eq("observed").to_numpy(dtype=bool)
            & np.isfinite(batch[label].to_numpy(dtype=float))
        )
        if np.sum(observed[:, column_index]) < 3:
            continue
        durations = (
            batch.loc[observed[:, column_index], label].to_numpy(dtype=float)
            - batch.loc[observed[:, column_index], "cycle_start"].to_numpy(dtype=float)
        )
        durations = np.maximum(durations, 0.0)
        raw_duration[observed[:, column_index], column_index] = durations
        batch_median_duration[column_index] = float(np.median(durations))
        logged = np.log1p(durations)
        center, scale = _robust_location_scale(logged)
        scale = max(scale, config.lifetime_log_duration_scale_floor)
        values[observed[:, column_index], column_index] = (logged - center) / scale
    return labels, values, observed, raw_duration, batch_median_duration


def _nearest_lifetime_peer_map(
    landmark_table: pd.DataFrame,
    config: FunctionalPatternConfig,
) -> dict[str, str]:
    """Return same-batch peers from common observed landmark distances."""

    peers: dict[str, str] = {}
    for _, batch in landmark_table.groupby("batch_id", sort=True):
        batch = batch.sort_values("battery_id").reset_index(drop=True)
        _, values, observed, _, _ = _standardized_lifetime_landmarks(batch, config)
        eligible = np.flatnonzero(
            np.sum(observed, axis=1)
            >= config.lifetime_group_min_observed_landmarks
        )
        if len(eligible) < 2:
            continue
        distance = _masked_landmark_distance(
            values[eligible],
            observed[eligible],
            np.arange(len(LANDMARK_THRESHOLDS)),
            minimum_common=config.lifetime_group_min_observed_landmarks,
        )
        battery_ids = batch.loc[eligible, "battery_id"].astype(str).to_numpy()
        for row_index, battery_id in enumerate(battery_ids):
            finite = np.flatnonzero(
                np.isfinite(distance[row_index])
                & (np.arange(len(eligible)) != row_index)
            )
            if len(finite):
                ordered = finite[np.argsort(distance[row_index, finite])]
                nearest = ordered[: min(config.lifetime_peer_count, len(ordered))]
                peers[battery_id] = ",".join(battery_ids[nearest])
    return peers


def _discover_lifetime_rare_groups(
    landmark_table: pd.DataFrame,
    config: FunctionalPatternConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find rare cohesive lifetime groups without treating censoring as an event.

    Each vector contains log durations to sustained T95/T90/T85/T80.  A pair
    distance uses only landmarks observed for both cells.  Censoring times are
    therefore never inserted as if the threshold had actually been reached.
    """

    group_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(config.random_state + 6_009)
    group_number = 0
    for batch_id, batch in landmark_table.groupby("batch_id", sort=True):
        batch = batch.sort_values("battery_id").reset_index(drop=True)
        n_batch = len(batch)
        (
            labels,
            values,
            observed,
            raw_duration,
            batch_median_duration,
        ) = _standardized_lifetime_landmarks(batch, config)
        eligible_mask = (
            np.sum(observed, axis=1)
            >= config.lifetime_group_min_observed_landmarks
        )
        eligible_indices = np.flatnonzero(eligible_mask)
        if len(eligible_indices) < 4:
            continue
        keys = batch.loc[
            eligible_indices, ["battery_id", "cell_id", "batch_id"]
        ].reset_index(drop=True)
        eligible_values = values[eligible_indices]
        eligible_observed = observed[eligible_indices]
        n_rows = len(keys)
        neighbor_count = min(config.lifetime_group_mutual_neighbors, n_rows - 1)
        landmark_count = len(labels)
        sampled_count = min(
            landmark_count,
            max(
                config.lifetime_group_min_common_landmarks,
                int(
                    math.ceil(
                        config.lifetime_group_landmark_fraction * landmark_count
                    )
                ),
            ),
        )
        edge_counts = np.zeros((n_rows, n_rows), dtype=float)
        valid_pair_counts = np.zeros((n_rows, n_rows), dtype=float)
        for repeat in range(config.stability_repeats):
            if sampled_count == landmark_count - 1:
                # Cycle evenly through leave-one-landmark-out subsets.  Purely
                # random draws would give unstable evaluable coverage when a
                # censored cell has exactly three observed landmarks.
                omitted = repeat % landmark_count
                columns = np.delete(np.arange(landmark_count), omitted)
            else:
                columns = np.sort(
                    rng.choice(landmark_count, size=sampled_count, replace=False)
                )
            distance = _masked_landmark_distance(
                eligible_values,
                eligible_observed,
                columns,
                minimum_common=config.lifetime_group_min_common_landmarks,
            )
            valid_pairs = np.isfinite(distance)
            np.fill_diagonal(valid_pairs, False)
            valid_pair_counts += valid_pairs
            directed = np.zeros((n_rows, n_rows), dtype=bool)
            for row_index in range(n_rows):
                finite = np.flatnonzero(
                    np.isfinite(distance[row_index])
                    & (np.arange(n_rows) != row_index)
                )
                if len(finite):
                    ordered = finite[np.argsort(distance[row_index, finite])]
                    directed[
                        row_index, ordered[: min(neighbor_count, len(ordered))]
                    ] = True
            edge_counts += directed & directed.T
        edge_frequency = np.divide(
            edge_counts,
            valid_pair_counts,
            out=np.zeros_like(edge_counts),
            where=valid_pair_counts > 0,
        )
        pair_coverage = valid_pair_counts / config.stability_repeats
        stable_adjacency = (
            edge_frequency >= config.lifetime_group_edge_stability
        ) & (
            pair_coverage >= config.lifetime_group_min_pair_coverage
        )
        np.fill_diagonal(stable_adjacency, False)
        full_distance = _masked_landmark_distance(
            eligible_values,
            eligible_observed,
            np.arange(landmark_count),
            minimum_common=config.lifetime_group_min_common_landmarks,
        )
        cell_center_distance = np.asarray(
            [
                np.sqrt(np.mean(row[mask] ** 2))
                for row, mask in zip(eligible_values, eligible_observed)
            ],
            dtype=float,
        )
        center_distance_location, center_distance_scale = _robust_location_scale(
            cell_center_distance
        )
        battery_ids = keys["battery_id"].astype(str).to_numpy()
        for left in range(n_rows):
            for right in range(left + 1, n_rows):
                if edge_frequency[left, right] > 0:
                    audit_rows.append(
                        {
                            "battery_id": battery_ids[left],
                            "peer_battery_id": battery_ids[right],
                            "cell_id": str(keys.iloc[left]["cell_id"]),
                            "batch_id": str(batch_id),
                            "view": "lifetime_rare_group_pair",
                            "stratum": str(batch_id),
                            "repeat": -1,
                            "sampled_grid_points": int(sampled_count),
                            "pair_edge_frequency": float(edge_frequency[left, right]),
                            "pair_evaluable_frequency": float(pair_coverage[left, right]),
                            "selected": bool(stable_adjacency[left, right]),
                            "censoring_rule": "observed_common_landmarks_only",
                        }
                    )
        for component in _connected_components(stable_adjacency):
            size = len(component)
            if size < config.lifetime_group_min_size:
                continue
            maximum_size = min(
                config.lifetime_group_max_size,
                max(
                    config.lifetime_group_min_size,
                    int(
                        math.floor(
                            config.lifetime_group_max_fraction * n_rows
                        )
                    ),
                ),
            )
            inside = np.zeros(n_rows, dtype=bool)
            inside[component] = True
            outside = np.flatnonzero(~inside)
            within_values = full_distance[np.ix_(component, component)][
                np.triu_indices(size, k=1)
            ]
            within_values = within_values[np.isfinite(within_values)]
            within_distance = (
                float(np.median(within_values)) if len(within_values) else np.inf
            )
            nearest_outside = []
            for member in component:
                finite = full_distance[member, outside]
                finite = finite[np.isfinite(finite)]
                if len(finite):
                    nearest_outside.append(float(np.min(finite)))
            outside_distance = (
                float(np.median(nearest_outside)) if nearest_outside else 0.0
            )
            separation_ratio = outside_distance / max(within_distance, 1e-12)
            component_edges = edge_frequency[np.ix_(component, component)][
                np.triu_indices(size, k=1)
            ]
            positive_edges = component_edges[component_edges > 0]
            stability = float(np.mean(positive_edges)) if len(positive_edges) else 0.0
            component_adjacency = stable_adjacency[np.ix_(component, component)]
            stable_edge_count = int(np.sum(np.triu(component_adjacency, k=1)))
            possible_edge_count = size * (size - 1) // 2
            edge_density = stable_edge_count / max(possible_edge_count, 1)
            # A group center uses only landmarks observed by every member.  This
            # strict intersection makes the accelerated/long-life direction
            # interpretable and prevents a censoring lower bound becoming a
            # fabricated event time.
            common_group_landmarks = np.flatnonzero(
                np.all(eligible_observed[component], axis=0)
            )
            if len(common_group_landmarks) >= config.lifetime_group_min_common_landmarks:
                group_center_vector = np.median(
                    eligible_values[np.ix_(component, common_group_landmarks)],
                    axis=0,
                )
                group_center_distance = float(
                    np.sqrt(np.mean(group_center_vector**2))
                )
                component_raw_duration = raw_duration[
                    np.ix_(eligible_indices[component], common_group_landmarks)
                ]
                group_median_duration = np.median(component_raw_duration, axis=0)
                reference_duration = batch_median_duration[common_group_landmarks]
                duration_ratios = group_median_duration / np.maximum(
                    reference_duration, 1e-12
                )
                member_duration_spread = np.max(
                    component_raw_duration, axis=0
                ) / np.maximum(np.min(component_raw_duration, axis=0), 1e-12)
                pair_duration_consistent = bool(
                    np.all(
                        member_duration_spread
                        <= config.lifetime_group_pair_duration_ratio
                    )
                )
                common_set = set(map(int, common_group_landmarks))
                t90_available = 1 in common_set
                late_anchor = 3 if 3 in common_set else (2 if 2 in common_set else -1)
                ratio_by_landmark = {
                    int(column): float(ratio)
                    for column, ratio in zip(common_group_landmarks, duration_ratios)
                }
                accelerated_direction = bool(
                    t90_available
                    and late_anchor >= 0
                    and ratio_by_landmark[1]
                    <= config.lifetime_group_acceleration_ratio
                    and ratio_by_landmark[late_anchor]
                    <= config.lifetime_group_acceleration_ratio
                    and np.all(duration_ratios < 1.0)
                    and pair_duration_consistent
                )
                lifetime_pattern = (
                    "accelerated_lifetime_pattern"
                    if accelerated_direction
                    else "nonqualifying_or_mixed_lifetime_pattern"
                )
            else:
                group_center_distance = 0.0
                lifetime_pattern = "insufficient_common_observed_landmarks"
                accelerated_direction = False
                pair_duration_consistent = False
                duration_ratios = np.asarray([], dtype=float)
                member_duration_spread = np.asarray([], dtype=float)
            group_center_rarity = max(
                (group_center_distance - center_distance_location)
                / center_distance_scale,
                0.0,
            )
            base_candidate = bool(
                size >= config.lifetime_group_min_size
                and size <= maximum_size
                and np.isfinite(within_distance)
                and stability >= config.lifetime_group_edge_stability
                and edge_density >= config.group_min_edge_density
                and separation_ratio >= config.lifetime_group_separation_ratio
                and group_center_distance >= config.group_center_min_effect
                and group_center_rarity
                >= config.lifetime_group_center_rarity_cutoff
                and len(common_group_landmarks)
                >= config.lifetime_group_min_common_landmarks
                and accelerated_direction
            )
            group_number += 1
            group_id = f"lifetime_{batch_id}_rare_group_{group_number:02d}"
            members = battery_ids[component]
            group_rows.append(
                {
                    "rare_group_id": group_id,
                    "view": "lifetime",
                    "batch_id": str(batch_id),
                    "group_size": int(size),
                    "batch_fraction": float(size / n_rows),
                    "member_battery_ids": ",".join(members),
                    "within_group_distance": within_distance,
                    "nearest_outside_distance": outside_distance,
                    "separation_ratio": float(separation_ratio),
                    "central_reference_battery_id": "batch_landmark_median",
                    "group_center_distance": group_center_distance,
                    "group_center_rarity": float(group_center_rarity),
                    "group_center_rarity_cutoff": float(
                        config.lifetime_group_center_rarity_cutoff
                    ),
                    "stable_edge_density": float(edge_density),
                    "minimum_member_degree": int(
                        np.min(np.sum(component_adjacency, axis=1))
                    ),
                    "rare_group_stability": stability,
                    "is_base_rare_group_candidate": base_candidate,
                    "lifetime_pattern": lifetime_pattern,
                    "common_observed_landmarks": ",".join(
                        labels[index] for index in common_group_landmarks
                    ),
                    "group_to_batch_duration_ratios": ",".join(
                        f"{labels[index]}:{ratio:.6g}"
                        for index, ratio in zip(
                            common_group_landmarks, duration_ratios
                        )
                    ),
                    "maximum_member_duration_ratio": (
                        float(np.max(member_duration_spread))
                        if len(member_duration_spread)
                        else np.nan
                    ),
                    "censoring_rule": "observed_common_landmarks_only",
                    "interpretation": (
                        "stable_small_cohesive_centrally_rare_lifetime_group_not_fault_label"
                        if base_candidate
                        else "descriptive_lifetime_neighbor_component"
                    ),
                }
            )
            for member in members:
                membership_rows.append(
                    {
                        "battery_id": str(member),
                        "rare_group_id": group_id,
                        "view": "lifetime",
                        "rare_group_stability": stability,
                        "is_base_rare_group_candidate": base_candidate,
                    }
                )
    return (
        pd.DataFrame(group_rows),
        pd.DataFrame(membership_rows),
        pd.DataFrame(audit_rows),
    )


def _combine_rare_groups(
    shape_representation: pd.DataFrame,
    absolute_representation: pd.DataFrame,
    landmark_table: pd.DataFrame,
    config: FunctionalPatternConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shape_groups, shape_members, shape_audit = _discover_rare_groups_for_view(
        shape_representation,
        view="shape",
        config=config,
        seed_offset=4_009,
    )
    absolute_groups, absolute_members, absolute_audit = _discover_rare_groups_for_view(
        absolute_representation,
        view="absolute",
        config=config,
        seed_offset=5_009,
    )
    lifetime_groups, lifetime_members, lifetime_group_audit = (
        _discover_lifetime_rare_groups(landmark_table, config)
    )
    group_summary = pd.concat(
        [
            frame
            for frame in [shape_groups, absolute_groups, lifetime_groups]
            if not frame.empty
        ],
        ignore_index=True,
    ) if (
        not shape_groups.empty
        or not absolute_groups.empty
        or not lifetime_groups.empty
    ) else pd.DataFrame(
        columns=[
            "rare_group_id",
            "view",
            "batch_id",
            "group_size",
            "batch_fraction",
            "member_battery_ids",
            "within_group_distance",
            "nearest_outside_distance",
            "separation_ratio",
            "central_reference_battery_id",
            "group_center_distance",
            "group_center_rarity",
            "group_center_rarity_cutoff",
            "rare_group_stability",
            "is_base_rare_group_candidate",
            "interpretation",
        ]
    )
    memberships = pd.concat(
        [
            frame
            for frame in [shape_members, absolute_members, lifetime_members]
            if not frame.empty
        ],
        ignore_index=True,
    ) if (
        not shape_members.empty
        or not absolute_members.empty
        or not lifetime_members.empty
    ) else pd.DataFrame(
        columns=[
            "battery_id",
            "rare_group_id",
            "view",
            "rare_group_stability",
            "is_base_rare_group_candidate",
        ]
    )
    if not memberships.empty:
        candidate_support = (
            memberships.loc[memberships["is_base_rare_group_candidate"]]
            .groupby("battery_id")["view"]
            .nunique()
        )
        memberships["cell_candidate_view_support"] = memberships["battery_id"].map(
            candidate_support
        ).fillna(0).astype(int)
        memberships["is_rare_group_candidate"] = (
            memberships["is_base_rare_group_candidate"]
            & (
                memberships["cell_candidate_view_support"]
                >= config.group_required_view_support
            )
        )
    pair_audit = pd.concat(
        [
            frame
            for frame in [shape_audit, absolute_audit, lifetime_group_audit]
            if not frame.empty
        ],
        ignore_index=True,
    ) if (
        not shape_audit.empty
        or not absolute_audit.empty
        or not lifetime_group_audit.empty
    ) else pd.DataFrame()
    if not group_summary.empty:
        # For the default one-view support rule this equals the base flag.  The
        # cell-level result below remains authoritative when stricter multi-view
        # support is requested.
        group_summary["is_rare_group_candidate"] = group_summary[
            "is_base_rare_group_candidate"
        ].astype(bool)
    return group_summary, memberships, pair_audit


def _review_status(row: pd.Series) -> str:
    persistent = bool(
        row["is_shape_candidate"]
        or row["is_absolute_pattern_candidate"]
        or row.get("is_shape_rare_group_candidate", False)
        or row.get("is_absolute_rare_group_candidate", False)
    )
    lifetime = bool(
        row["is_lifetime_candidate"]
        or row.get("is_lifetime_rare_group_candidate", False)
    )
    transient = bool(row["is_transient_candidate"])
    if persistent and lifetime and transient:
        return "persistent_pattern_lifetime_and_transient_review"
    if persistent and lifetime:
        return "persistent_pattern_and_lifetime_review"
    if persistent and transient:
        return "persistent_pattern_and_transient_review"
    if persistent:
        return "persistent_degradation_pattern_review"
    if lifetime and transient:
        return "lifetime_and_transient_review"
    if lifetime:
        return "lifetime_or_degradation_speed_review"
    if transient:
        return "transient_or_data_quality_review"
    if (
        not bool(row.get("shape_analysis_eligible", False))
        and "gap" in str(row.get("shape_ineligibility_reason", "")).lower()
    ):
        return "cycle_gap_shape_ineligible_data_quality_review"
    return "no_stable_functional_rarity_evidence"


def build_functional_representations(
    raw_curves: pd.DataFrame,
    *,
    config: FunctionalPatternConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess SOH and build the two functional views without scoring.

    Returns ``(processed_curves, shape_representation,
    absolute_representation, landmark_table)``.  This public split is useful
    when plots or sensitivity analyses need the functional inputs without
    rerunning the anomaly scoring stage.
    """

    config = config or FunctionalPatternConfig()
    _validate_config(config)
    processed_curves, landmark_table = _preprocess_curves(raw_curves, config)
    absolute_representation, absolute_eligible, absolute_availability = (
        _build_absolute_representation(processed_curves, landmark_table, config)
    )
    shape_representation, shape_eligible, shape_reason = _build_shape_representation(
        processed_curves, landmark_table, config
    )
    landmark_table = landmark_table.copy()
    landmark_table["shape_analysis_eligible"] = landmark_table["battery_id"].map(
        shape_eligible
    ).fillna(False).astype(bool)
    landmark_table["shape_ineligibility_reason"] = landmark_table["battery_id"].map(
        shape_reason
    ).fillna("unknown")
    landmark_table["absolute_analysis_eligible"] = landmark_table["battery_id"].map(
        absolute_eligible
    ).fillna(False).astype(bool)
    landmark_table = landmark_table.merge(
        absolute_availability,
        on="batch_id",
        how="left",
        validate="many_to_one",
    )
    return (
        processed_curves,
        shape_representation,
        absolute_representation,
        landmark_table,
    )


def analyze_functional_patterns(
    raw_curves: pd.DataFrame,
    *,
    config: FunctionalPatternConfig | None = None,
) -> FunctionalPatternResult:
    """Analyze persistent MATR degradation patterns without anomaly labels.

    Parameters
    ----------
    raw_curves:
        Long table with one row per battery/cycle and columns ``battery_id``,
        ``cell_id``, ``batch_id``, ``cycle``, and ``soh``.  SOH should already
        use exactly the target definition used by the forecasting model.
    config:
        Optional immutable analysis configuration.

    Returns
    -------
    FunctionalPatternResult
        Separate absolute-cycle, T90-normalized shape, landmark lifetime, and
        transient evidence.  Candidate flags express stable cohort rarity,
        never a probability or a confirmed physical fault.

    Notes
    -----
    * Absolute-cycle analysis uses only the batch-specific common support.
      Shorter cells are not extrapolated; their lifetime landmarks remain in
      the output.
    * Shape analysis requires an observed sustained T90.  A censored cell is
      explicitly ineligible rather than normalized by its arbitrary last
      observation.
    * The detector assumes a majority of each comparison stratum is typical.
      Planned charging policy or temperature covariates should be used to form
      finer strata when available.
    """

    config = config or FunctionalPatternConfig()
    _validate_config(config)
    if config.verbose:
        print("[functional] preprocessing and functional representations", flush=True)
    (
        processed_curves,
        shape_representation,
        absolute_representation,
        landmark_table,
    ) = build_functional_representations(
        raw_curves,
        config=config,
    )

    if config.verbose:
        print("[functional] robust FPCA/outlyingness/distance scoring", flush=True)
    shape_scores, shape_audit = _score_representation(
        shape_representation,
        view_prefix="shape",
        config=config,
        stratify_by_batch=True,
    )
    absolute_scores, absolute_audit = _score_representation(
        absolute_representation,
        view_prefix="absolute",
        config=config,
        stratify_by_batch=True,
    )
    lifetime_scores, lifetime_audit = _score_lifetime(landmark_table, config)
    transient_scores, transient_audit = _score_transient(landmark_table, config)
    if config.verbose:
        print("[functional] stable rare-group discovery", flush=True)
    rare_group_summary, memberships, group_audit = _combine_rare_groups(
        shape_representation,
        absolute_representation,
        landmark_table,
        config,
    )

    keys = ["battery_id", "cell_id", "batch_id"]
    summary = landmark_table.copy()
    for score_frame in [shape_scores, absolute_scores, lifetime_scores, transient_scores]:
        if not score_frame.empty:
            summary = summary.merge(score_frame, on=keys, how="left", validate="one_to_one")

    # Stable group evidence is aggregated per cell without hiding multi-view groups.
    group_columns = [
        "battery_id",
        "rare_group_id",
        "shape_rare_group_id",
        "absolute_rare_group_id",
        "lifetime_rare_group_id",
        "rare_group_stability",
        "rare_group_view_support",
        "is_shape_rare_group_candidate",
        "is_absolute_rare_group_candidate",
        "is_lifetime_rare_group_candidate",
        "is_rare_group_candidate",
    ]
    if memberships.empty:
        group_by_cell = pd.DataFrame(columns=group_columns)
    else:
        candidate_memberships = memberships.loc[
            memberships["is_rare_group_candidate"]
        ].copy()
        if candidate_memberships.empty:
            group_by_cell = pd.DataFrame(columns=group_columns)
        else:
            rows = []
            for battery_id, group in candidate_memberships.groupby(
                "battery_id", sort=True
            ):
                ids_by_view = {
                    view: ";".join(
                        sorted(
                            set(
                                group.loc[
                                    group["view"].eq(view), "rare_group_id"
                                ].astype(str)
                            )
                        )
                    )
                    for view in ["shape", "absolute", "lifetime"]
                }
                rows.append(
                    {
                        "battery_id": str(battery_id),
                        "rare_group_id": ";".join(
                            sorted(set(group["rare_group_id"].astype(str)))
                        ),
                        "shape_rare_group_id": ids_by_view["shape"],
                        "absolute_rare_group_id": ids_by_view["absolute"],
                        "lifetime_rare_group_id": ids_by_view["lifetime"],
                        "rare_group_stability": float(
                            group["rare_group_stability"].max()
                        ),
                        "rare_group_view_support": int(group["view"].nunique()),
                        "is_shape_rare_group_candidate": bool(ids_by_view["shape"]),
                        "is_absolute_rare_group_candidate": bool(
                            ids_by_view["absolute"]
                        ),
                        "is_lifetime_rare_group_candidate": bool(
                            ids_by_view["lifetime"]
                        ),
                        "is_rare_group_candidate": True,
                    }
                )
            group_by_cell = pd.DataFrame(rows, columns=group_columns)
    summary = summary.merge(group_by_cell, on="battery_id", how="left", validate="one_to_one")
    summary["rare_group_id"] = summary["rare_group_id"].fillna("")
    summary["rare_group_stability"] = summary["rare_group_stability"].fillna(0.0)
    summary["rare_group_view_support"] = summary["rare_group_view_support"].fillna(0).astype(int)
    summary["is_rare_group_candidate"] = summary[
        "is_rare_group_candidate"
    ].fillna(False).astype(bool)
    for column in [
        "shape_rare_group_id",
        "absolute_rare_group_id",
        "lifetime_rare_group_id",
    ]:
        summary[column] = summary[column].fillna("")
    for column in [
        "is_shape_rare_group_candidate",
        "is_absolute_rare_group_candidate",
        "is_lifetime_rare_group_candidate",
    ]:
        summary[column] = summary[column].fillna(False).astype(bool)

    # Public result-contract aliases and explicit defaults for ineligible views.
    if "shape_pattern_score" not in summary:
        summary["shape_pattern_score"] = np.nan
    summary["shape_score"] = summary["shape_pattern_score"]
    if "absolute_pattern_score" not in summary:
        summary["absolute_pattern_score"] = np.nan
    for column in [
        "shape_selection_frequency",
        "absolute_selection_frequency",
    ]:
        if column not in summary:
            summary[column] = 0.0
        summary[column] = summary[column].fillna(0.0)
    if "is_shape_candidate" not in summary:
        summary["is_shape_candidate"] = False
    if "is_absolute_candidate" not in summary:
        summary["is_absolute_candidate"] = False
    summary["is_shape_candidate"] = summary["is_shape_candidate"].fillna(False).astype(bool)
    summary["is_absolute_pattern_candidate"] = summary[
        "is_absolute_candidate"
    ].fillna(False).astype(bool)
    for column in [
        "lifetime_selection_frequency",
        "transient_selection_frequency",
    ]:
        summary[column] = summary[column].fillna(0.0)
    for column in ["is_lifetime_candidate", "is_transient_candidate"]:
        summary[column] = summary[column].fillna(False).astype(bool)

    peer_map = _nearest_peer_map(shape_representation, config)
    summary["nearest_shape_peers"] = summary["battery_id"].map(peer_map).fillna("")
    lifetime_peer_map = _nearest_lifetime_peer_map(landmark_table, config)
    summary["nearest_lifetime_peers"] = summary["battery_id"].map(
        lifetime_peer_map
    ).fillna("")
    summary["review_status"] = summary.apply(_review_status, axis=1)
    summary["detector_scope"] = "retrospective_full_curve_functional_screen"
    summary["rarity_interpretation"] = (
        "robust_descriptive_rarity_not_p_value_not_fault_probability"
    )
    summary["reference_assumption"] = (
        "majority_typical_within_batch;condition_on_policy_when_available"
    )
    summary["config_snapshot"] = str(asdict(config))

    required_contract = [
        "battery_id",
        "cell_id",
        "batch_id",
        "shape_analysis_eligible",
        "absolute_analysis_eligible",
        "shape_score",
        "shape_selection_frequency",
        "is_shape_candidate",
        "absolute_pattern_score",
        "absolute_selection_frequency",
        "is_absolute_pattern_candidate",
        "lifetime_score",
        "lifetime_selection_frequency",
        "is_lifetime_candidate",
        "transient_score",
        "transient_selection_frequency",
        "is_transient_candidate",
        "rare_group_id",
        "rare_group_stability",
        "is_rare_group_candidate",
        "shape_rare_group_id",
        "absolute_rare_group_id",
        "lifetime_rare_group_id",
        "is_shape_rare_group_candidate",
        "is_absolute_rare_group_candidate",
        "is_lifetime_rare_group_candidate",
        "nearest_shape_peers",
        "nearest_lifetime_peers",
        "review_status",
        "shape_score_distance",
        "shape_orthogonal_distance",
        "shape_sd_ratio",
        "shape_od_ratio",
        "absolute_score_distance",
        "absolute_orthogonal_distance",
        "absolute_sd_ratio",
        "absolute_od_ratio",
    ]
    for column in required_contract:
        if column not in summary:
            if column.startswith("is_") or column.endswith("_eligible"):
                summary[column] = False
            elif column in {
                "rare_group_id",
                "shape_rare_group_id",
                "absolute_rare_group_id",
                "lifetime_rare_group_id",
                "nearest_shape_peers",
                "nearest_lifetime_peers",
                "review_status",
            }:
                summary[column] = ""
            else:
                summary[column] = np.nan
    if len(summary) != raw_curves["battery_id"].nunique():
        raise RuntimeError("functional summary does not contain exactly one row per battery")
    if summary["battery_id"].duplicated().any():
        raise RuntimeError("functional summary contains duplicate battery IDs")
    summary = summary.sort_values(
        [
            "is_rare_group_candidate",
            "is_shape_candidate",
            "is_absolute_pattern_candidate",
            "is_lifetime_candidate",
            "shape_score",
            "absolute_pattern_score",
        ],
        ascending=[False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    stability_parts = [
        frame
        for frame in [
            shape_audit,
            absolute_audit,
            lifetime_audit,
            transient_audit,
            group_audit,
        ]
        if not frame.empty
    ]
    stability_audit = (
        pd.concat(stability_parts, ignore_index=True, sort=False)
        if stability_parts
        else pd.DataFrame()
    )
    return FunctionalPatternResult(
        pattern_summary=summary,
        processed_curves=processed_curves,
        shape_representation=shape_representation,
        absolute_representation=absolute_representation,
        landmark_table=landmark_table,
        rare_group_summary=rare_group_summary,
        stability_audit=stability_audit,
    )


__all__ = [
    "FunctionalPatternConfig",
    "FunctionalPatternResult",
    "analyze_functional_patterns",
    "build_functional_representations",
    "assert_curve_prediction_parity",
    "load_matr_soh_curves",
]
