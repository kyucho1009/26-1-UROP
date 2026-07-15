from __future__ import annotations

"""Cross-fitted forecast-residual evidence for MATR degradation screening.

The functions in this module deliberately call the result an empirical OOF
tail rank, not a strict conformal p-value.  Every target battery must have been
excluded from model fitting, early stopping, and normalization.  Fold-specific
inner-validation predictions are reused after early stopping to calibrate
residual components and horizon scales; they are not an independent conformal set.
outer-fold batteries are then compared only with same-batch batteries having
similar observation coverage.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from matr_anomaly_scoring import (
        add_scores_by_seed,
        aggregate_cell_nonconformity,
        apply_component_calibration,
        apply_horizon_severity_calibration,
        fit_component_calibration,
        fit_horizon_severity_calibration,
        prepare_residual_features,
        select_alpha_by_seed,
        summarize_common_horizon_scores,
    )
except ImportError:  # Package-style import used by tests and notebooks.
    from .matr_anomaly_scoring import (  # type: ignore
        add_scores_by_seed,
        aggregate_cell_nonconformity,
        apply_component_calibration,
        apply_horizon_severity_calibration,
        fit_component_calibration,
        fit_horizon_severity_calibration,
        prepare_residual_features,
        select_alpha_by_seed,
        summarize_common_horizon_scores,
    )


@dataclass(frozen=True)
class OOFResidualConfig:
    """Configuration for repeated battery-level OOF residual analysis."""

    target_model: str = "cpmlp_cpdsconv_fusion"
    expected_horizons: tuple[int, ...] = (10, 50, 100)
    alpha_selection_horizons: tuple[int, int] = (50, 100)
    alpha_grid: tuple[float, ...] = tuple(
        float(value) for value in np.linspace(0.0, 1.0, 21)
    )
    min_common_windows: int = 5
    min_paired_cells: int = 5
    bootstrap_repeats: int = 500
    candidate_p: float = 0.10
    strong_p: float = 0.05
    coverage_reference_cells: int = 20
    minimum_coverage_reference_cells: int = 9
    coverage_warning_abs_rank_corr: float = 0.50
    required_repeat_fraction: float = 2.0 / 3.0
    random_state: int = 20260715
    fold_warning_adjusted_rank_effect: float = 0.10
    fold_warning_permutation_p: float = 0.05
    fold_permutation_repeats: int = 1_000
    minimum_fold_cells: int = 3


@dataclass
class OOFResidualResult:
    """All auditable tables from :func:`analyze_oof_residual_predictions`."""

    cell_score_by_repeat: pd.DataFrame
    physical_cell_summary: pd.DataFrame
    fold_calibration: pd.DataFrame
    alpha_search: pd.DataFrame
    coverage_audit: pd.DataFrame
    scored_outer_windows: pd.DataFrame
    outer_prediction_parity: pd.DataFrame


@dataclass
class FunctionalOOFComparisonResult:
    """Agreement audits for complementary, non-independent SOH views."""

    cell_comparison: pd.DataFrame
    comparison_by_batch: pd.DataFrame
    overlap_audit: pd.DataFrame
    rank_agreement_audit: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{context} is missing columns: {missing}")


def _validate_config(config: OOFResidualConfig) -> None:
    horizons = tuple(int(value) for value in config.expected_horizons)
    if len(horizons) < 2 or len(set(horizons)) != len(horizons):
        raise ValueError("expected_horizons must contain distinct values")
    if not set(config.alpha_selection_horizons).issubset(horizons):
        raise ValueError("alpha_selection_horizons must be in expected_horizons")
    if config.min_common_windows < 1 or config.min_paired_cells < 3:
        raise ValueError("minimum window/cell counts are too small")
    if config.bootstrap_repeats < 100:
        raise ValueError("bootstrap_repeats must be at least 100")
    if not 0 < config.strong_p <= config.candidate_p < 1:
        raise ValueError("Require 0 < strong_p <= candidate_p < 1")
    if config.coverage_reference_cells < config.minimum_coverage_reference_cells:
        raise ValueError("coverage_reference_cells is below its minimum")
    if config.minimum_coverage_reference_cells < 2:
        raise ValueError("minimum_coverage_reference_cells must be at least 2")
    if not 0 < config.coverage_warning_abs_rank_corr <= 1:
        raise ValueError("coverage warning correlation must be in (0, 1]")
    if not 0 < config.required_repeat_fraction <= 1:
        raise ValueError("required_repeat_fraction must be in (0, 1]")
    if not 0 <= config.fold_warning_adjusted_rank_effect <= 1:
        raise ValueError("fold rank-effect warning threshold must be in [0, 1]")
    if not 0 < config.fold_warning_permutation_p < 1:
        raise ValueError("fold permutation p cutoff must be in (0, 1)")
    if config.fold_permutation_repeats < 100:
        raise ValueError("fold_permutation_repeats must be at least 100")
    if config.minimum_fold_cells < 2:
        raise ValueError("minimum_fold_cells must be at least 2")


def _normalize_prediction_keys(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    """Return predictions with explicit repeat/fold keys and verify uniqueness."""

    result = frame.copy()
    if "repeat_seed" not in result.columns:
        _require_columns(result, ["seed"], context)
        result["repeat_seed"] = result["seed"]
    if "seed" not in result.columns:
        result["seed"] = result["repeat_seed"]
    _require_columns(
        result,
        [
            "repeat_seed",
            "seed",
            "outer_fold",
            "model",
            "battery_id",
            "cell_id",
            "batch_id",
            "horizon",
            "target_cycle",
        ],
        context,
    )
    if not result["repeat_seed"].astype(str).equals(result["seed"].astype(str)):
        raise ValueError(f"{context} requires seed == repeat_seed")
    keys = [
        "repeat_seed",
        "outer_fold",
        "model",
        "battery_id",
        "horizon",
        "target_cycle",
    ]
    duplicated = result.duplicated(keys, keep=False)
    if duplicated.any():
        raise ValueError(
            f"{context} contains duplicate prediction keys: "
            + result.loc[duplicated, keys].head(10).to_dict("records").__repr__()
        )
    return result


def _validate_prediction_provenance(
    frame: pd.DataFrame,
    *,
    context: str,
    expected_split: str,
    expected_run_id: str | None = None,
) -> None:
    """Validate runner provenance when the optional columns are present.

    Older synthetic/unit-test prediction tables do not carry these columns, so
    absence is accepted.  A present column is never silently ignored.
    """

    expected_values: dict[str, str] = {
        "split": str(expected_split),
        "stage": "oof_cross_validation",
        "normalizer_fit_role": "inner_train",
    }
    if expected_run_id is not None:
        expected_values["run_id"] = str(expected_run_id)
    for column, expected in expected_values.items():
        if column not in frame.columns:
            continue
        observed = sorted(frame[column].dropna().astype(str).unique())
        if observed != [expected]:
            raise ValueError(
                f"{context} requires {column}={expected!r}, got {observed}"
            )


def _compact_prediction_parity(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only unique model-target SOH rows needed for curve parity checks."""

    columns = ["battery_id", "target_cycle", "actual_soh"]
    _require_columns(frame, columns, "outer prediction parity")
    parity = frame[columns].copy()
    parity["battery_id"] = parity["battery_id"].astype(str)
    parity["target_cycle"] = pd.to_numeric(
        parity["target_cycle"], errors="raise"
    )
    parity["actual_soh"] = pd.to_numeric(parity["actual_soh"], errors="raise")
    spread = parity.groupby(["battery_id", "target_cycle"])["actual_soh"].agg(
        lambda values: float(np.max(values) - np.min(values))
    )
    if spread.gt(1e-8).any():
        bad = spread[spread.gt(1e-8)].head(10).to_dict()
        raise RuntimeError(
            "OOF prediction rows disagree on actual SOH for battery/cycle: "
            f"{bad}"
        )
    return (
        parity.drop_duplicates(["battery_id", "target_cycle"])
        .sort_values(["battery_id", "target_cycle"])
        .reset_index(drop=True)
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    left = pd.Series(np.asarray(x, dtype=float)).rank(method="average")
    right = pd.Series(np.asarray(y, dtype=float)).rank(method="average")
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right))


def _coverage_distance(target: pd.Series, references: pd.DataFrame) -> np.ndarray:
    """Robust distance using only observation-coverage variables."""

    metric_columns = [
        "n_common_windows",
        "common_input_cycle_end",
        "common_input_cycle_span",
    ]
    all_rows = pd.concat(
        [target.to_frame().T[metric_columns], references[metric_columns]],
        ignore_index=True,
    ).apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(all_rows.to_numpy(dtype=float)).all():
        raise ValueError("Coverage matching received non-finite values")

    standardized: list[np.ndarray] = []
    for column in metric_columns:
        values = all_rows[column].to_numpy(dtype=float)
        if column == "n_common_windows":
            values = np.log1p(np.maximum(values, 0.0))
        center = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(values, ddof=0))
        if np.isfinite(scale) and scale > 1e-12:
            standardized.append((values - center) / scale)
    if not standardized:
        return np.zeros(len(references), dtype=float)
    matrix = np.column_stack(standardized)
    return np.sqrt(np.mean((matrix[1:] - matrix[0]) ** 2, axis=1))


def _stable_group_seed(
    *, random_state: int, repeat_seed: object, batch_id: object
) -> int:
    """Derive a process-independent RNG seed for one repeat/batch audit."""

    payload = (
        f"matr-oof-fold-audit|{int(random_state)}|{repeat_seed}|{batch_id}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32 - 1)


def _fold_rank_shift_audit(
    scores: np.ndarray,
    fold_labels: np.ndarray,
    *,
    minimum_fold_cells: int,
    permutation_repeats: int,
    random_state: int,
) -> dict[str, object]:
    """Audit fold-local score shifts with a robust rank-ANOVA permutation test.

    The adjusted rank effect is an adjusted R-squared computed after replacing
    scores by average ranks. It is zero under a balanced null in expectation
    and remains insensitive to the absolute score scale used by each fold.
    Permuting fold labels preserves the observed fold sizes.
    """

    values = np.asarray(scores, dtype=float)
    labels = np.asarray(fold_labels).astype(str)
    if len(values) != len(labels) or len(values) == 0:
        raise ValueError("Fold audit requires equally sized, non-empty arrays")
    if not np.isfinite(values).all():
        raise ValueError("Fold audit received non-finite cell scores")

    folds, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n_cells = int(len(values))
    n_folds = int(len(folds))
    minimum_cells = int(counts.min()) if len(counts) else 0
    available = bool(
        n_folds >= 2
        and n_cells > n_folds
        and minimum_cells >= int(minimum_fold_cells)
    )
    base: dict[str, object] = {
        "fold_calibration_audit_available": available,
        "n_outer_folds": n_folds,
        "fold_cell_count_min": minimum_cells,
        "fold_cell_count_max": int(counts.max()) if len(counts) else 0,
        "fold_rank_eta_squared": np.nan,
        "fold_rank_adjusted_eta_squared": np.nan,
        "fold_permutation_p_greater_equal": np.nan,
        "fold_permutation_repeats": int(permutation_repeats),
        "fold_effect_method": (
            "average_rank_adjusted_eta_squared_with_fixed_size_label_permutation"
        ),
    }
    if not available:
        return base

    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    grand_mean = float(np.mean(ranks))
    total_ss = float(np.sum((ranks - grand_mean) ** 2))
    if total_ss <= 1e-15:
        base.update(
            {
                "fold_rank_eta_squared": 0.0,
                "fold_rank_adjusted_eta_squared": 0.0,
                "fold_permutation_p_greater_equal": 1.0,
            }
        )
        return base

    def between_fold_ss(group_index: np.ndarray) -> float:
        sums = np.bincount(group_index, weights=ranks, minlength=n_folds)
        means = sums / counts
        return float(np.sum(counts * (means - grand_mean) ** 2))

    observed_between = between_fold_ss(inverse)
    eta_squared = float(np.clip(observed_between / total_ss, 0.0, 1.0))
    adjusted = float(
        np.clip(
            1.0 - (1.0 - eta_squared) * (n_cells - 1) / (n_cells - n_folds),
            0.0,
            1.0,
        )
    )
    rng = np.random.default_rng(int(random_state))
    exceedances = 0
    tolerance = 1e-12 * max(1.0, abs(observed_between))
    for _ in range(int(permutation_repeats)):
        permuted_index = inverse[rng.permutation(n_cells)]
        if between_fold_ss(permuted_index) >= observed_between - tolerance:
            exceedances += 1
    permutation_p = float((1 + exceedances) / (int(permutation_repeats) + 1))
    base.update(
        {
            "fold_rank_eta_squared": eta_squared,
            "fold_rank_adjusted_eta_squared": adjusted,
            "fold_permutation_p_greater_equal": permutation_p,
        }
    )
    return base


def apply_coverage_matched_tail_ranks(
    cell_scores: pd.DataFrame,
    *,
    config: OOFResidualConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate same-batch, coverage-matched empirical OOF tail ranks.

    References are selected without looking at their anomaly score.  ``>=`` tie
    handling and the ``+1`` correction mean identical scores never force an
    anomaly candidate. Large score/coverage association or a material shift
    between fold-local score distributions is an explicit safety gate, not
    merely a printed warning.
    """

    _validate_config(config)
    required = [
        "repeat_seed",
        "outer_fold",
        "battery_id",
        "cell_id",
        "batch_id",
        "cell_nonconformity_score",
        "n_common_windows",
        "common_input_cycle_start",
        "common_input_cycle_end",
        "common_input_cycle_span",
    ]
    _require_columns(cell_scores, required, "OOF cell scores")
    duplicate_keys = ["repeat_seed", "battery_id"]
    if cell_scores.duplicated(duplicate_keys, keep=False).any():
        raise ValueError("Each battery must have exactly one OOF score per repeat")

    audit_rows: list[dict[str, object]] = []
    for (repeat_seed, batch_id), group in cell_scores.groupby(
        ["repeat_seed", "batch_id"], sort=True
    ):
        correlations: dict[str, float] = {}
        scores = group["cell_nonconformity_score"].to_numpy(dtype=float)
        for metric in [
            "n_common_windows",
            "common_input_cycle_end",
            "common_input_cycle_span",
        ]:
            correlations[metric] = _spearman(
                scores, group[metric].to_numpy(dtype=float)
            )
        finite = [abs(value) for value in correlations.values() if np.isfinite(value)]
        maximum = max(finite) if finite else float("nan")
        fold_audit = _fold_rank_shift_audit(
            scores,
            group["outer_fold"].to_numpy(),
            minimum_fold_cells=config.minimum_fold_cells,
            permutation_repeats=config.fold_permutation_repeats,
            random_state=_stable_group_seed(
                random_state=config.random_state,
                repeat_seed=repeat_seed,
                batch_id=batch_id,
            ),
        )
        coverage_warning = bool(
            np.isfinite(maximum)
            and maximum >= config.coverage_warning_abs_rank_corr
        )
        fold_effect = float(fold_audit["fold_rank_adjusted_eta_squared"])
        fold_p = float(fold_audit["fold_permutation_p_greater_equal"])
        fold_warning = bool(
            fold_audit["fold_calibration_audit_available"]
            and np.isfinite(fold_effect)
            and np.isfinite(fold_p)
            and fold_effect >= config.fold_warning_adjusted_rank_effect
            and fold_p <= config.fold_warning_permutation_p
        )
        audit_rows.append(
            {
                "repeat_seed": repeat_seed,
                "batch_id": str(batch_id),
                "n_cells": int(len(group)),
                **{
                    f"rank_corr_score_vs_{metric}": value
                    for metric, value in correlations.items()
                },
                "max_abs_coverage_rank_corr": maximum,
                "warning_abs_rank_corr_cutoff": float(
                    config.coverage_warning_abs_rank_corr
                ),
                "has_large_coverage_association": coverage_warning,
                **fold_audit,
                "fold_warning_adjusted_rank_effect_cutoff": float(
                    config.fold_warning_adjusted_rank_effect
                ),
                "fold_warning_permutation_p_cutoff": float(
                    config.fold_warning_permutation_p
                ),
                "has_large_fold_association": fold_warning,
                "has_coverage_or_fold_confounding": bool(
                    coverage_warning or fold_warning
                ),
                "audit_scope": "repeat_and_batch_coverage_and_outer_fold",
            }
        )
    coverage_audit = pd.DataFrame(audit_rows)
    audit_index = coverage_audit.set_index(["repeat_seed", "batch_id"])
    coverage_warning_map = audit_index["has_large_coverage_association"]
    fold_warning_map = audit_index["has_large_fold_association"]
    combined_warning_map = audit_index["has_coverage_or_fold_confounding"]
    fold_audit_available_map = audit_index["fold_calibration_audit_available"]

    output_rows: list[dict[str, object]] = []
    for repeat_seed, repeat_group in cell_scores.groupby("repeat_seed", sort=True):
        for _, target in repeat_group.iterrows():
            batch = repeat_group[
                repeat_group["batch_id"].astype(str) == str(target["batch_id"])
            ]
            references = batch[
                batch["battery_id"].astype(str) != str(target["battery_id"])
            ].copy()
            n_available = len(references)
            row = target.to_dict()
            row["available_same_batch_reference_cells"] = int(n_available)
            audit_key = (repeat_seed, str(target["batch_id"]))
            coverage_warning = bool(coverage_warning_map.loc[audit_key])
            fold_warning = bool(fold_warning_map.loc[audit_key])
            combined_warning = bool(combined_warning_map.loc[audit_key])
            fold_audit_available = bool(fold_audit_available_map.loc[audit_key])
            if n_available < config.minimum_coverage_reference_cells:
                row.update(
                    {
                        "coverage_reference_cells": int(n_available),
                        "coverage_reference_max_distance": np.nan,
                        "coverage_matched_empirical_tail_p": np.nan,
                        "same_batch_empirical_tail_p": np.nan,
                        "oof_empirical_rarity_percentile": np.nan,
                        "candidate_resolution_available": False,
                        "strong_resolution_available": False,
                        "raw_is_oof_tail_candidate": False,
                        "raw_is_oof_strong_candidate": False,
                        "has_large_coverage_association": coverage_warning,
                        "fold_calibration_audit_available": fold_audit_available,
                        "has_large_fold_association": fold_warning,
                        "has_coverage_or_fold_confounding": combined_warning,
                        "is_oof_tail_candidate": False,
                        "is_oof_strong_candidate": False,
                        "repeat_residual_status": "insufficient_same_batch_references",
                    }
                )
                output_rows.append(row)
                continue

            distance = _coverage_distance(target, references)
            references = references.assign(_coverage_distance=distance).sort_values(
                ["_coverage_distance", "battery_id"]
            )
            n_reference = min(config.coverage_reference_cells, n_available)
            matched = references.iloc[:n_reference]
            target_score = float(target["cell_nonconformity_score"])
            matched_tail = int(
                np.sum(
                    matched["cell_nonconformity_score"].to_numpy(dtype=float)
                    >= target_score
                )
            )
            batch_tail = int(
                np.sum(
                    references["cell_nonconformity_score"].to_numpy(dtype=float)
                    >= target_score
                )
            )
            matched_p = float((1 + matched_tail) / (n_reference + 1))
            batch_p = float((1 + batch_tail) / (n_available + 1))
            candidate_resolution = (1.0 / (n_reference + 1)) <= config.candidate_p
            strong_resolution = (1.0 / (n_reference + 1)) <= config.strong_p
            raw_candidate = bool(
                candidate_resolution
                and matched_p <= config.candidate_p
                and batch_p <= config.candidate_p
            )
            raw_strong = bool(
                strong_resolution
                and matched_p <= config.strong_p
                and batch_p <= config.strong_p
            )
            safe_candidate = raw_candidate and not combined_warning
            safe_strong = raw_strong and not combined_warning
            if coverage_warning and fold_warning:
                status = "coverage_and_fold_confounding_review"
            elif coverage_warning:
                status = "coverage_confounding_review"
            elif fold_warning:
                status = "fold_calibration_confounding_review"
            elif safe_strong:
                status = "strong_oof_forecast_mismatch"
            elif safe_candidate:
                status = "oof_forecast_mismatch"
            else:
                status = "no_oof_forecast_mismatch"
            row.update(
                {
                    "coverage_reference_cells": int(n_reference),
                    "coverage_reference_max_distance": float(
                        matched["_coverage_distance"].max()
                    ),
                    "coverage_matched_empirical_tail_p": matched_p,
                    "same_batch_empirical_tail_p": batch_p,
                    "oof_empirical_rarity_percentile": 100.0 * (1.0 - matched_p),
                    "candidate_resolution_available": bool(candidate_resolution),
                    "strong_resolution_available": bool(strong_resolution),
                    "raw_is_oof_tail_candidate": raw_candidate,
                    "raw_is_oof_strong_candidate": raw_strong,
                    "has_large_coverage_association": coverage_warning,
                    "fold_calibration_audit_available": fold_audit_available,
                    "has_large_fold_association": fold_warning,
                    "has_coverage_or_fold_confounding": combined_warning,
                    "is_oof_tail_candidate": safe_candidate,
                    "is_oof_strong_candidate": safe_strong,
                    "repeat_residual_status": status,
                    "tail_rank_method": (
                        "same_batch_coverage_nearest_empirical_rank_plus_one"
                    ),
                }
            )
            output_rows.append(row)

    result = pd.DataFrame(output_rows)
    return (
        result.sort_values(["repeat_seed", "battery_id"]).reset_index(drop=True),
        coverage_audit.sort_values(["repeat_seed", "batch_id"]).reset_index(
            drop=True
        ),
    )


def aggregate_oof_physical_cells(
    scored_by_repeat: pd.DataFrame,
    *,
    expected_repeat_seeds: Sequence[int],
    required_repeat_fraction: float,
) -> pd.DataFrame:
    """Aggregate repeat-level ranks without turning missing evidence into normal."""

    working = scored_by_repeat.copy()
    # Legacy repeat tables did not contain the fold audit. Treat that evidence
    # as unavailable rather than silently assuming that no fold shift existed.
    if "fold_calibration_audit_available" not in working.columns:
        working["fold_calibration_audit_available"] = False
    if "has_large_fold_association" not in working.columns:
        working["has_large_fold_association"] = False
    if "has_coverage_or_fold_confounding" not in working.columns:
        working["has_coverage_or_fold_confounding"] = (
            working.get(
                "has_large_coverage_association",
                pd.Series(False, index=working.index),
            ).fillna(False).astype(bool)
            | working["has_large_fold_association"].fillna(False).astype(bool)
        )
    required = [
        "repeat_seed",
        "battery_id",
        "cell_id",
        "batch_id",
        "cell_nonconformity_score",
        "coverage_matched_empirical_tail_p",
        "oof_empirical_rarity_percentile",
        "is_oof_tail_candidate",
        "is_oof_strong_candidate",
        "raw_is_oof_tail_candidate",
        "has_large_coverage_association",
        "fold_calibration_audit_available",
        "has_large_fold_association",
        "has_coverage_or_fold_confounding",
        "n_common_windows",
        "common_input_cycle_end",
    ]
    _require_columns(working, required, "repeat-level OOF residual scores")
    expected = sorted(set(int(value) for value in expected_repeat_seeds))
    if not expected:
        raise ValueError("expected_repeat_seeds cannot be empty")
    if not 0 < required_repeat_fraction <= 1:
        raise ValueError("required_repeat_fraction must be in (0, 1]")
    required_count = int(math.ceil(required_repeat_fraction * len(expected)))

    rows: list[dict[str, object]] = []
    for keys, group in working.groupby(
        ["battery_id", "cell_id", "batch_id"], sort=True
    ):
        observed = sorted(set(int(value) for value in group["repeat_seed"]))
        finite_p = pd.to_numeric(
            group["coverage_matched_empirical_tail_p"], errors="coerce"
        )
        evaluated = int(finite_p.notna().sum())
        complete = observed == expected and evaluated == len(expected)
        fold_audit_count = int(
            group["fold_calibration_audit_available"].fillna(False).astype(bool).sum()
        )
        complete_fold_audit = bool(
            observed == expected and fold_audit_count == len(expected)
        )
        complete_evidence = bool(complete and complete_fold_audit)
        candidate_count = int(group["is_oof_tail_candidate"].astype(bool).sum())
        strong_count = int(group["is_oof_strong_candidate"].astype(bool).sum())
        raw_candidate_count = int(
            group["raw_is_oof_tail_candidate"].astype(bool).sum()
        )
        coverage_confounded_count = int(
            group["has_large_coverage_association"].astype(bool).sum()
        )
        fold_confounded_count = int(
            group["has_large_fold_association"].astype(bool).sum()
        )
        confounded_count = int(
            group["has_coverage_or_fold_confounding"].astype(bool).sum()
        )
        stable_candidate = bool(
            complete_evidence and candidate_count >= required_count
        )
        stable_strong = bool(complete_evidence and strong_count >= required_count)
        coverage_confounding = coverage_confounded_count >= required_count
        fold_confounding = fold_confounded_count >= required_count
        combined_confounding = confounded_count >= required_count
        if not complete:
            status = "insufficient_oof_coverage"
            confounding_type = "not_evaluated"
        elif not complete_fold_audit:
            status = "insufficient_fold_calibration_audit"
            confounding_type = "not_evaluated"
        elif coverage_confounding and fold_confounding:
            status = "coverage_and_fold_confounding"
            confounding_type = "coverage_and_fold"
        elif coverage_confounding:
            status = "coverage_confounding"
            confounding_type = "coverage"
        elif fold_confounding:
            status = "fold_calibration_confounding"
            confounding_type = "fold_calibration"
        elif combined_confounding:
            status = "mixed_coverage_or_fold_confounding"
            confounding_type = "mixed_across_repeats"
        elif stable_strong:
            status = "stable_strong_forecast_mismatch"
            confounding_type = "none"
        elif stable_candidate:
            status = "stable_forecast_mismatch"
            confounding_type = "none"
        elif raw_candidate_count or candidate_count:
            status = "unstable_forecast_mismatch"
            confounding_type = "none"
        else:
            status = "no_forecast_mismatch"
            confounding_type = "none"
        is_residual_confounding = confounding_type not in {
            "none",
            "not_evaluated",
        }
        rows.append(
            {
                "battery_id": str(keys[0]),
                "cell_id": str(keys[1]),
                "batch_id": str(keys[2]),
                "expected_oof_repeats": int(len(expected)),
                "evaluated_oof_repeats": evaluated,
                "oof_repeat_seeds": ",".join(str(value) for value in observed),
                "has_complete_oof_coverage": complete,
                "evaluated_fold_calibration_audit_repeats": fold_audit_count,
                "has_complete_fold_calibration_audit": complete_fold_audit,
                "has_complete_oof_residual_evidence": complete_evidence,
                "required_candidate_repeats": required_count,
                "median_oof_cell_nonconformity_score": float(
                    group["cell_nonconformity_score"].median()
                ),
                "median_oof_empirical_tail_p": float(finite_p.median())
                if finite_p.notna().any()
                else np.nan,
                "most_extreme_oof_empirical_tail_p": float(finite_p.min())
                if finite_p.notna().any()
                else np.nan,
                "worst_oof_empirical_tail_p": float(finite_p.max())
                if finite_p.notna().any()
                else np.nan,
                "median_oof_rarity_percentile": float(
                    group["oof_empirical_rarity_percentile"].median()
                ),
                "candidate_repeat_count": candidate_count,
                "strong_repeat_count": strong_count,
                "raw_candidate_repeat_count": raw_candidate_count,
                "coverage_confounded_repeat_count": coverage_confounded_count,
                "fold_confounded_repeat_count": fold_confounded_count,
                "confounded_repeat_count": confounded_count,
                "candidate_repeat_frequency": candidate_count / len(expected),
                "strong_repeat_frequency": strong_count / len(expected),
                "raw_candidate_repeat_frequency": raw_candidate_count / len(expected),
                "coverage_confounded_repeat_frequency": (
                    coverage_confounded_count / len(expected)
                ),
                "fold_confounded_repeat_frequency": (
                    fold_confounded_count / len(expected)
                ),
                "confounded_repeat_frequency": confounded_count / len(expected),
                "median_common_windows": float(group["n_common_windows"].median()),
                "median_common_input_cycle_end": float(
                    group["common_input_cycle_end"].median()
                ),
                "is_stable_oof_residual_candidate": stable_candidate,
                "is_stable_oof_residual_strong_candidate": stable_strong,
                "is_residual_confounding": is_residual_confounding,
                "residual_confounding_type": confounding_type,
                "residual_status": status,
                "residual_evidence_scope": (
                    "cross_fitted_with_preselected_hyperparameters"
                ),
                "residual_interpretation": (
                    "exploratory_faster_than_predicted_degradation_mismatch_"
                    "not_fault_probability"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("battery_id").reset_index(drop=True)


def _score_one_oof_fold(
    outer_predictions: pd.DataFrame,
    inner_validation_predictions: pd.DataFrame,
    *,
    config: OOFResidualConfig,
    keep_scored_outer_windows: bool,
    expected_run_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score one repeat/fold and release its window rows after summarization."""

    fold_outer_raw = _normalize_prediction_keys(
        outer_predictions, "outer OOF fold predictions"
    )
    fold_validation_raw = _normalize_prediction_keys(
        inner_validation_predictions, "inner-validation fold predictions"
    )
    _validate_prediction_provenance(
        fold_outer_raw,
        context="outer OOF fold predictions",
        expected_split="oof",
        expected_run_id=expected_run_id,
    )
    _validate_prediction_provenance(
        fold_validation_raw,
        context="inner-validation fold predictions",
        expected_split="inner_validation",
        expected_run_id=expected_run_id,
    )
    fold_outer_raw = fold_outer_raw[
        fold_outer_raw["model"].astype(str) == config.target_model
    ].copy()
    fold_validation_raw = fold_validation_raw[
        fold_validation_raw["model"].astype(str) == config.target_model
    ].copy()
    if fold_outer_raw.empty or fold_validation_raw.empty:
        raise ValueError(f"No fold predictions found for model={config.target_model!r}")

    outer_groups = set(
        map(
            tuple,
            fold_outer_raw[["repeat_seed", "outer_fold"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    validation_groups = set(
        map(
            tuple,
            fold_validation_raw[["repeat_seed", "outer_fold"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    if len(outer_groups) != 1 or outer_groups != validation_groups:
        raise ValueError(
            "Single-fold scoring requires one matching repeat/fold: "
            f"outer={sorted(outer_groups)}, validation={sorted(validation_groups)}"
        )
    repeat_seed, outer_fold = next(iter(outer_groups))
    train_ids = set(fold_validation_raw["battery_id"].astype(str))
    test_ids = set(fold_outer_raw["battery_id"].astype(str))
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise RuntimeError(
            f"repeat={repeat_seed} fold={outer_fold} validation/outer overlap: "
            f"{overlap[:10]}"
        )

    parity = _compact_prediction_parity(fold_outer_raw)
    validation_features = prepare_residual_features(
        fold_validation_raw,
        "validation",
        target_model=config.target_model,
        expected_horizons=config.expected_horizons,
    )
    outer_features = prepare_residual_features(
        fold_outer_raw,
        "test",
        target_model=config.target_model,
        expected_horizons=config.expected_horizons,
    )
    component_calibration = fit_component_calibration(validation_features)
    validation_features = apply_component_calibration(
        validation_features, component_calibration
    )
    outer_features = apply_component_calibration(
        outer_features, component_calibration
    )
    alpha_by_seed, alpha_search = select_alpha_by_seed(
        validation_features,
        alpha_grid=config.alpha_grid,
        horizons=config.alpha_selection_horizons,
        severity_aggregation="mean",
        min_common_windows=config.min_common_windows,
        min_paired_cells=config.min_paired_cells,
        bootstrap_repeats=config.bootstrap_repeats,
        random_state=(
            config.random_state
            + int(repeat_seed) * 1_000_003
            + int(outer_fold) * 10_007
        )
        % (2**32 - 1),
    )
    validation_features = add_scores_by_seed(validation_features, alpha_by_seed)
    outer_features = add_scores_by_seed(outer_features, alpha_by_seed)

    validation_horizon = summarize_common_horizon_scores(
        validation_features,
        expected_horizons=config.expected_horizons,
        min_common_windows=config.min_common_windows,
    )
    outer_horizon = summarize_common_horizon_scores(
        outer_features,
        expected_horizons=config.expected_horizons,
        min_common_windows=config.min_common_windows,
    )
    horizon_calibration = fit_horizon_severity_calibration(validation_horizon)
    outer_horizon = apply_horizon_severity_calibration(
        outer_horizon, horizon_calibration
    )
    cell_part = aggregate_cell_nonconformity(
        outer_horizon, expected_horizons=config.expected_horizons
    )
    selected_alpha = alpha_by_seed.iloc[0]
    cell_part["selected_alpha"] = float(selected_alpha["alpha"])
    cell_part["alpha_selection_status"] = str(
        selected_alpha["selection_status"]
    )
    for frame in [cell_part, outer_features, component_calibration, alpha_search]:
        frame["repeat_seed"] = int(repeat_seed)
        frame["outer_fold"] = int(outer_fold)

    fold_calibration = component_calibration.merge(
        horizon_calibration,
        on=["seed", "horizon"],
        how="outer",
        validate="one_to_one",
    ).merge(
        alpha_by_seed[["seed", "alpha", "beta", "selection_status"]],
        on="seed",
        how="left",
        validate="many_to_one",
    )
    fold_calibration["repeat_seed"] = int(repeat_seed)
    fold_calibration["outer_fold"] = int(outer_fold)
    fold_calibration["calibration_scope"] = (
        "fold_inner_validation_reused_after_early_stopping"
    )
    retained_windows = outer_features if keep_scored_outer_windows else pd.DataFrame()
    return (
        cell_part,
        fold_calibration,
        alpha_search,
        retained_windows,
        parity,
    )


def analyze_oof_residual_predictions(
    outer_predictions: pd.DataFrame,
    inner_validation_predictions: pd.DataFrame,
    *,
    config: OOFResidualConfig | None = None,
) -> OOFResidualResult:
    """Score repeated OOF predictions using fold-local calibration only."""

    config = config or OOFResidualConfig()
    _validate_config(config)
    outer = _normalize_prediction_keys(outer_predictions, "outer OOF predictions")
    validation = _normalize_prediction_keys(
        inner_validation_predictions, "inner-validation predictions"
    )
    outer = outer[outer["model"].astype(str) == config.target_model].copy()
    validation = validation[
        validation["model"].astype(str) == config.target_model
    ].copy()
    if outer.empty or validation.empty:
        raise ValueError(f"No predictions found for model={config.target_model!r}")

    outer_groups = set(
        map(tuple, outer[["repeat_seed", "outer_fold"]].drop_duplicates().to_numpy())
    )
    validation_groups = set(
        map(
            tuple,
            validation[["repeat_seed", "outer_fold"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    if outer_groups != validation_groups:
        raise ValueError(
            "Outer/inner fold mismatch: "
            f"outer={sorted(outer_groups)}, validation={sorted(validation_groups)}"
        )

    cell_parts: list[pd.DataFrame] = []
    outer_window_parts: list[pd.DataFrame] = []
    parity_parts: list[pd.DataFrame] = []
    fold_calibration_parts: list[pd.DataFrame] = []
    alpha_search_parts: list[pd.DataFrame] = []
    for repeat_seed, outer_fold in sorted(outer_groups):
        fold_outer_raw = outer[
            (outer["repeat_seed"] == repeat_seed)
            & (outer["outer_fold"] == outer_fold)
        ].copy()
        fold_validation_raw = validation[
            (validation["repeat_seed"] == repeat_seed)
            & (validation["outer_fold"] == outer_fold)
        ].copy()
        (
            cell_part,
            fold_calibration,
            alpha_search,
            retained_windows,
            parity,
        ) = _score_one_oof_fold(
            fold_outer_raw,
            fold_validation_raw,
            config=config,
            keep_scored_outer_windows=True,
        )
        cell_parts.append(cell_part)
        fold_calibration_parts.append(fold_calibration)
        alpha_search_parts.append(alpha_search)
        outer_window_parts.append(retained_windows)
        parity_parts.append(parity)

    cell_scores = pd.concat(cell_parts, ignore_index=True)
    expected_repeat_seeds = sorted(int(value) for value in outer["repeat_seed"].unique())
    expected_batteries = sorted(outer["battery_id"].astype(str).unique())
    coverage = cell_scores.groupby("repeat_seed")["battery_id"].nunique()
    if not coverage.eq(len(expected_batteries)).all():
        raise RuntimeError(
            "Every repeat must provide exactly one OOF score for every battery: "
            f"expected={len(expected_batteries)}, observed={coverage.to_dict()}"
        )

    scored_by_repeat, coverage_audit = apply_coverage_matched_tail_ranks(
        cell_scores, config=config
    )
    physical = aggregate_oof_physical_cells(
        scored_by_repeat,
        expected_repeat_seeds=expected_repeat_seeds,
        required_repeat_fraction=config.required_repeat_fraction,
    )
    if len(physical) != len(expected_batteries):
        raise RuntimeError(
            f"OOF physical summary lost batteries: {len(physical)} != {len(expected_batteries)}"
        )
    return OOFResidualResult(
        cell_score_by_repeat=scored_by_repeat,
        physical_cell_summary=physical,
        fold_calibration=pd.concat(fold_calibration_parts, ignore_index=True),
        alpha_search=pd.concat(alpha_search_parts, ignore_index=True),
        coverage_audit=coverage_audit,
        scored_outer_windows=pd.concat(outer_window_parts, ignore_index=True),
        outer_prediction_parity=_compact_prediction_parity(
            pd.concat(parity_parts, ignore_index=True)
        ),
    )


def _read_json_object(path: Path, context: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {context}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {context}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain a JSON object: {path}")
    return value


def _read_oof_chunk_predictions(
    chunk_dir: Path,
    *,
    run_id: str,
    repeat_seed: int,
    outer_fold: int,
    horizon: int,
    target_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Read and provenance-check one horizon artifact within one fold."""

    completion = _read_json_object(
        chunk_dir / "complete.json", "OOF chunk completion marker"
    )
    expected_metadata: dict[str, object] = {
        "run_id": str(run_id),
        "repeat_seed": int(repeat_seed),
        "outer_fold": int(outer_fold),
        "horizon": int(horizon),
        "model": str(target_model),
    }
    mismatches = {
        key: (completion.get(key), expected)
        for key, expected in expected_metadata.items()
        if completion.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"OOF chunk provenance mismatch in {chunk_dir}: {mismatches}"
        )

    outer_path = chunk_dir / "oof_predictions.csv"
    validation_path = chunk_dir / "inner_validation_predictions.csv"
    if not outer_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(
            f"OOF chunk is missing prediction CSVs: {chunk_dir}"
        )
    outer = pd.read_csv(outer_path)
    validation = pd.read_csv(validation_path)
    for name, frame in [("outer", outer), ("inner validation", validation)]:
        _require_columns(frame, ["horizon"], f"{name} OOF chunk")
        observed_horizons = sorted(
            pd.to_numeric(frame["horizon"], errors="raise").astype(int).unique()
        )
        if observed_horizons != [int(horizon)]:
            raise RuntimeError(
                f"{name} chunk {chunk_dir} requires horizon={horizon}, "
                f"got {observed_horizons}"
            )

    expected_outer_rows = int(completion.get("n_outer_oof_windows", len(outer)))
    expected_validation_rows = int(
        completion.get("n_inner_validation_windows", len(validation))
    )
    if len(outer) != expected_outer_rows or len(validation) != expected_validation_rows:
        raise RuntimeError(
            f"OOF chunk row-count mismatch in {chunk_dir}: "
            f"outer={len(outer)}/{expected_outer_rows}, "
            f"inner={len(validation)}/{expected_validation_rows}"
        )
    return outer, validation, len(outer), len(validation)


def analyze_oof_residual_artifacts(
    oof_output_dir: str | Path,
    *,
    config: OOFResidualConfig | None = None,
    keep_scored_outer_windows: bool = False,
) -> OOFResidualResult:
    """Analyze runner artifacts one repeat/fold at a time.

    This is the preferred API for a full 169-cell repeated OOF run.  It never
    reads the two combined, potentially very large prediction CSVs.  By
    default only fold calibration/search tables, cell summaries, coverage
    audits, and unique ``battery_id/target_cycle/actual_soh`` parity rows are
    retained.  Set ``keep_scored_outer_windows=True`` only when window-level
    diagnostics are genuinely needed and sufficient memory is available.
    """

    config = config or OOFResidualConfig()
    _validate_config(config)
    output_dir = Path(oof_output_dir).expanduser().resolve()
    completion = _read_json_object(
        output_dir / "oof_completion.json", "OOF run completion marker"
    )
    required_top_metadata = {
        "stage": "oof_cross_validation",
        "status": "complete",
        "selected_model": config.target_model,
    }
    top_mismatches = {
        key: (completion.get(key), expected)
        for key, expected in required_top_metadata.items()
        if completion.get(key) != expected
    }
    if top_mismatches:
        raise RuntimeError(
            f"OOF run completion metadata mismatch in {output_dir}: "
            f"{top_mismatches}"
        )
    run_id = str(completion.get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError("OOF run completion marker has no run_id")
    repeat_seeds = sorted(
        set(int(value) for value in completion.get("repeat_seeds", []))
    )
    if not repeat_seeds:
        raise RuntimeError("OOF run completion marker has no repeat seeds")
    n_splits = int(completion.get("n_splits", 0))
    if n_splits < 2:
        raise RuntimeError(f"OOF run requires at least two folds, got {n_splits}")
    marker_horizons = tuple(
        sorted(int(value) for value in completion.get("horizons", []))
    )
    expected_horizons = tuple(sorted(int(value) for value in config.expected_horizons))
    if marker_horizons != expected_horizons:
        raise RuntimeError(
            f"OOF run horizons={marker_horizons}, expected={expected_horizons}"
        )
    expected_battery_count = int(completion.get("oof_unique_batteries", 0))
    if expected_battery_count < 1:
        raise RuntimeError("OOF run completion marker has no battery count")
    eligible_by_horizon = completion.get("eligible_batteries_by_horizon")
    if eligible_by_horizon is not None:
        observed_horizon_counts = {
            int(horizon): int(count)
            for horizon, count in dict(eligible_by_horizon).items()
        }
        expected_horizon_counts = {
            int(horizon): expected_battery_count for horizon in expected_horizons
        }
        if observed_horizon_counts != expected_horizon_counts:
            raise RuntimeError(
                "Every residual horizon must cover the same full OOF cohort; "
                f"observed={observed_horizon_counts}, "
                f"expected={expected_horizon_counts}"
            )

    cell_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    alpha_parts: list[pd.DataFrame] = []
    parity_parts: list[pd.DataFrame] = []
    retained_window_parts: list[pd.DataFrame] = []
    observed_outer_rows = 0
    observed_validation_rows = 0
    artifact_root = output_dir / "fold_artifacts"
    for repeat_seed in repeat_seeds:
        for outer_fold in range(n_splits):
            fold_outer_parts: list[pd.DataFrame] = []
            fold_validation_parts: list[pd.DataFrame] = []
            for horizon in expected_horizons:
                chunk_dir = (
                    artifact_root
                    / f"seed{repeat_seed}"
                    / f"fold{outer_fold}"
                    / f"horizon{horizon}"
                )
                outer_chunk, validation_chunk, n_outer, n_validation = (
                    _read_oof_chunk_predictions(
                        chunk_dir,
                        run_id=run_id,
                        repeat_seed=repeat_seed,
                        outer_fold=outer_fold,
                        horizon=horizon,
                        target_model=config.target_model,
                    )
                )
                fold_outer_parts.append(outer_chunk)
                fold_validation_parts.append(validation_chunk)
                observed_outer_rows += n_outer
                observed_validation_rows += n_validation

            fold_outer = pd.concat(fold_outer_parts, ignore_index=True)
            fold_validation = pd.concat(fold_validation_parts, ignore_index=True)
            (
                cell_part,
                fold_calibration,
                alpha_search,
                retained_windows,
                parity,
            ) = _score_one_oof_fold(
                fold_outer,
                fold_validation,
                config=config,
                keep_scored_outer_windows=keep_scored_outer_windows,
                expected_run_id=run_id,
            )
            cell_parts.append(cell_part)
            calibration_parts.append(fold_calibration)
            alpha_parts.append(alpha_search)
            parity_parts.append(parity)
            if keep_scored_outer_windows:
                retained_window_parts.append(retained_windows)

    expected_outer_rows = int(
        completion.get("oof_prediction_rows", observed_outer_rows)
    )
    expected_validation_rows = int(
        completion.get(
            "inner_validation_prediction_rows", observed_validation_rows
        )
    )
    if (
        observed_outer_rows != expected_outer_rows
        or observed_validation_rows != expected_validation_rows
    ):
        raise RuntimeError(
            "Fold artifacts do not match run-level prediction row counts: "
            f"outer={observed_outer_rows}/{expected_outer_rows}, "
            f"inner={observed_validation_rows}/{expected_validation_rows}"
        )

    cell_scores = pd.concat(cell_parts, ignore_index=True)
    repeat_batteries = {
        int(seed): set(group["battery_id"].astype(str))
        for seed, group in cell_scores.groupby("repeat_seed", sort=True)
    }
    if set(repeat_batteries) != set(repeat_seeds):
        raise RuntimeError(
            f"OOF score repeats differ from completion marker: "
            f"{sorted(repeat_batteries)} != {repeat_seeds}"
        )
    first_batteries = repeat_batteries[repeat_seeds[0]]
    coverage_problem = {
        seed: len(batteries)
        for seed, batteries in repeat_batteries.items()
        if batteries != first_batteries
        or len(batteries) != expected_battery_count
    }
    if coverage_problem:
        raise RuntimeError(
            "Every repeat must contain the same complete physical-cell set: "
            f"expected={expected_battery_count}, observed={coverage_problem}"
        )

    scored_by_repeat, coverage_audit = apply_coverage_matched_tail_ranks(
        cell_scores, config=config
    )
    physical = aggregate_oof_physical_cells(
        scored_by_repeat,
        expected_repeat_seeds=repeat_seeds,
        required_repeat_fraction=config.required_repeat_fraction,
    )
    if len(physical) != expected_battery_count:
        raise RuntimeError(
            f"OOF physical summary lost batteries: "
            f"{len(physical)} != {expected_battery_count}"
        )
    scored_windows = (
        pd.concat(retained_window_parts, ignore_index=True)
        if keep_scored_outer_windows
        else pd.DataFrame()
    )
    return OOFResidualResult(
        cell_score_by_repeat=scored_by_repeat,
        physical_cell_summary=physical,
        fold_calibration=pd.concat(calibration_parts, ignore_index=True),
        alpha_search=pd.concat(alpha_parts, ignore_index=True),
        coverage_audit=coverage_audit,
        scored_outer_windows=scored_windows,
        outer_prediction_parity=_compact_prediction_parity(
            pd.concat(parity_parts, ignore_index=True)
        ),
    )


def _functional_persistent_flag(frame: pd.DataFrame) -> pd.Series:
    if "is_persistent_review_candidate" in frame.columns:
        return frame["is_persistent_review_candidate"].fillna(False).astype(bool)
    columns = [
        column
        for column in [
            "is_shape_candidate",
            "is_absolute_pattern_candidate",
            "is_lifetime_candidate",
            "is_shape_rare_group_candidate",
            "is_absolute_rare_group_candidate",
            "is_lifetime_rare_group_candidate",
        ]
        if column in frame.columns
    ]
    if not columns:
        raise KeyError("Functional summary has no persistent evidence columns")
    return frame[columns].fillna(False).astype(bool).any(axis=1)


def _continuous_rank_agreement_audit(
    eligible: pd.DataFrame,
    *,
    permutation_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """Compare continuous functional and OOF ranks within protocol batches."""

    score_columns = ("shape_score", "absolute_pattern_score", "lifetime_score")
    oof_column = "median_oof_rarity_percentile"
    rows: list[dict[str, object]] = []
    for column_index, functional_column in enumerate(score_columns):
        base: dict[str, object] = {
            "functional_score": functional_column,
            "oof_score": oof_column,
            "rank_agreement_method": (
                "pooled_pearson_correlation_of_within_batch_average_percent_ranks"
            ),
            "permutation_repeats": int(permutation_repeats),
            "null_scope": "oof_rarity_ranks_permuted_within_batch",
            "interpretation": (
                "continuous_method_agreement_not_anomaly_ground_truth_validation"
            ),
        }
        if functional_column not in eligible.columns:
            rows.append(
                {
                    **base,
                    "rank_agreement_available": False,
                    "availability_status": "functional_score_column_missing",
                    "eligible_cells": 0,
                    "eligible_batches": 0,
                    "within_batch_rank_correlation": np.nan,
                    "permutation_expected_correlation": np.nan,
                    "permutation_p_greater_equal": np.nan,
                    "permutation_p_abs_greater_equal": np.nan,
                }
            )
            continue

        working = eligible[["batch_id", functional_column, oof_column]].copy()
        working[functional_column] = pd.to_numeric(
            working[functional_column], errors="coerce"
        )
        working[oof_column] = pd.to_numeric(working[oof_column], errors="coerce")
        working = working[
            np.isfinite(working[functional_column])
            & np.isfinite(working[oof_column])
        ].copy()
        batch_sizes = working.groupby("batch_id").size()
        usable_batches = set(batch_sizes[batch_sizes >= 3].index.astype(str))
        working = working[
            working["batch_id"].astype(str).isin(usable_batches)
        ].copy()
        if len(working) < 3 or not usable_batches:
            rows.append(
                {
                    **base,
                    "rank_agreement_available": False,
                    "availability_status": "insufficient_finite_cells_within_batch",
                    "eligible_cells": int(len(working)),
                    "eligible_batches": int(len(usable_batches)),
                    "within_batch_rank_correlation": np.nan,
                    "permutation_expected_correlation": np.nan,
                    "permutation_p_greater_equal": np.nan,
                    "permutation_p_abs_greater_equal": np.nan,
                }
            )
            continue

        working["_functional_rank"] = working.groupby("batch_id")[
            functional_column
        ].rank(method="average", pct=True)
        working["_oof_rank"] = working.groupby("batch_id")[oof_column].rank(
            method="average", pct=True
        )
        functional_rank = working["_functional_rank"].to_numpy(dtype=float)
        oof_rank = working["_oof_rank"].to_numpy(dtype=float)
        observed = float(pd.Series(functional_rank).corr(pd.Series(oof_rank)))
        available = bool(np.isfinite(observed))
        if not available:
            rows.append(
                {
                    **base,
                    "rank_agreement_available": False,
                    "availability_status": "constant_within_batch_ranks",
                    "eligible_cells": int(len(working)),
                    "eligible_batches": int(len(usable_batches)),
                    "within_batch_rank_correlation": np.nan,
                    "permutation_expected_correlation": np.nan,
                    "permutation_p_greater_equal": np.nan,
                    "permutation_p_abs_greater_equal": np.nan,
                }
            )
            continue

        rng = np.random.default_rng(
            int(random_state) + (column_index + 1) * 1_000_003
        )
        strata = working["batch_id"].astype(str).to_numpy()
        permuted_correlations = np.empty(int(permutation_repeats), dtype=float)
        for repeat in range(int(permutation_repeats)):
            permuted_oof = oof_rank.copy()
            for batch_id in np.unique(strata):
                indices = np.flatnonzero(strata == batch_id)
                permuted_oof[indices] = rng.permutation(permuted_oof[indices])
            permuted_correlations[repeat] = float(
                pd.Series(functional_rank).corr(pd.Series(permuted_oof))
            )
        finite_permuted = permuted_correlations[
            np.isfinite(permuted_correlations)
        ]
        if len(finite_permuted) != int(permutation_repeats):
            raise RuntimeError(
                f"Non-finite rank permutation statistics for {functional_column}"
            )
        rows.append(
            {
                **base,
                "rank_agreement_available": True,
                "availability_status": "available",
                "eligible_cells": int(len(working)),
                "eligible_batches": int(len(usable_batches)),
                "within_batch_rank_correlation": observed,
                "permutation_expected_correlation": float(
                    np.mean(finite_permuted)
                ),
                "permutation_p_greater_equal": float(
                    (1 + np.sum(finite_permuted >= observed))
                    / (len(finite_permuted) + 1)
                ),
                "permutation_p_abs_greater_equal": float(
                    (1 + np.sum(np.abs(finite_permuted) >= abs(observed)))
                    / (len(finite_permuted) + 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_functional_and_oof_evidence(
    functional_summary: pd.DataFrame,
    oof_physical_summary: pd.DataFrame,
    *,
    expected_battery_count: int | None = None,
    permutation_repeats: int = 2_000,
    random_state: int = 20260715,
) -> FunctionalOOFComparisonResult:
    """Join complementary SOH-derived views without inventing a truth label."""

    keys = ["battery_id", "cell_id", "batch_id"]
    _require_columns(functional_summary, keys, "functional summary")
    _require_columns(
        oof_physical_summary,
        keys
        + [
            "residual_status",
            "has_complete_oof_coverage",
            "is_stable_oof_residual_candidate",
            "is_stable_oof_residual_strong_candidate",
            "median_oof_rarity_percentile",
            "candidate_repeat_frequency",
        ],
        "OOF residual summary",
    )
    for name, frame in [
        ("functional summary", functional_summary),
        ("OOF residual summary", oof_physical_summary),
    ]:
        if frame["battery_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} must contain one row per battery")

    functional = functional_summary.copy()
    functional["is_persistent_review_candidate"] = _functional_persistent_flag(
        functional
    )
    metadata_check = functional[keys].merge(
        oof_physical_summary[keys],
        on="battery_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_functional", "_oof"),
        indicator=True,
    )
    metadata_problem = (
        metadata_check["_merge"].ne("both")
        | metadata_check["cell_id_functional"].astype(str).ne(
            metadata_check["cell_id_oof"].astype(str)
        )
        | metadata_check["batch_id_functional"].astype(str).ne(
            metadata_check["batch_id_oof"].astype(str)
        )
    )
    if metadata_problem.any():
        raise RuntimeError(
            "Functional/OOF battery coverage or metadata differs: "
            + metadata_check.loc[metadata_problem]
            .head(20)
            .to_dict("records")
            .__repr__()
        )
    comparison = functional.merge(
        oof_physical_summary,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if expected_battery_count is not None and len(comparison) != int(
        expected_battery_count
    ):
        raise RuntimeError(
            f"Comparison requires {expected_battery_count} batteries, got {len(comparison)}"
        )
    mismatched = comparison["_merge"] != "both"
    if mismatched.any():
        raise RuntimeError(
            "Functional/OOF battery coverage differs: "
            + comparison.loc[mismatched, ["battery_id", "_merge"]]
            .head(20)
            .to_dict("records")
            .__repr__()
        )
    comparison = comparison.drop(columns="_merge")

    availability_column = (
        "has_complete_oof_residual_evidence"
        if "has_complete_oof_residual_evidence" in comparison.columns
        else "has_complete_oof_coverage"
    )
    residual_available = comparison[availability_column].fillna(False).astype(bool)
    if "is_residual_confounding" in comparison.columns:
        residual_confounded = comparison["is_residual_confounding"].fillna(
            False
        ).astype(bool)
    else:
        # Accept both the legacy generic status and the explicit new statuses.
        residual_confounded = comparison["residual_status"].isin(
            {
                "coverage_or_fold_confounding",
                "coverage_confounding",
                "fold_calibration_confounding",
                "coverage_and_fold_confounding",
                "mixed_coverage_or_fold_confounding",
            }
        )
    residual_candidate = comparison[
        "is_stable_oof_residual_candidate"
    ].fillna(False).astype(bool)
    persistent = comparison["is_persistent_review_candidate"].astype(bool)
    transient = comparison.get(
        "is_transient_only_candidate", pd.Series(False, index=comparison.index)
    ).fillna(False).astype(bool)

    status = pd.Series("no_joint_evidence", index=comparison.index, dtype=object)
    status.loc[transient & ~persistent & ~residual_candidate] = "transient_qc_only"
    status.loc[residual_candidate & ~persistent] = "forecast_only_review"
    status.loc[persistent & ~residual_candidate] = "functional_only_review"
    status.loc[persistent & residual_candidate] = (
        "concordant_persistent_and_forecast"
    )
    status.loc[residual_confounded & ~persistent] = "comparison_confounded"
    status.loc[residual_confounded & persistent] = (
        "functional_primary_residual_confounded"
    )
    status.loc[~residual_available & ~persistent] = "comparison_unavailable"
    status.loc[~residual_available & persistent] = (
        "functional_primary_residual_unavailable"
    )
    comparison["combined_evidence_status"] = status
    priority_map = {
        "concordant_persistent_and_forecast": "R1_concordant",
        "functional_only_review": "R2_functional_primary",
        "forecast_only_review": "R3_forecast_auxiliary",
        "transient_qc_only": "QC_transient",
        "comparison_confounded": "QC_confounded",
        "comparison_unavailable": "NA_unavailable",
        "functional_primary_residual_confounded": (
            "R2_functional_primary_residual_confounded"
        ),
        "functional_primary_residual_unavailable": (
            "R2_functional_primary_residual_unavailable"
        ),
        "no_joint_evidence": "no_joint_evidence",
    }
    comparison["combined_review_priority"] = status.map(priority_map)
    comparison["comparison_interpretation"] = (
        "agreement_between_two_soh_derived_views_not_ground_truth"
    )

    eligible = comparison[residual_available & ~residual_confounded].copy()
    batch_rows: list[dict[str, object]] = []
    for batch_id, group in eligible.groupby("batch_id", sort=True):
        f = group["is_persistent_review_candidate"].astype(bool)
        r = group["is_stable_oof_residual_candidate"].astype(bool)
        batch_rows.append(
            {
                "batch_id": str(batch_id),
                "eligible_cells": int(len(group)),
                "both": int((f & r).sum()),
                "functional_only": int((f & ~r).sum()),
                "residual_only": int((~f & r).sum()),
                "neither": int((~f & ~r).sum()),
                "functional_candidate_rate": float(f.mean()),
                "residual_candidate_rate": float(r.mean()),
                "agreement_rate": float((f == r).mean()),
            }
        )
    by_batch = pd.DataFrame(batch_rows)

    if permutation_repeats < 100:
        raise ValueError("permutation_repeats must be at least 100")
    observed_f = eligible["is_persistent_review_candidate"].to_numpy(dtype=bool)
    observed_r = eligible["is_stable_oof_residual_candidate"].to_numpy(dtype=bool)
    observed_overlap = int(np.sum(observed_f & observed_r))
    union = int(np.sum(observed_f | observed_r))
    rng = np.random.default_rng(int(random_state))
    permuted_overlap = np.zeros(permutation_repeats, dtype=int)
    strata = eligible["batch_id"].astype(str).to_numpy()
    for repeat in range(permutation_repeats):
        permuted = observed_r.copy()
        for batch_id in np.unique(strata):
            indices = np.flatnonzero(strata == batch_id)
            permuted[indices] = rng.permutation(permuted[indices])
        permuted_overlap[repeat] = int(np.sum(observed_f & permuted))
    expected_overlap = float(np.mean(permuted_overlap))
    overlap_audit = pd.DataFrame(
        [
            {
                "eligible_cells": int(len(eligible)),
                "functional_candidates": int(observed_f.sum()),
                "oof_residual_candidates": int(observed_r.sum()),
                "observed_overlap": observed_overlap,
                "expected_overlap_under_batch_preserving_permutation": expected_overlap,
                "overlap_enrichment": observed_overlap / max(expected_overlap, 1e-12),
                "jaccard_index": observed_overlap / max(union, 1),
                "permutation_p_greater_equal": float(
                    (1 + np.sum(permuted_overlap >= observed_overlap))
                    / (permutation_repeats + 1)
                ),
                "permutation_repeats": int(permutation_repeats),
                "null_scope": "residual_flags_permuted_within_batch",
                "interpretation": (
                    "method_agreement_test_not_anomaly_ground_truth_validation"
                ),
            }
        ]
    )
    rank_agreement_audit = _continuous_rank_agreement_audit(
        eligible,
        permutation_repeats=permutation_repeats,
        random_state=random_state,
    )
    return FunctionalOOFComparisonResult(
        cell_comparison=comparison.sort_values("battery_id").reset_index(drop=True),
        comparison_by_batch=by_batch,
        overlap_audit=overlap_audit,
        rank_agreement_audit=rank_agreement_audit,
    )
