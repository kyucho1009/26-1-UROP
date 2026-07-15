from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.matr_oof_residual_analysis import (
    OOFResidualConfig,
    aggregate_oof_physical_cells,
    analyze_oof_residual_artifacts,
    analyze_oof_residual_predictions,
    apply_coverage_matched_tail_ranks,
    compare_functional_and_oof_evidence,
)
from scripts.run_matr_oof_cross_validation import (
    ROLE_OUTER,
    ROLE_TRAIN,
    ROLE_VALIDATION,
    assign_stratified_folds,
    split_inner_train_validation,
)


def make_repeat_scores(*, repeats: tuple[int, ...] = (42, 43, 44)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for repeat_seed in repeats:
        for index in range(24):
            rows.append(
                {
                    "repeat_seed": repeat_seed,
                    "outer_fold": index % 5,
                    "battery_id": f"cell_{index:02d}",
                    "cell_id": f"cell_{index:02d}",
                    "batch_id": "b1",
                    "cell_nonconformity_score": float(index),
                    "n_common_windows": 100 + index,
                    "common_input_cycle_start": 10.0,
                    "common_input_cycle_end": float(109 + index),
                    "common_input_cycle_span": float(99 + index),
                }
            )
    return pd.DataFrame(rows)


def make_fold_shift_scores(
    *,
    shifted: bool,
    repeats: tuple[int, ...] = (42,),
) -> pd.DataFrame:
    """Make identical coverage with either aligned or fold-shifted score ranks."""

    rows: list[dict[str, object]] = []
    for repeat_seed in repeats:
        for outer_fold in range(5):
            for within_fold in range(10):
                battery_id = f"cell_f{outer_fold}_{within_fold:02d}"
                score = (
                    float(100 * outer_fold + within_fold)
                    if shifted
                    else float(within_fold)
                )
                rows.append(
                    {
                        "repeat_seed": repeat_seed,
                        "outer_fold": outer_fold,
                        "battery_id": battery_id,
                        "cell_id": battery_id,
                        "batch_id": "b1",
                        "cell_nonconformity_score": score,
                        "n_common_windows": 100,
                        "common_input_cycle_start": 10.0,
                        "common_input_cycle_end": 109.0,
                        "common_input_cycle_span": 99.0,
                    }
                )
    return pd.DataFrame(rows)


def make_synthetic_fold_predictions(
    *,
    repeat_seed: int,
    outer_fold: int,
    horizon: int,
    battery_indices: range,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for battery_index in battery_indices:
        battery_id = f"cell_{battery_index:02d}"
        degradation_rate = 0.00045 + 0.00001 * battery_index
        for window_index, input_end_cycle in enumerate(range(20, 26)):
            target_cycle = input_end_cycle + horizon
            current_soh = 1.0 - degradation_rate * input_end_cycle
            actual_soh = 1.0 - degradation_rate * target_cycle
            residual_bias = (
                0.00002
                * (battery_index + 1)
                * (1.0 + horizon / 100.0)
                * (1.0 + 0.08 * window_index)
            )
            pred_soh = actual_soh + residual_bias
            rows.append(
                {
                    "repeat_seed": repeat_seed,
                    "seed": repeat_seed,
                    "outer_fold": outer_fold,
                    "model": "cpmlp_cpdsconv_fusion",
                    "battery_id": battery_id,
                    "cell_id": battery_id,
                    "batch_id": "b1",
                    "horizon": horizon,
                    "input_end_cycle": input_end_cycle,
                    "target_cycle": target_cycle,
                    "actual_soh": actual_soh,
                    "current_soh": current_soh,
                    "pred_soh": pred_soh,
                    "actual_delta_soh": current_soh - actual_soh,
                    "pred_delta_soh": current_soh - pred_soh,
                    "sample_mode": "sliding-window",
                    "split": split,
                    "stage": "oof_cross_validation",
                    "normalizer_fit_role": "inner_train",
                }
            )
    return pd.DataFrame(rows)


class OOFResidualAnalysisTest(unittest.TestCase):
    def test_fold_artifact_streaming_matches_in_memory_analysis(self):
        config = OOFResidualConfig(
            alpha_grid=(0.0, 0.5, 1.0),
            min_common_windows=3,
            min_paired_cells=3,
            bootstrap_repeats=100,
            coverage_reference_cells=5,
            minimum_coverage_reference_cells=2,
            coverage_warning_abs_rank_corr=1.0,
            fold_permutation_repeats=100,
            minimum_fold_cells=2,
        )
        repeat_seed = 42
        horizons = (10, 50, 100)
        all_outer: list[pd.DataFrame] = []
        all_validation: list[pd.DataFrame] = []
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            total_outer_rows = 0
            total_validation_rows = 0
            for outer_fold in range(2):
                outer_indices = range(0, 5) if outer_fold == 0 else range(5, 10)
                validation_indices = (
                    range(5, 10) if outer_fold == 0 else range(0, 5)
                )
                for horizon in horizons:
                    outer = make_synthetic_fold_predictions(
                        repeat_seed=repeat_seed,
                        outer_fold=outer_fold,
                        horizon=horizon,
                        battery_indices=outer_indices,
                        split="oof",
                    )
                    validation = make_synthetic_fold_predictions(
                        repeat_seed=repeat_seed,
                        outer_fold=outer_fold,
                        horizon=horizon,
                        battery_indices=validation_indices,
                        split="inner_validation",
                    )
                    all_outer.append(outer)
                    all_validation.append(validation)
                    total_outer_rows += len(outer)
                    total_validation_rows += len(validation)
                    chunk_dir = (
                        output_dir
                        / "fold_artifacts"
                        / f"seed{repeat_seed}"
                        / f"fold{outer_fold}"
                        / f"horizon{horizon}"
                    )
                    chunk_dir.mkdir(parents=True)
                    outer.to_csv(chunk_dir / "oof_predictions.csv", index=False)
                    validation.to_csv(
                        chunk_dir / "inner_validation_predictions.csv", index=False
                    )
                    (chunk_dir / "complete.json").write_text(
                        json.dumps(
                            {
                                "run_id": "synthetic-run",
                                "repeat_seed": repeat_seed,
                                "outer_fold": outer_fold,
                                "horizon": horizon,
                                "model": config.target_model,
                                "n_outer_oof_windows": len(outer),
                                "n_inner_validation_windows": len(validation),
                            }
                        ),
                        encoding="utf-8",
                    )
            (output_dir / "oof_completion.json").write_text(
                json.dumps(
                    {
                        "run_id": "synthetic-run",
                        "stage": "oof_cross_validation",
                        "status": "complete",
                        "selected_model": config.target_model,
                        "repeat_seeds": [repeat_seed],
                        "n_splits": 2,
                        "horizons": list(horizons),
                        "oof_unique_batteries": 10,
                        "oof_prediction_rows": total_outer_rows,
                        "inner_validation_prediction_rows": total_validation_rows,
                    }
                ),
                encoding="utf-8",
            )

            in_memory = analyze_oof_residual_predictions(
                pd.concat(all_outer, ignore_index=True),
                pd.concat(all_validation, ignore_index=True),
                config=config,
            )
            streamed = analyze_oof_residual_artifacts(
                output_dir,
                config=config,
            )

        self.assertTrue(streamed.scored_outer_windows.empty)
        for left, right, sort_columns in [
            (
                in_memory.cell_score_by_repeat,
                streamed.cell_score_by_repeat,
                ["repeat_seed", "battery_id"],
            ),
            (
                in_memory.physical_cell_summary,
                streamed.physical_cell_summary,
                ["battery_id"],
            ),
            (
                in_memory.fold_calibration,
                streamed.fold_calibration,
                ["repeat_seed", "outer_fold", "horizon"],
            ),
            (
                in_memory.alpha_search,
                streamed.alpha_search,
                ["repeat_seed", "outer_fold", "alpha"],
            ),
            (
                in_memory.coverage_audit,
                streamed.coverage_audit,
                ["repeat_seed", "batch_id"],
            ),
            (
                in_memory.outer_prediction_parity,
                streamed.outer_prediction_parity,
                ["battery_id", "target_cycle"],
            ),
        ]:
            pd.testing.assert_frame_equal(
                left.sort_values(sort_columns).reset_index(drop=True),
                right.sort_values(sort_columns).reset_index(drop=True),
                check_dtype=False,
            )

    def test_battery_folds_are_exhaustive_batch_and_length_balanced(self):
        rows = []
        for batch_id, count in {"b1": 41, "b2": 43, "b3": 40, "b4": 45}.items():
            for index in range(count):
                rows.append(
                    {
                        "battery_id": f"{batch_id}_cell_{index:02d}",
                        "cell_id": f"{batch_id}_cell_{index:02d}",
                        "batch_id": batch_id,
                        "observed_cycle_end": 150 + 7 * index + (index % 3),
                    }
                )
        cells = pd.DataFrame(rows)
        assignment = assign_stratified_folds(cells, n_splits=5, seed=42)
        repeated = assign_stratified_folds(
            cells.sample(frac=1.0, random_state=99), n_splits=5, seed=42
        )
        self.assertEqual(len(assignment), 169)
        self.assertFalse(assignment["battery_id"].duplicated().any())
        pd.testing.assert_series_equal(
            assignment.set_index("battery_id")["outer_fold"].sort_index(),
            repeated.set_index("battery_id")["outer_fold"].sort_index(),
        )
        per_batch = assignment.groupby(["batch_id", "outer_fold"]).size().unstack()
        self.assertTrue((per_batch.max(axis=1) - per_batch.min(axis=1)).le(1).all())
        complete_strata = assignment.groupby(
            ["batch_id", "observed_length_stratum"]
        ).filter(lambda group: len(group) == 5)
        self.assertTrue(
            complete_strata.groupby(
                ["batch_id", "observed_length_stratum"]
            )["outer_fold"].nunique().eq(5).all()
        )

        roles = split_inner_train_validation(
            assignment,
            outer_fold=0,
            validation_fraction=0.20,
            seed=123,
        )
        train = set(roles[ROLE_TRAIN])
        validation = set(roles[ROLE_VALIDATION])
        outer = set(roles[ROLE_OUTER])
        self.assertFalse(train & validation)
        self.assertFalse(train & outer)
        self.assertFalse(validation & outer)
        self.assertEqual(train | validation | outer, set(cells["battery_id"]))

    def test_ties_do_not_force_tail_candidate(self):
        frame = make_repeat_scores(repeats=(42,))
        frame["cell_nonconformity_score"] = 1.0
        config = OOFResidualConfig(
            coverage_reference_cells=20,
            minimum_coverage_reference_cells=9,
            coverage_warning_abs_rank_corr=0.50,
        )
        scored, audit = apply_coverage_matched_tail_ranks(frame, config=config)
        self.assertTrue(scored["coverage_matched_empirical_tail_p"].eq(1.0).all())
        self.assertFalse(scored["raw_is_oof_tail_candidate"].any())
        self.assertFalse(scored["is_oof_tail_candidate"].any())
        self.assertFalse(audit["has_large_coverage_association"].any())

    def test_coverage_association_is_a_candidate_safety_gate(self):
        frame = make_repeat_scores(repeats=(42,))
        config = OOFResidualConfig(
            coverage_reference_cells=20,
            minimum_coverage_reference_cells=9,
            coverage_warning_abs_rank_corr=0.50,
        )
        scored, audit = apply_coverage_matched_tail_ranks(frame, config=config)
        extreme = scored.set_index("battery_id").loc["cell_23"]
        self.assertTrue(bool(extreme["raw_is_oof_tail_candidate"]))
        self.assertTrue(bool(extreme["has_large_coverage_association"]))
        self.assertFalse(bool(extreme["is_oof_tail_candidate"]))
        self.assertEqual(extreme["repeat_residual_status"], "coverage_confounding_review")
        self.assertTrue(bool(audit.iloc[0]["has_large_coverage_association"]))

    def test_repeated_confounding_is_not_reported_as_normal(self):
        frame = make_repeat_scores()
        config = OOFResidualConfig(
            coverage_reference_cells=20,
            minimum_coverage_reference_cells=9,
            coverage_warning_abs_rank_corr=0.50,
        )
        scored, _ = apply_coverage_matched_tail_ranks(frame, config=config)
        physical = aggregate_oof_physical_cells(
            scored,
            expected_repeat_seeds=(42, 43, 44),
            required_repeat_fraction=2.0 / 3.0,
        ).set_index("battery_id")
        self.assertEqual(
            physical.loc["cell_23", "residual_status"],
            "coverage_confounding",
        )
        self.assertEqual(
            int(physical.loc["cell_23", "coverage_confounded_repeat_count"]), 3
        )
        self.assertEqual(
            int(physical.loc["cell_23", "fold_confounded_repeat_count"]), 0
        )
        self.assertFalse(
            bool(physical.loc["cell_23", "is_stable_oof_residual_candidate"])
        )

    def test_fold_rank_shift_is_deterministic_and_gates_raw_candidates(self):
        config = OOFResidualConfig(
            coverage_reference_cells=20,
            minimum_coverage_reference_cells=9,
            fold_permutation_repeats=200,
            fold_warning_adjusted_rank_effect=0.10,
            fold_warning_permutation_p=0.05,
        )
        aligned, aligned_audit = apply_coverage_matched_tail_ranks(
            make_fold_shift_scores(shifted=False), config=config
        )
        shifted, shifted_audit = apply_coverage_matched_tail_ranks(
            make_fold_shift_scores(shifted=True), config=config
        )
        _, repeated_audit = apply_coverage_matched_tail_ranks(
            make_fold_shift_scores(shifted=True), config=config
        )

        self.assertTrue(bool(aligned_audit.iloc[0]["fold_calibration_audit_available"]))
        self.assertFalse(bool(aligned_audit.iloc[0]["has_large_fold_association"]))
        self.assertTrue(bool(shifted_audit.iloc[0]["has_large_fold_association"]))
        self.assertFalse(bool(shifted_audit.iloc[0]["has_large_coverage_association"]))
        self.assertGreater(
            float(shifted_audit.iloc[0]["fold_rank_adjusted_eta_squared"]), 0.8
        )
        self.assertLessEqual(
            float(shifted_audit.iloc[0]["fold_permutation_p_greater_equal"]), 0.05
        )
        self.assertEqual(
            float(shifted_audit.iloc[0]["fold_permutation_p_greater_equal"]),
            float(repeated_audit.iloc[0]["fold_permutation_p_greater_equal"]),
        )
        self.assertTrue(shifted["has_coverage_or_fold_confounding"].all())
        self.assertTrue(shifted["raw_is_oof_tail_candidate"].any())
        self.assertFalse(shifted["is_oof_tail_candidate"].any())
        self.assertFalse(aligned["has_coverage_or_fold_confounding"].any())

    def test_repeated_fold_shift_has_an_explicit_physical_status(self):
        config = OOFResidualConfig(
            coverage_reference_cells=20,
            minimum_coverage_reference_cells=9,
            fold_permutation_repeats=200,
        )
        scored, _ = apply_coverage_matched_tail_ranks(
            make_fold_shift_scores(
                shifted=True,
                repeats=(42, 43, 44),
            ),
            config=config,
        )
        physical = aggregate_oof_physical_cells(
            scored,
            expected_repeat_seeds=(42, 43, 44),
            required_repeat_fraction=2.0 / 3.0,
        )
        self.assertTrue(physical["has_complete_fold_calibration_audit"].all())
        self.assertTrue(physical["is_residual_confounding"].all())
        self.assertTrue(
            physical["residual_status"].eq("fold_calibration_confounding").all()
        )
        self.assertTrue(physical["fold_confounded_repeat_count"].eq(3).all())
        self.assertTrue(physical["coverage_confounded_repeat_count"].eq(0).all())

    def test_comparison_keeps_confounding_and_unavailability_explicit(self):
        ids = [f"cell_{index}" for index in range(7)]
        functional = pd.DataFrame(
            {
                "battery_id": ids,
                "cell_id": ids,
                "batch_id": ["b1", "b1", "b1", "b2", "b2", "b2", "b2"],
                "is_persistent_review_candidate": [
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    False,
                ],
                "is_transient_only_candidate": [
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ],
            }
        )
        residual = pd.DataFrame(
            {
                "battery_id": ids,
                "cell_id": ids,
                "batch_id": functional["batch_id"],
                "residual_status": [
                    "stable_forecast_mismatch",
                    "no_forecast_mismatch",
                    "stable_forecast_mismatch",
                    "no_forecast_mismatch",
                    "coverage_or_fold_confounding",
                    "insufficient_oof_coverage",
                    "no_forecast_mismatch",
                ],
                "has_complete_oof_coverage": [True, True, True, True, True, False, True],
                "is_stable_oof_residual_candidate": [True, False, True, False, False, False, False],
                "is_stable_oof_residual_strong_candidate": [False] * 7,
                "median_oof_rarity_percentile": np.linspace(99, 50, 7),
                "candidate_repeat_frequency": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        result = compare_functional_and_oof_evidence(
            functional,
            residual,
            expected_battery_count=7,
            permutation_repeats=100,
            random_state=7,
        ).cell_comparison.set_index("battery_id")
        self.assertEqual(
            result.loc["cell_0", "combined_evidence_status"],
            "concordant_persistent_and_forecast",
        )
        self.assertEqual(
            result.loc["cell_1", "combined_evidence_status"],
            "functional_only_review",
        )
        self.assertEqual(
            result.loc["cell_2", "combined_evidence_status"],
            "forecast_only_review",
        )
        self.assertEqual(
            result.loc["cell_3", "combined_evidence_status"],
            "transient_qc_only",
        )
        self.assertEqual(
            result.loc["cell_4", "combined_evidence_status"],
            "functional_primary_residual_confounded",
        )
        self.assertEqual(
            result.loc["cell_5", "combined_evidence_status"],
            "functional_primary_residual_unavailable",
        )
        self.assertEqual(
            result.loc["cell_6", "combined_evidence_status"],
            "no_joint_evidence",
        )

    def test_continuous_rank_agreement_is_batch_preserving_and_auditable(self):
        ids = [f"cell_{index:02d}" for index in range(24)]
        batches = ["b1"] * 12 + ["b2"] * 12
        within_batch_order = np.tile(np.arange(12, dtype=float), 2)
        functional = pd.DataFrame(
            {
                "battery_id": ids,
                "cell_id": ids,
                "batch_id": batches,
                "is_persistent_review_candidate": [False] * 24,
                "shape_score": within_batch_order,
                "absolute_pattern_score": within_batch_order**2,
                "lifetime_score": -within_batch_order,
            }
        )
        residual = pd.DataFrame(
            {
                "battery_id": ids,
                "cell_id": ids,
                "batch_id": batches,
                "residual_status": ["no_forecast_mismatch"] * 24,
                "has_complete_oof_coverage": [True] * 24,
                "has_complete_oof_residual_evidence": [True] * 24,
                "is_residual_confounding": [False] * 24,
                "is_stable_oof_residual_candidate": [False] * 24,
                "is_stable_oof_residual_strong_candidate": [False] * 24,
                "median_oof_rarity_percentile": within_batch_order,
                "candidate_repeat_frequency": [0.0] * 24,
            }
        )
        result = compare_functional_and_oof_evidence(
            functional,
            residual,
            expected_battery_count=24,
            permutation_repeats=200,
            random_state=17,
        )
        repeated = compare_functional_and_oof_evidence(
            functional,
            residual,
            expected_battery_count=24,
            permutation_repeats=200,
            random_state=17,
        )
        audit = result.rank_agreement_audit.set_index("functional_score")
        self.assertEqual(set(audit.index), {
            "shape_score",
            "absolute_pattern_score",
            "lifetime_score",
        })
        self.assertTrue(audit["rank_agreement_available"].all())
        self.assertAlmostEqual(
            float(audit.loc["shape_score", "within_batch_rank_correlation"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(audit.loc["lifetime_score", "within_batch_rank_correlation"]),
            -1.0,
        )
        self.assertLessEqual(
            float(audit.loc["shape_score", "permutation_p_greater_equal"]),
            0.02,
        )
        self.assertTrue(
            audit["interpretation"]
            .str.contains("not_anomaly_ground_truth")
            .all()
        )
        pd.testing.assert_frame_equal(
            result.rank_agreement_audit,
            repeated.rank_agreement_audit,
        )


if __name__ == "__main__":
    unittest.main()
