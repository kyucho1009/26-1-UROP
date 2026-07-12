import unittest

import numpy as np
import pandas as pd

from scripts.matr_anomaly_scoring import (
    AnomalyConfig,
    add_scores_by_seed,
    aggregate_cell_nonconformity,
    aggregate_physical_cell_evidence,
    audit_cell_score_coverage,
    apply_component_calibration,
    apply_empirical_conformal_pvalues,
    apply_horizon_severity_calibration,
    assert_seed_split_isolation,
    fit_component_calibration,
    fit_horizon_severity_calibration,
    prepare_residual_features,
    select_alpha_by_seed,
    split_validation_roles,
    summarize_common_horizon_scores,
)


def make_predictions(split: str, cells_by_seed: dict[int, list[str]]) -> pd.DataFrame:
    rows = []
    for seed, cells in cells_by_seed.items():
        for cell_index, battery_id in enumerate(cells):
            batch_id = "b1" if cell_index % 2 == 0 else "b2"
            for horizon in (10, 50, 100):
                for input_end_cycle in range(10, 25):
                    current_soh = 1.0 - 0.00015 * input_end_cycle - 0.0004 * cell_index
                    phase = 0.45 * input_end_cycle + 0.2 * cell_index
                    actual_delta = (
                        0.006
                        + 0.00004 * horizon
                        + 0.00025 * cell_index
                        + 0.00035 * np.sin(phase)
                    )
                    residual = (
                        0.00035
                        + 0.00006 * cell_index
                        + 0.00018 * np.sin(phase + 0.3)
                    )
                    pred_delta = actual_delta - residual
                    rows.append(
                        {
                            "dataset": "MATR",
                            "split": split,
                            "sample_mode": "sliding-window",
                            "seed": seed,
                            "model": "cpmlp_cpdsconv_fusion",
                            "battery_id": battery_id,
                            "cell_id": battery_id,
                            "batch_id": batch_id,
                            "input_end_cycle": input_end_cycle,
                            "target_cycle": input_end_cycle + horizon,
                            "horizon": horizon,
                            "current_soh": current_soh,
                            "actual_delta_soh": actual_delta,
                            "pred_delta_soh": pred_delta,
                            "actual_soh": current_soh - actual_delta,
                            "pred_soh": current_soh - pred_delta,
                        }
                    )
    return pd.DataFrame(rows)


class EmpiricalConformalScoringTest(unittest.TestCase):
    def test_configuration_and_validation_roles(self):
        config = AnomalyConfig()
        self.assertEqual(config.expected_horizons, (10, 50, 100))
        self.assertEqual(config.alpha_selection_horizons, (50, 100))
        self.assertEqual(config.alpha_cell_aggregation, "mean")
        self.assertAlmostEqual(config.validation_selection_fraction, 1.0 / 3.0)
        self.assertEqual(config.conformal_candidate_p, 0.10)
        self.assertEqual(config.conformal_strong_p, 0.05)
        self.assertEqual(config.threshold_method, "cell_empirical_conformal")

        validation = prepare_residual_features(
            make_predictions(
                "validation", {42: [f"val42_{index}" for index in range(18)]}
            ),
            "validation",
            target_model=config.target_model,
            expected_horizons=config.expected_horizons,
        )
        validation, assignments = split_validation_roles(
            validation,
            selection_fraction=config.validation_selection_fraction,
            random_state=config.random_state,
        )
        self.assertEqual(
            set(assignments["validation_role"]),
            {"alpha_selection", "conformal_calibration"},
        )
        self.assertEqual(
            assignments["battery_id"].nunique(),
            assignments.groupby("validation_role")["battery_id"].nunique().sum(),
        )

    def test_end_to_end_cell_conformal_pipeline(self):
        config = AnomalyConfig(
            alpha_grid=(0.0, 0.5, 1.0),
            min_paired_cells=3,
            bootstrap_repeats=100,
        )
        validation = prepare_residual_features(
            make_predictions(
                "validation",
                {
                    42: [f"val42_{index}" for index in range(18)],
                    43: [f"val43_{index}" for index in range(18)],
                },
            ),
            "validation",
            target_model=config.target_model,
            expected_horizons=config.expected_horizons,
        )
        test = prepare_residual_features(
            make_predictions(
                "test",
                {
                    42: [f"test42_{index}" for index in range(5)],
                    43: [f"test43_{index}" for index in range(5)],
                },
            ),
            "test",
            target_model=config.target_model,
            expected_horizons=config.expected_horizons,
        )
        assert_seed_split_isolation(validation, test)
        validation, _ = split_validation_roles(
            validation,
            selection_fraction=config.validation_selection_fraction,
            random_state=config.random_state,
        )

        alpha_raw = validation[
            validation["validation_role"] == "alpha_selection"
        ].copy()
        component_stats = fit_component_calibration(alpha_raw)
        validation = apply_component_calibration(validation, component_stats)
        test = apply_component_calibration(test, component_stats)

        alpha_selection = validation[
            validation["validation_role"] == "alpha_selection"
        ].copy()
        alpha_by_seed, search = select_alpha_by_seed(
            alpha_selection,
            alpha_grid=config.alpha_grid,
            horizons=config.alpha_selection_horizons,
            severity_aggregation=config.alpha_cell_aggregation,
            min_common_windows=config.min_common_windows,
            min_paired_cells=config.min_paired_cells,
            bootstrap_repeats=config.bootstrap_repeats,
            random_state=config.random_state,
        )
        self.assertTrue(search.groupby("seed")["selected"].sum().eq(1).all())
        validation = add_scores_by_seed(validation, alpha_by_seed)
        test = add_scores_by_seed(test, alpha_by_seed)

        alpha_horizon = summarize_common_horizon_scores(
            validation[validation["validation_role"] == "alpha_selection"],
            expected_horizons=config.expected_horizons,
            min_common_windows=config.min_common_windows,
        )
        calibration_horizon = summarize_common_horizon_scores(
            validation[
                validation["validation_role"] == "conformal_calibration"
            ],
            expected_horizons=config.expected_horizons,
            min_common_windows=config.min_common_windows,
        )
        test_horizon = summarize_common_horizon_scores(
            test,
            expected_horizons=config.expected_horizons,
            min_common_windows=config.min_common_windows,
        )
        horizon_stats = fit_horizon_severity_calibration(alpha_horizon)
        calibration_horizon = apply_horizon_severity_calibration(
            calibration_horizon, horizon_stats
        )
        test_horizon = apply_horizon_severity_calibration(test_horizon, horizon_stats)
        calibration_cells = aggregate_cell_nonconformity(
            calibration_horizon, expected_horizons=config.expected_horizons
        )
        test_cells = aggregate_cell_nonconformity(
            test_horizon, expected_horizons=config.expected_horizons
        )
        scored, audit = apply_empirical_conformal_pvalues(
            test_cells,
            calibration_cells,
            candidate_p=config.conformal_candidate_p,
            strong_p=config.conformal_strong_p,
        )

        self.assertEqual(len(scored), 10)
        self.assertTrue(scored["empirical_conformal_p_value"].between(0, 1).all())
        self.assertTrue(
            scored["leave_one_out_candidate_frequency"].between(0, 1).all()
        )
        self.assertEqual(set(audit["seed"]), {42, 43})
        coverage = pd.concat(
            [
                audit_cell_score_coverage(
                    calibration_cells, data_role="conformal_calibration"
                ),
                audit_cell_score_coverage(scored, data_role="locked_test"),
            ],
            ignore_index=True,
        )
        self.assertEqual(
            set(coverage["data_role"]),
            {"conformal_calibration", "locked_test"},
        )
        self.assertTrue(coverage["common_windows_min"].ge(5).all())
        self.assertFalse(
            {"is_cycle_anomaly_top5", "anomaly_ratio", "n_anomaly_horizons"}
            & set(scored.columns)
        )

    def test_median_horizon_aggregation_is_continuous_two_of_three(self):
        rows = []
        specifications = {
            "one_spike": {10: 8.0, 50: 0.5, 100: 0.6},
            "two_high": {10: 3.0, 50: 2.5, 100: 0.6},
        }
        for cell_id, severities in specifications.items():
            for horizon, severity in severities.items():
                rows.append(
                    {
                        "seed": 42,
                        "battery_id": cell_id,
                        "cell_id": cell_id,
                        "batch_id": "b1",
                        "horizon": horizon,
                        "horizon_mean_score": max(severity, 0.0),
                        "horizon_relative_severity": severity,
                        "n_common_windows": 10,
                        "common_input_cycle_start": 10,
                        "common_input_cycle_end": 19,
                        "horizon_max_score": max(severity, 0.0),
                    }
                )
        result = aggregate_cell_nonconformity(
            pd.DataFrame(rows), expected_horizons=(10, 50, 100)
        ).set_index("cell_id")
        self.assertAlmostEqual(
            result.loc["one_spike", "cell_nonconformity_score"], 0.6
        )
        self.assertAlmostEqual(
            result.loc["two_high", "cell_nonconformity_score"], 2.5
        )

    def test_empirical_pvalue_plus_one_ties_and_no_forced_candidates(self):
        calibration = pd.DataFrame(
            {
                "seed": [42] * 19,
                "battery_id": [f"cal_{index}" for index in range(19)],
                "cell_id": [f"cal_{index}" for index in range(19)],
                "batch_id": ["b1"] * 19,
                "validation_role": ["conformal_calibration"] * 19,
                "cell_nonconformity_score": np.arange(19, dtype=float),
            }
        )
        test = pd.DataFrame(
            {
                "seed": [42, 42, 42],
                "battery_id": ["above_all", "tied_max", "ordinary"],
                "cell_id": ["above_all", "tied_max", "ordinary"],
                "batch_id": ["b1", "b1", "b1"],
                "cell_nonconformity_score": [20.0, 18.0, 5.0],
            }
        )
        result, audit = apply_empirical_conformal_pvalues(
            test,
            calibration,
            candidate_p=0.10,
            strong_p=0.05,
        )
        result = result.set_index("cell_id")
        self.assertAlmostEqual(
            result.loc["above_all", "empirical_conformal_p_value"], 0.05
        )
        self.assertTrue(result.loc["above_all", "is_strong_candidate"])
        self.assertAlmostEqual(
            result.loc["tied_max", "empirical_conformal_p_value"], 0.10
        )
        self.assertTrue(result.loc["tied_max", "is_conformal_candidate"])
        self.assertFalse(result.loc["ordinary", "is_conformal_candidate"])
        self.assertAlmostEqual(audit.loc[0, "minimum_attainable_p_value"], 0.05)
        self.assertAlmostEqual(
            result.loc[
                "above_all", "leave_one_out_minimum_attainable_p_value"
            ],
            1.0 / 19.0,
        )
        self.assertFalse(
            result.loc[
                "above_all", "leave_one_out_strong_resolution_available"
            ]
        )
        self.assertFalse(
            audit.loc[0, "leave_one_out_strong_resolution_available"]
        )

    def test_component_moments_weight_batteries_equally(self):
        rows = []
        for target_cycle, residual, slope in [(1, 0.0, 0.0)]:
            rows.append(
                {
                    "seed": 42,
                    "horizon": 10,
                    "battery_id": "short",
                    "target_cycle": target_cycle,
                    "degradation_residual": residual,
                    "degradation_slope_score": slope,
                    "validation_role": "alpha_selection",
                }
            )
        for target_cycle, slope in enumerate([2.0, 4.0, 6.0], start=1):
            rows.append(
                {
                    "seed": 42,
                    "horizon": 10,
                    "battery_id": "long",
                    "target_cycle": target_cycle,
                    "degradation_residual": 10.0,
                    "degradation_slope_score": slope,
                    "validation_role": "alpha_selection",
                }
            )
        stats = fit_component_calibration(pd.DataFrame(rows)).iloc[0]
        self.assertAlmostEqual(stats["residual_mean"], 5.0)
        self.assertAlmostEqual(stats["slope_mean"], 2.0)
        self.assertEqual(stats["component_weighting"], "equal_battery_window_moments")

    def test_duplicate_cells_and_metadata_conflicts_are_rejected(self):
        calibration = pd.DataFrame(
            {
                "seed": [42, 42],
                "battery_id": ["cal_a", "cal_b"],
                "cell_id": ["cal_a", "cal_b"],
                "validation_role": ["conformal_calibration"] * 2,
                "cell_nonconformity_score": [0.0, 1.0],
            }
        )
        duplicated_test = pd.DataFrame(
            {
                "seed": [42, 42],
                "battery_id": ["test_a", "test_a"],
                "cell_id": ["test_a", "test_a"],
                "cell_nonconformity_score": [2.0, 2.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "one row per seed/physical cell"):
            apply_empirical_conformal_pvalues(
                duplicated_test,
                calibration,
                candidate_p=0.10,
                strong_p=0.05,
            )

        rows = []
        for horizon, batch_id in zip((10, 50, 100), ("b1", "b1", "b2")):
            rows.append(
                {
                    "seed": 42,
                    "battery_id": "cell_a",
                    "cell_id": "cell_a",
                    "batch_id": batch_id,
                    "horizon": horizon,
                    "horizon_mean_score": 1.0,
                    "horizon_relative_severity": 1.0,
                    "n_common_windows": 10,
                    "common_input_cycle_start": 10,
                    "common_input_cycle_end": 19,
                    "horizon_max_score": 1.0,
                }
            )
        with self.assertRaisesRegex(ValueError, "one batch_id"):
            aggregate_cell_nonconformity(
                pd.DataFrame(rows), expected_horizons=(10, 50, 100)
            )

    def test_missing_horizon_is_rejected(self):
        predictions = make_predictions("test", {42: ["cell_a"]})
        scored = prepare_residual_features(
            predictions,
            "test",
            target_model="cpmlp_cpdsconv_fusion",
            expected_horizons=(10, 50, 100),
        )
        scored["degradation_anomaly_score"] = 1.0
        missing = scored[scored["horizon"] != 100].copy()
        with self.assertRaisesRegex(ValueError, "Expected scored horizons"):
            summarize_common_horizon_scores(
                missing,
                expected_horizons=(10, 50, 100),
                min_common_windows=5,
            )

    def test_physical_cell_requires_all_available_seed_confirmations(self):
        seed_rows = pd.DataFrame(
            {
                "seed": [42, 43, 42],
                "battery_id": ["repeated", "repeated", "single"],
                "cell_id": ["repeated", "repeated", "single"],
                "batch_id": ["b1", "b1", "b2"],
                "cell_nonconformity_score": [3.0, 2.5, 4.0],
                "empirical_conformal_p_value": [0.05, 0.20, 0.05],
                "leave_one_out_candidate_frequency": [0.9, 0.2, 0.8],
                "is_conformal_candidate": [True, False, True],
                "is_strong_candidate": [True, False, True],
                "h10_relative_severity": [3.0, 2.0, 4.0],
                "h50_relative_severity": [2.5, 2.5, 3.5],
                "h100_relative_severity": [3.5, 3.0, 4.5],
            }
        )
        physical = aggregate_physical_cell_evidence(seed_rows).set_index("cell_id")
        self.assertFalse(physical.loc["repeated", "is_physical_candidate"])
        self.assertTrue(physical.loc["single", "is_physical_candidate"])
        self.assertTrue(physical.loc["single", "is_single_seed_candidate"])
        self.assertFalse(physical.loc["single", "is_repeated_seed_candidate"])
        self.assertEqual(physical.loc["repeated", "n_test_seeds"], 2)


if __name__ == "__main__":
    unittest.main()
