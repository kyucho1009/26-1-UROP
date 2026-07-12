"""Behavioral tests for the second-generation functional MATR detector.

These tests intentionally exercise scientific behavior rather than only checking
that an output table exists.  In particular, they keep persistent curve shape,
absolute-cycle behavior, observed lifetime, and one-cycle transients separate.
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np
import pandas as pd

from scripts.matr_functional_pattern_analysis import (
    FunctionalPatternConfig,
    FunctionalPatternResult,
    analyze_functional_patterns,
    build_functional_representations,
)


SUMMARY_COLUMNS = {
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
    "lifetime_rare_group_id",
    "is_lifetime_rare_group_candidate",
    "nearest_shape_peers",
    "nearest_lifetime_peers",
    "review_status",
}


def make_config(seed: int = 2718) -> FunctionalPatternConfig:
    """Use a deterministic, test-sized configuration across API revisions."""

    candidates = {
        "normalized_grid_points": 48,
        "random_state": seed,
        "verbose": False,
        "bootstrap_repeats": 10,
        "stability_repeats": 10,
        "crossfit_repeats": 8,
        "cluster_bootstrap_repeats": 10,
        "minimum_curve_points": 40,
        "min_curve_points": 40,
        "rare_group_min_size": 3,
        "cluster_min_size": 3,
        "lifetime_group_min_size": 2,
    }
    parameters = inspect.signature(FunctionalPatternConfig).parameters
    return FunctionalPatternConfig(
        **{key: value for key, value in candidates.items() if key in parameters}
    )


def make_curve(
    battery_id: str,
    *,
    batch_id: str = "b1",
    observed_cycles: int = 320,
    reference_lifetime: int | None = None,
    total_drop: float = 0.16,
    curve_kind: str = "normal",
    vertical_offset: float = 0.0,
    ripple_phase: float = 0.0,
    ripple_scale: float = 0.00015,
    spike_cycle: int | None = None,
    missing_cycle_range: tuple[int, int] | None = None,
    observation_complete: bool = True,
) -> pd.DataFrame:
    """Construct a deterministic SOH trajectory with known ground truth.

    ``reference_lifetime`` controls the physical degradation clock.  It is
    deliberately distinct from ``observed_cycles`` so a right-truncated curve
    can be generated without warping its partial prefix to a complete life.
    """

    reference_lifetime = int(reference_lifetime or observed_cycles)
    cycles = np.arange(1, int(observed_cycles) + 1, dtype=int)
    phase = np.clip((cycles - 1) / max(reference_lifetime - 1, 1), 0.0, 1.0)

    if curve_kind == "normal":
        cumulative_fade = total_drop * (0.58 * phase + 0.42 * phase**2)
    elif curve_kind == "late_acceleration":
        late = np.maximum((phase - 0.62) / 0.38, 0.0)
        cumulative_fade = total_drop * (0.72 * phase + 0.28 * phase**2)
        cumulative_fade += 0.075 * late**2
    elif curve_kind == "rare_s_curve":
        # Same starting/ending scale as a normal curve but a clearly different
        # persistent shape: long early plateau followed by rapid late fading.
        cumulative_fade = total_drop * (0.12 * phase + 0.88 * phase**4)
    else:
        raise ValueError(f"Unsupported synthetic curve kind: {curve_kind}")

    soh = (
        1.0
        + float(vertical_offset)
        - cumulative_fade
        + float(ripple_scale) * np.sin(cycles * 0.11 + float(ripple_phase))
    )
    if spike_cycle is not None:
        matches = np.flatnonzero(cycles == int(spike_cycle))
        if len(matches):
            # Only one observation is depressed; the following point returns
            # exactly to the underlying persistent trajectory.
            soh[matches[0]] -= 0.055

    keep = np.ones(len(cycles), dtype=bool)
    if missing_cycle_range is not None:
        lower, upper = missing_cycle_range
        keep &= ~((cycles >= int(lower)) & (cycles <= int(upper)))

    return pd.DataFrame(
        {
            "battery_id": battery_id,
            "cell_id": battery_id,
            "batch_id": batch_id,
            "cycle": cycles[keep],
            "soh": soh[keep],
            # Extra metadata is safe for implementations that infer censoring,
            # while implementations using only the five core columns may ignore it.
            "observation_complete": bool(observation_complete),
            "is_censored": not bool(observation_complete),
        }
    )


def normal_population(
    count: int,
    *,
    batch_id: str = "b1",
    cycles: int = 320,
    vertical_offset: float = 0.0,
    prefix: str = "normal",
) -> list[pd.DataFrame]:
    curves: list[pd.DataFrame] = []
    for index in range(count):
        curves.append(
            make_curve(
                f"{prefix}_{index:03d}",
                batch_id=batch_id,
                observed_cycles=cycles,
                total_drop=0.158 + 0.001 * (index % 5),
                vertical_offset=vertical_offset,
                ripple_phase=0.37 * index,
            )
        )
    return curves


def make_landmark_profile_curve(
    battery_id: str,
    *,
    landmark_cycles: tuple[int, int, int, int],
    batch_id: str = "b2",
    followup_cycles: int = 30,
) -> pd.DataFrame:
    """Build a monotone curve with a controlled T95/T90/T85/T80 profile."""

    t95, t90, t85, t80 = map(int, landmark_cycles)
    if not 1 < t95 < t90 < t85 < t80:
        raise ValueError("landmark cycles must be strictly increasing")
    cycle_end = t80 + int(followup_cycles)
    cycles = np.arange(1, cycle_end + 1, dtype=int)
    # Values just below each threshold make the intended crossing unambiguous,
    # while the final follow-up segment satisfies the sustained-crossing rule.
    anchor_cycles = np.asarray([1, t95, t90, t85, t80, cycle_end], dtype=float)
    anchor_soh = np.asarray([1.005, 0.949, 0.899, 0.849, 0.799, 0.792])
    soh = np.interp(cycles.astype(float), anchor_cycles, anchor_soh)
    return pd.DataFrame(
        {
            "battery_id": battery_id,
            "cell_id": battery_id,
            "batch_id": batch_id,
            "cycle": cycles,
            "soh": soh,
            "observation_complete": True,
            "is_censored": False,
        }
    )


class FunctionalPatternAnalysisBehaviorTest(unittest.TestCase):
    def run_analysis(
        self, curves: list[pd.DataFrame], *, seed: int = 2718
    ) -> tuple[FunctionalPatternResult, pd.DataFrame]:
        raw = pd.concat(curves, ignore_index=True)
        result = analyze_functional_patterns(raw, config=make_config(seed))
        self.assertIsInstance(result, FunctionalPatternResult)
        summary = result.pattern_summary.copy()
        self.assertTrue(
            SUMMARY_COLUMNS.issubset(summary.columns),
            f"Missing V2 result columns: {sorted(SUMMARY_COLUMNS - set(summary.columns))}",
        )
        self.assertEqual(len(summary), raw["battery_id"].nunique())
        self.assertFalse(summary["battery_id"].duplicated().any())
        return result, summary.set_index("battery_id")

    def test_same_shape_different_lifetime_is_lifetime_not_shape(self):
        curves = []
        for index in range(35):
            lifetime = 300 + (index % 7 - 3) * 4
            curves.append(
                make_curve(
                    f"ordinary_life_{index:03d}",
                    observed_cycles=lifetime,
                    total_drop=0.16,
                    ripple_phase=0.23 * index,
                )
            )
        curves.append(
            make_curve(
                "short_complete_life",
                observed_cycles=145,
                total_drop=0.16,
                ripple_phase=0.4,
            )
        )

        _, summary = self.run_analysis(curves, seed=101)
        short = summary.loc["short_complete_life"]
        self.assertTrue(bool(short["shape_analysis_eligible"]))
        self.assertFalse(bool(short["is_shape_candidate"]))
        self.assertTrue(bool(short["is_lifetime_candidate"]))

    def test_late_acceleration_is_persistent_shape_not_transient(self):
        curves = normal_population(42)
        curves.append(
            make_curve(
                "late_acceleration",
                curve_kind="late_acceleration",
                observed_cycles=320,
                total_drop=0.16,
            )
        )

        _, summary = self.run_analysis(curves, seed=202)
        candidate = summary.loc["late_acceleration"]
        self.assertTrue(bool(candidate["shape_analysis_eligible"]))
        self.assertTrue(bool(candidate["is_shape_candidate"]))
        # A persistent late acceleration can also advance T90/T85/T80, so the
        # lifetime channel may legitimately fire as a second, interpretable
        # label.  The essential separation is that it is not transient-only.
        self.assertFalse(bool(candidate["is_transient_candidate"]))

    def test_one_cycle_drop_and_recovery_is_transient_only(self):
        curves = normal_population(42)
        curves.append(
            make_curve(
                "single_cycle_drop",
                spike_cycle=157,
                observed_cycles=320,
                total_drop=0.16,
            )
        )

        _, summary = self.run_analysis(curves, seed=303)
        spike = summary.loc["single_cycle_drop"]
        self.assertTrue(bool(spike["is_transient_candidate"]))
        self.assertFalse(bool(spike["is_shape_candidate"]))
        self.assertFalse(bool(spike["is_lifetime_candidate"]))

    def test_early_truncation_is_ineligible_or_not_a_shape_candidate(self):
        curves = normal_population(36)
        curves.append(
            make_curve(
                "right_truncated_prefix",
                observed_cycles=125,
                reference_lifetime=320,
                total_drop=0.16,
                observation_complete=False,
            )
        )

        _, summary = self.run_analysis(curves, seed=404)
        truncated = summary.loc["right_truncated_prefix"]
        self.assertTrue(
            (not bool(truncated["shape_analysis_eligible"]))
            or (not bool(truncated["is_shape_candidate"])),
            "A right-truncated normal prefix must not become a persistent-shape anomaly",
        )

    def test_three_similar_rare_curves_are_reported_as_a_stable_rare_group(self):
        curves = normal_population(48)
        for index in range(3):
            curves.append(
                make_curve(
                    f"rare_group_{index}",
                    curve_kind="rare_s_curve",
                    observed_cycles=320,
                    total_drop=0.16,
                    ripple_phase=0.15 * index,
                )
            )

        _, summary = self.run_analysis(curves, seed=505)
        rare = summary.loc[[f"rare_group_{index}" for index in range(3)]]
        self.assertTrue(rare["is_rare_group_candidate"].astype(bool).all())
        self.assertTrue(rare["rare_group_id"].notna().all())
        self.assertEqual(rare["rare_group_id"].astype(str).nunique(), 1)
        self.assertGreater(float(rare["rare_group_stability"].min()), 0.0)
        for battery_id, peers in rare["nearest_shape_peers"].astype(str).items():
            other_rare_ids = {
                f"rare_group_{index}" for index in range(3)
            } - {battery_id}
            self.assertTrue(other_rare_ids.intersection(peers.split(",")))

    def test_accelerated_lifetime_pair_forms_its_own_rare_group(self):
        curves: list[pd.DataFrame] = []
        # Ordinary b2 cells have approximately 320-cycle landmark lifetime and
        # small natural variation, but the same normalized landmark profile.
        for index in range(38):
            lifetime = 312 + (index % 9) * 2
            curves.append(
                make_landmark_profile_curve(
                    f"b2_ordinary_{index:02d}",
                    landmark_cycles=(
                        int(round(0.30 * lifetime)),
                        int(round(0.60 * lifetime)),
                        int(round(0.83 * lifetime)),
                        lifetime,
                    ),
                )
            )

        # These two cells share both accelerated scale and landmark-spacing
        # profile; their approximately 10% scale difference should not split them.
        curves.extend(
            [
                make_landmark_profile_curve(
                    "fast_pair_a", landmark_cycles=(45, 90, 125, 150)
                ),
                make_landmark_profile_curve(
                    "fast_pair_b", landmark_cycles=(50, 99, 137, 165)
                ),
                # This cell is even shorter-lived, but the very early T95 followed
                # by a long plateau gives it a different lifetime profile.  It may
                # be an individual lifetime candidate, but not a member of the pair.
                make_landmark_profile_curve(
                    "extreme_different_profile", landmark_cycles=(8, 55, 68, 80)
                ),
            ]
        )

        _, summary = self.run_analysis(curves, seed=550)
        pair = summary.loc[["fast_pair_a", "fast_pair_b"]]
        self.assertTrue(
            pair["is_lifetime_rare_group_candidate"].astype(bool).all()
        )
        self.assertTrue(pair["lifetime_rare_group_id"].notna().all())
        self.assertEqual(
            pair["lifetime_rare_group_id"].astype(str).nunique(), 1
        )
        pair_group_id = str(pair.iloc[0]["lifetime_rare_group_id"])
        self.assertTrue(pair_group_id)
        self.assertIn(
            "fast_pair_b",
            str(summary.loc["fast_pair_a", "nearest_lifetime_peers"]).split(","),
        )
        self.assertIn(
            "fast_pair_a",
            str(summary.loc["fast_pair_b", "nearest_lifetime_peers"]).split(","),
        )

        excluded = summary.loc[
            ["b2_ordinary_00", "b2_ordinary_01", "extreme_different_profile"]
        ]
        self.assertFalse(
            excluded["lifetime_rare_group_id"].astype(str).eq(pair_group_id).any(),
            "Ordinary cells and a different extreme profile must not join the fast pair",
        )

    def test_batch_vertical_offset_does_not_create_within_batch_shape_anomaly(self):
        curves = normal_population(
            28, batch_id="b1", vertical_offset=0.0, prefix="batch1"
        )
        curves.extend(
            normal_population(
                28, batch_id="b2", vertical_offset=0.045, prefix="batch2"
            )
        )

        _, summary = self.run_analysis(curves, seed=606)
        self.assertFalse(summary["is_shape_candidate"].astype(bool).any())
        self.assertFalse(summary["is_rare_group_candidate"].astype(bool).any())

    def test_identical_curves_do_not_force_any_candidate(self):
        curves = []
        for batch_id in ["b1", "b2"]:
            for index in range(25):
                curves.append(
                    make_curve(
                        f"{batch_id}_identical_{index:02d}",
                        batch_id=batch_id,
                        observed_cycles=300,
                        total_drop=0.16,
                        ripple_scale=0.0,
                    )
                )

        _, summary = self.run_analysis(curves, seed=707)
        candidate_columns = [
            "is_shape_candidate",
            "is_absolute_pattern_candidate",
            "is_lifetime_candidate",
            "is_transient_candidate",
            "is_rare_group_candidate",
        ]
        self.assertFalse(summary[candidate_columns].astype(bool).any().any())

    def test_large_cycle_gap_is_explicitly_ineligible_or_remains_non_anomalous(self):
        curves = normal_population(36)
        curves.append(
            make_curve(
                "large_internal_gap",
                observed_cycles=320,
                total_drop=0.16,
                missing_cycle_range=(112, 188),
                ripple_phase=0.7,
            )
        )
        raw = pd.concat(curves, ignore_index=True)
        # The public representation builder itself must handle sparse cycle axes
        # without crashing or manufacturing non-finite detector inputs.
        representations = build_functional_representations(
            raw, config=make_config(808)
        )
        self.assertIsNotNone(representations)

        _, summary = self.run_analysis(curves, seed=808)
        gapped = summary.loc["large_internal_gap"]
        if bool(gapped["shape_analysis_eligible"]):
            self.assertTrue(np.isfinite(float(gapped["shape_score"])))
            self.assertFalse(bool(gapped["is_shape_candidate"]))
            self.assertFalse(bool(gapped["is_transient_candidate"]))
        else:
            status = str(gapped["review_status"]).lower()
            self.assertTrue(
                any(token in status for token in ["gap", "ineligible", "quality"]),
                f"Ineligibility should explain the cycle-gap issue, got {status!r}",
            )


if __name__ == "__main__":
    unittest.main()
