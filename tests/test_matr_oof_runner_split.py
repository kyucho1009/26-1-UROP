from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_matr_oof_cross_validation import (
    ROLE_VALIDATION,
    assign_stratified_folds,
    split_inner_train_validation,
)


class OOFRunnerSplitTest(unittest.TestCase):
    def test_inner_validation_spans_observed_length_ranks_in_every_batch(self):
        rows: list[dict[str, object]] = []
        for batch_id, count in {"b1": 41, "b2": 43, "b3": 40, "b4": 45}.items():
            for index in range(count):
                rows.append(
                    {
                        "battery_id": f"{batch_id}_cell_{index:02d}",
                        "cell_id": f"{batch_id}_cell_{index:02d}",
                        "batch_id": batch_id,
                        "observed_cycle_end": 120 + 11 * index,
                    }
                )
        cells = pd.DataFrame(rows)
        assignment = assign_stratified_folds(cells, n_splits=5, seed=42)
        roles = split_inner_train_validation(
            assignment,
            outer_fold=0,
            validation_fraction=0.20,
            seed=123,
        )
        repeated = split_inner_train_validation(
            assignment.sample(frac=1.0, random_state=99),
            outer_fold=0,
            validation_fraction=0.20,
            seed=123,
        )
        self.assertEqual(roles[ROLE_VALIDATION], repeated[ROLE_VALIDATION])

        selected = set(roles[ROLE_VALIDATION])
        pool = assignment[assignment["outer_fold"] != 0]
        for _, group in pool.groupby("batch_id", sort=True):
            ordered = group.sort_values(
                ["observed_cycle_end", "battery_id"]
            ).reset_index(drop=True)
            selected_positions = {
                index
                for index, battery_id in enumerate(ordered["battery_id"])
                if str(battery_id) in selected
            }
            rank_bins = np.array_split(
                np.arange(len(ordered)), len(selected_positions)
            )
            self.assertTrue(
                all(len(set(rank_bin) & selected_positions) == 1 for rank_bin in rank_bins)
            )


if __name__ == "__main__":
    unittest.main()
