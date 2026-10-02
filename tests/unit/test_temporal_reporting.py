import pandas as pd
import pytest

from sync2act.reporting.temporal import prepare_mse_tables


def test_relative_plot_uses_ensemble_clean_baseline_and_mean_of_ratios():
    frame = pd.DataFrame(
        [
            ["a", "act_lite", "clean", 7, 2.0, 4.0],
            ["a", "act_lite", "mixed_action_damaged", 7, 6.0, 8.0],
            ["b", "act_lite", "clean", 7, 10.0, 20.0],
            ["b", "act_lite", "mixed_action_damaged", 7, 20.0, 30.0],
        ],
        columns=[
            "dataset",
            "model",
            "condition",
            "seed",
            "first_step_action_mse",
            "ensemble_action_mse",
        ],
    )
    tables = prepare_mse_tables(frame)
    impact = tables["relative_mse_by_model_condition"].iloc[0]
    assert impact.first_step_relative_mse == pytest.approx((3 + 2) / 2)
    assert impact.ensemble_relative_mse == pytest.approx((2 + 1.5) / 2)
    assert impact.ensemble_relative_mse != pytest.approx((8 + 30) / (4 + 20))
    changes = tables["mse_change_by_model_condition"].set_index("condition")
    assert changes.loc["mixed_action_damaged", "mean_paired_mse_change_percent"] == pytest.approx(
        ((8 / 6 - 1) + (30 / 20 - 1)) * 50
    )
    row = tables["by_dataset_model"].set_index("dataset").loc["a"]
    assert row.first_step_mse == 4
    assert row.ensemble_mse == 6
    assert row.mse_difference == 2
    assert row.mse_change_percent == 50
    # The presentation data is exclusively MSE, not MAE or difference metrics.
    assert all(
        "smooth" not in c and "jerk" not in c and "mae" not in c
        for t in tables.values()
        for c in t.columns
    )


def test_missing_clean_baseline_is_rejected():
    frame = pd.DataFrame(
        [
            {
                "dataset": "a",
                "model": "act_lite",
                "condition": "mixed_state_damaged",
                "seed": 7,
                "first_step_action_mse": 2.0,
                "ensemble_action_mse": 3.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="missing matched clean"):
        prepare_mse_tables(frame)
