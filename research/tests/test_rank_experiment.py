import numpy as np
import pandas as pd

from scripts.run_rank_experiment import centered_rank, daily_scores, partitions


def test_calendar_split_purges_outcome_overlap():
    dates = pd.date_range("2020-01-01", periods=100)
    table = pd.DataFrame({"date": dates, "label_end": dates + pd.Timedelta(days=7)})
    train, validation, start, reserve = partitions(table)
    assert table.loc[train, "label_end"].max() < pd.Timestamp(start)
    assert table.loc[validation, "date"].min() == pd.Timestamp(start)
    assert table.loc[validation, "label_end"].max() < pd.Timestamp(reserve)
    assert not (train & validation).any()


def test_rank_ties_and_market_isolation():
    assert centered_rank(pd.Series([1, 1, 1])).eq(0).all()
    table = pd.DataFrame(
        {
            "market": ["US"] * 3 + ["UK"] * 3,
            "date": ["2020-01-01"] * 6,
            "target": [1, 2, 3, 3, 2, 1],
            "ridge": [1, 2, 3, 1, 2, 3],
            "momentum": [1, 2, 3, 1, 2, 3],
            "equal": [0] * 6,
        }
    )
    score = daily_scores(table).set_index("market")
    assert np.isclose(score.loc["US", "ridge_ic"], 1)
    assert np.isclose(score.loc["UK", "ridge_ic"], -1)
    assert score.equal_ic.eq(0).all()
