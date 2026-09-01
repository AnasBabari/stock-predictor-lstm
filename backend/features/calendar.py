"""Cyclical calendar feature encoding (Day of Week, Month)."""

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode day of week (5-7 trading days) and month (12 months)
    as cyclical sin/cos signals for neural network consumption.
    """
    result = df.copy()

    # Month: 1 to 12
    months = result.index.month
    result["Month_Sin"] = np.sin(2 * np.pi * months / 12.0)
    result["Month_Cos"] = np.cos(2 * np.pi * months / 12.0)

    # Day of week: 0 to 6
    day_of_week = result.index.dayofweek
    result["Day_Sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    result["Day_Cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    return result
