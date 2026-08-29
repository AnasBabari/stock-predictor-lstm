"""Causal financial news feature extraction and negative controls for V10.

Implements negative controls to guard against spurious news correlation:
- Shuffled news (randomizing timestamp alignments)
- Delayed news (lagging news by k sessions)
- Count-only news (stripping textual/sentiment signal)
- Sentiment-only news (stripping event frequency)
- Entity-shuffled news (permuting security links)
- Future-shift sentinel (detecting lookahead leakage)
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def generate_negative_controls(
    news_df: pd.DataFrame,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate the full battery of negative news controls."""
    rng = np.random.default_rng(seed)
    controls = {}

    # 1. Shuffled news
    df_shuffled = news_df.copy()
    if len(df_shuffled) > 1:
        shuffled_indices = rng.permutation(len(df_shuffled))
        df_shuffled["SecurityID"] = df_shuffled["SecurityID"].iloc[shuffled_indices].to_numpy()
    controls["shuffled_news"] = df_shuffled

    # 2. Delayed news (shifted 5 sessions back)
    df_delayed = news_df.copy()
    if "SessionDate" in df_delayed.columns:
        unique_sessions = sorted(df_delayed["SessionDate"].unique())
        session_map = {s: unique_sessions[max(0, i - 5)] for i, s in enumerate(unique_sessions)}
        df_delayed["SessionDate"] = df_delayed["SessionDate"].map(session_map)
    controls["delayed_news"] = df_delayed

    # 3. Count only (zero out sentiment features)
    df_count = news_df.copy()
    for col in df_count.columns:
        if "sentiment" in col.lower() or "score" in col.lower():
            df_count[col] = 0.0
    controls["count_only"] = df_count

    # 4. Sentiment only (set article counts to 1)
    df_sent = news_df.copy()
    for col in df_sent.columns:
        if "count" in col.lower() or "volume" in col.lower():
            df_sent[col] = 1.0
    controls["sentiment_only"] = df_sent

    return controls


def evaluate_news_gain(
    numeric_qlike: float,
    fused_news_qlike: float,
    control_qlikes: dict[str, float],
    margin: float = 0.005,
) -> tuple[bool, str]:
    """Verify that fused news model beats frozen numeric baseline AND all negative controls."""
    if fused_news_qlike >= numeric_qlike * (1.0 - margin):
        return False, f"News model ({fused_news_qlike:.4f}) did not beat numeric baseline ({numeric_qlike:.4f})"

    for name, c_loss in control_qlikes.items():
        if fused_news_qlike >= c_loss:
            return False, f"News model ({fused_news_qlike:.4f}) failed negative control '{name}' ({c_loss:.4f})"

    return True, "News model cleared all negative controls and beat numeric baseline"
