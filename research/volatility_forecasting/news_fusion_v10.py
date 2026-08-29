"""Causal financial news feature extraction and negative controls for V10.

Implements negative controls to guard against spurious news correlation:
- Shuffled news (randomizing timestamp alignments)
- Delayed news (lagging news availability forward by 5 sessions)
- Count-only news (stripping textual/sentiment signal)
- Sentiment-only news (stripping event frequency)
- Entity-shuffled news (permuting security links)
- Future-shift sentinel (deliberate lookahead for leakage testing)
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
    sec_col = "SecurityID" if "SecurityID" in news_df.columns else "Ticker"

    # 1. Shuffled news
    df_shuffled = news_df.copy()
    if len(df_shuffled) > 1:
        shuffled_indices = rng.permutation(len(df_shuffled))
        df_shuffled[sec_col] = df_shuffled[sec_col].iloc[shuffled_indices].to_numpy()
    controls["shuffled_news"] = df_shuffled

    # 2. Delayed news (shifted 5 sessions into future availability)
    df_delayed = news_df.copy()
    if "SessionDate" in df_delayed.columns:
        unique_sessions = sorted(df_delayed["SessionDate"].unique())
        session_map = {
            s: unique_sessions[min(len(unique_sessions) - 1, i + 5)]
            for i, s in enumerate(unique_sessions)
        }
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

    # 5. Entity-shuffled news
    df_entity = news_df.copy()
    if sec_col in df_entity.columns:
        unique_secs = df_entity[sec_col].unique()
        if len(unique_secs) > 1:
            perm_map = dict(zip(unique_secs, rng.permutation(unique_secs), strict=True))
            df_entity[sec_col] = df_entity[sec_col].map(perm_map)
    controls["entity_shuffled"] = df_entity

    # 6. Future-shift sentinel (deliberate lookahead for leakage testing)
    df_future = news_df.copy()
    if "SessionDate" in df_future.columns:
        unique_sessions = sorted(df_future["SessionDate"].unique())
        session_map_future = {
            s: unique_sessions[max(0, i - 5)] for i, s in enumerate(unique_sessions)
        }
        df_future["SessionDate"] = df_future["SessionDate"].map(session_map_future)
    controls["future_shift_sentinel"] = df_future

    return controls


def extract_causal_news_features(
    news_df: pd.DataFrame,
    target_sessions: list[str],
    security_ids: list[str],
    windows: tuple[int, ...] = (1, 3, 5, 20),
) -> pd.DataFrame:
    """Extract causal rolling window news aggregates per security and session."""
    sec_col = "SecurityID" if "SecurityID" in news_df.columns else "Ticker"
    rows = []

    # Sort chronologically
    sorted_news = news_df.sort_values(by=["SessionDate"]).copy()

    for sec in security_ids:
        sec_news = sorted_news[sorted_news[sec_col] == sec]
        for s_idx, current_session in enumerate(target_sessions):
            # Prior causal sessions <= current_session
            causal_news = sec_news[sec_news["SessionDate"] <= current_session]
            feat_dict: dict[str, Any] = {
                "SessionDate": current_session,
                "SecurityID": sec,
            }

            for w in windows:
                w_start_idx = max(0, s_idx - w + 1)
                w_sessions = set(target_sessions[w_start_idx : s_idx + 1])
                w_news = causal_news[causal_news["SessionDate"].isin(w_sessions)]

                feat_dict[f"news_count_{w}d"] = float(len(w_news))
                if len(w_news) > 0 and "sentiment" in w_news.columns:
                    feat_dict[f"news_sentiment_mean_{w}d"] = float(w_news["sentiment"].mean())
                    feat_dict[f"news_sentiment_std_{w}d"] = float(w_news["sentiment"].std(ddof=0))
                else:
                    feat_dict[f"news_sentiment_mean_{w}d"] = 0.0
                    feat_dict[f"news_sentiment_std_{w}d"] = 0.0

            rows.append(feat_dict)

    return pd.DataFrame(rows)


def evaluate_news_gain(
    numeric_qlike: float,
    fused_news_qlike: float,
    control_qlikes: dict[str, float],
    margin: float = 0.005,
) -> tuple[bool, str]:
    """Verify that fused news model beats frozen numeric baseline AND all negative controls."""
    if fused_news_qlike >= numeric_qlike * (1.0 - margin):
        return (
            False,
            f"News model ({fused_news_qlike:.4f}) did not beat numeric baseline ({numeric_qlike:.4f})",
        )

    for name, c_loss in control_qlikes.items():
        if fused_news_qlike >= c_loss:
            return (
                False,
                f"News model ({fused_news_qlike:.4f}) failed negative control '{name}' ({c_loss:.4f})",
            )

    return True, "News model cleared all negative controls and beat numeric baseline"
