"""Canonical schemas for the provider-agnostic options-IV adapter."""

from __future__ import annotations

# Raw ORATS near-EOD strikes file grain: one row per ticker/expiry/strike
# with call and put legs side by side.
ORATS_STRIKES_COLUMNS: tuple[str, ...] = (
    "ticker",
    "cOpra",
    "pOpra",
    "stkPx",
    "expirDate",
    "yte",
    "strike",
    "cVolu",
    "cOi",
    "pVolu",
    "pOi",
    "cBidPx",
    "cValue",
    "cAskPx",
    "pBidPx",
    "pValue",
    "pAskPx",
    "cBidIv",
    "cMidIv",
    "cAskIv",
    "smoothSmvVol",
    "pBidIv",
    "pMidIv",
    "pAskIv",
    "iRate",
    "divRate",
    "residualRateData",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "phi",
    "driftlessTheta",
    "extVol",
    "extCTheo",
    "extPTheo",
    "spot_px",
    "trade_date",
)

# Normalized strike record: every provider maps into this grain.
NORMALIZED_COLUMNS: tuple[str, ...] = (
    "ticker",
    "trade_date",
    "expir_date",
    "dte_days",
    "strike",
    "underlying_px",
    "call_mid_iv",
    "put_mid_iv",
    "call_bid",
    "put_bid",
    "call_volume",
    "put_volume",
    "call_oi",
    "put_oi",
    "call_delta",
    "source",
    "source_timestamp",
)

# Canonical daily feature row (study contract).
FEATURE_ROW_FIELDS: tuple[str, ...] = (
    "date",
    "ticker",
    "atm_iv_30d",
    "atm_iv_60d",
    "iv_term_60_30",
    "iv_term_90_30",
    "put_iv_25d",
    "call_iv_25d",
    "skew_25d",
    "iv_rv_spread_20d",
    "option_volume",
    "open_interest",
    "source",
    "source_timestamp",
)

# Source labels. Only allow-listed historical vendor snapshots are study
# eligible; everything else must fail the eligibility gate.
SOURCE_ORATS_NEAR_EOD = "ORATS_NEAR_EOD"
SOURCE_ORATS_INTRADAY = "ORATS_INTRADAY_1M"
SOURCE_IVYDB = "IVYDB_US"
SOURCE_YFINANCE_PLUMBING = "YFINANCE_EPHEMERAL_PLUMBING_ONLY"
STUDY_ELIGIBLE_SOURCES: tuple[str, ...] = (
    SOURCE_ORATS_NEAR_EOD,
    SOURCE_ORATS_INTRADAY,
    SOURCE_IVYDB,
)
