"""Rich multi-horizon numeric, trend, momentum, volatility, and sector/market relative features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnrichedMarketFeatures:
    # 1. Multi-horizon returns & trends
    return_1d: float
    return_2d: float
    return_3d: float
    return_5d: float
    return_10d: float
    return_20d: float
    return_60d: float
    ema5_to_ema20_ratio: float
    slope_ema5: float
    slope_ema20: float
    price_to_sma20_ratio: float
    drawdown_20d: float
    drawdown_60d: float

    # 2. Momentum & acceleration
    momentum_5d: float
    momentum_10d: float
    momentum_20d: float
    momentum_acceleration_5_20: float
    short_minus_long_momentum: float

    # 3. Volatility components (Realized, Parkinson, Garman-Klass, HAR)
    vol_realized_5d: float
    vol_realized_20d: float
    vol_realized_60d: float
    vol_parkinson_20d: float
    vol_garman_klass_20d: float
    har_daily_vol: float
    har_weekly_vol: float
    har_monthly_vol: float

    # 4. Volume & liquidity dynamics
    volume_z_score_20d: float
    volume_5d_to_20d_ratio: float
    price_volume_interaction_5d: float
    abnormal_volume_flag: float

    # 5. Relative Sector & Market dynamics (XLV, QQQ, SPY)
    rel_return_5d_vs_sector: float
    rel_return_20d_vs_sector: float
    rel_return_5d_vs_market: float
    rel_return_20d_vs_market: float

    def to_array(self) -> np.ndarray:
        """Convert all feature values to 1D float array."""
        return np.array(list(self.__dict__.values()), dtype=float)

    @classmethod
    def feature_names(cls) -> list[str]:
        return list(cls.__annotations__.keys())


class EnrichedFeatureExtractor:
    """Extracts rich trend, momentum, volatility, and sector/market relative features from market panels."""

    @staticmethod
    def extract_from_series(
        target_df: pd.DataFrame,
        sector_df: pd.DataFrame | None = None,
        market_df: pd.DataFrame | None = None,
    ) -> EnrichedMarketFeatures:
        df = target_df.copy().sort_index()
        c = df["Close"].to_numpy(dtype=float)
        h = df["High"].to_numpy(dtype=float)
        low_p = df["Low"].to_numpy(dtype=float)
        o = df["Open"].to_numpy(dtype=float)
        v = df["Volume"].to_numpy(dtype=float)

        n = len(c)
        if n < 65:
            raise ValueError(f"Insufficient history for enriched feature extraction: {n} < 65")

        # 1. Multi-horizon log returns
        r1 = float(np.log(c[-1] / c[-2]))
        r2 = float(np.log(c[-1] / c[-3]))
        r3 = float(np.log(c[-1] / c[-4]))
        r5 = float(np.log(c[-1] / c[-6]))
        r10 = float(np.log(c[-1] / c[-11]))
        r20 = float(np.log(c[-1] / c[-21]))
        r60 = float(np.log(c[-1] / c[-61]))

        # Trends
        close_series = pd.Series(c)
        ema5 = close_series.ewm(span=5, adjust=False).mean().to_numpy()
        ema20 = close_series.ewm(span=20, adjust=False).mean().to_numpy()
        sma20 = close_series.rolling(20).mean().to_numpy()

        ema5_to_ema20 = float((ema5[-1] / max(ema20[-1], 1e-8)) - 1.0)
        slope_ema5 = float(np.polyfit(np.arange(5), ema5[-5:], 1)[0] / c[-1])
        slope_ema20 = float(np.polyfit(np.arange(20), ema20[-20:], 1)[0] / c[-1])
        price_to_sma20 = float((c[-1] / max(sma20[-1], 1e-8)) - 1.0)

        # Drawdown
        peak20 = float(np.max(c[-20:]))
        peak60 = float(np.max(c[-60:]))
        dd20 = float((c[-1] - peak20) / max(peak20, 1e-8))
        dd60 = float((c[-1] - peak60) / max(peak60, 1e-8))

        # Momentum & acceleration
        mom5 = r5
        mom10 = r10
        mom20 = r20
        # Acceleration: rate of change of 5d return vs 20d return
        mom_accel = float(mom5 - (mom20 / 4.0))
        short_minus_long = float(mom5 - mom20)

        # Volatility measures
        daily_log_rets = np.log(c[1:] / c[:-1])
        vol5 = float(np.std(daily_log_rets[-5:]))
        vol20 = float(np.std(daily_log_rets[-20:]))
        vol60 = float(np.std(daily_log_rets[-60:]))

        # Parkinson 20d
        log_hl = np.log(h[-20:] / low_p[-20:]) ** 2 / (4.0 * math.log(2.0))
        vol_parkinson20 = float(math.sqrt(max(1e-8, float(np.mean(log_hl)))))

        # Garman-Klass 20d
        gk = (
            0.5 * (np.log(h[-20:] / low_p[-20:])) ** 2
            - (2.0 * math.log(2.0) - 1.0) * (np.log(c[-20:] / o[-20:])) ** 2
        )
        vol_gk20 = float(math.sqrt(max(1e-8, float(np.mean(gk)))))

        # HAR volatility components
        har_d = vol5
        har_w = vol20
        har_m = vol60

        # Volume dynamics
        vol_series = pd.Series(v)
        v_mean20 = float(vol_series.rolling(20).mean().iloc[-1])
        v_std20 = float(vol_series.rolling(20).std().iloc[-1])
        v_zscore20 = float((v[-1] - v_mean20) / max(v_std20, 1.0))
        v_5_to_20 = float(np.mean(v[-5:]) / max(v_mean20, 1.0))
        pv_interact = float(r5 * (v_5_to_20 - 1.0))
        abnormal_vol = 1.0 if abs(v_zscore20) > 2.0 else 0.0

        # Relative Sector & Market returns
        rel_sec_5 = 0.0
        rel_sec_20 = 0.0
        if sector_df is not None and len(sector_df) >= 21:
            sec_c = sector_df["Close"].to_numpy(dtype=float)
            sec_r5 = float(np.log(sec_c[-1] / sec_c[-6]))
            sec_r20 = float(np.log(sec_c[-1] / sec_c[-21]))
            rel_sec_5 = r5 - sec_r5
            rel_sec_20 = r20 - sec_r20

        rel_mkt_5 = 0.0
        rel_mkt_20 = 0.0
        if market_df is not None and len(market_df) >= 21:
            mkt_c = market_df["Close"].to_numpy(dtype=float)
            mkt_r5 = float(np.log(mkt_c[-1] / mkt_c[-6]))
            mkt_r20 = float(np.log(mkt_c[-1] / mkt_c[-21]))
            rel_mkt_5 = r5 - mkt_r5
            rel_mkt_20 = r20 - mkt_r20

        return EnrichedMarketFeatures(
            return_1d=r1,
            return_2d=r2,
            return_3d=r3,
            return_5d=r5,
            return_10d=r10,
            return_20d=r20,
            return_60d=r60,
            ema5_to_ema20_ratio=ema5_to_ema20,
            slope_ema5=slope_ema5,
            slope_ema20=slope_ema20,
            price_to_sma20_ratio=price_to_sma20,
            drawdown_20d=dd20,
            drawdown_60d=dd60,
            momentum_5d=mom5,
            momentum_10d=mom10,
            momentum_20d=mom20,
            momentum_acceleration_5_20=mom_accel,
            short_minus_long_momentum=short_minus_long,
            vol_realized_5d=vol5,
            vol_realized_20d=vol20,
            vol_realized_60d=vol60,
            vol_parkinson_20d=vol_parkinson20,
            vol_garman_klass_20d=vol_gk20,
            har_daily_vol=har_d,
            har_weekly_vol=har_w,
            har_monthly_vol=har_m,
            volume_z_score_20d=v_zscore20,
            volume_5d_to_20d_ratio=v_5_to_20,
            price_volume_interaction_5d=pv_interact,
            abnormal_volume_flag=abnormal_vol,
            rel_return_5d_vs_sector=rel_sec_5,
            rel_return_20d_vs_sector=rel_sec_20,
            rel_return_5d_vs_market=rel_mkt_5,
            rel_return_20d_vs_market=rel_mkt_20,
        )
