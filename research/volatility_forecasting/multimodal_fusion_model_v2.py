"""Multimodal fusion network predicting non-linear multi-horizon return and volatility distributions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn


@dataclass(frozen=True)
class MultimodalHorizonForecast:
    horizon: int  # 1, 3, 5, 7
    expected_cumulative_return: float
    reconstructed_median_price: float
    volatility_sigma: float
    prediction_interval_80pct: tuple[float, float]
    quantiles_90pct: tuple[float, float]
    direction_probability_up: float
    direction_probability_neutral: float
    direction_probability_down: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultimodalFusionModel(nn.Module):
    """Multi-branch network combining numeric trends, causal news events, and market regime context."""

    def __init__(
        self,
        numeric_dim: int = 32,
        news_dim: int = 18,
        regime_dim: int = 4,
        hidden_dim: int = 64,
        horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    ) -> None:
        super().__init__()
        self.horizons = horizons
        n_h = len(horizons)

        # 1. Numeric Trend Branch
        self.numeric_branch = nn.Sequential(
            nn.Linear(numeric_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # 2. News Event Branch
        self.news_branch = nn.Sequential(
            nn.Linear(news_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # 3. Fusion Layer
        fusion_dim = (hidden_dim // 2) * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Location residual head (Delta mu_h)
        self.location_head = nn.Linear(hidden_dim, n_h)

        # Log-volatility residual head (Delta log_sigma_h)
        self.volatility_head = nn.Linear(hidden_dim, n_h)

    def forward(
        self,
        numeric_features: torch.Tensor,
        news_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_emb = self.numeric_branch(numeric_features)

        if news_features is not None:
            news_emb = self.news_branch(news_features)
        else:
            news_emb = torch.zeros_like(num_emb)

        fused = self.fusion(torch.cat([num_emb, news_emb], dim=-1))

        delta_mu = self.location_head(fused)
        delta_log_vol = self.volatility_head(fused)

        return delta_mu, delta_log_vol

    def predict_distribution(
        self,
        base_price: float,
        numeric_arr: np.ndarray,
        news_arr: np.ndarray | None,
        har_vol_daily: float = 0.0168,
        degrees_of_freedom: float = 6.0,  # Student-t heavy tail parameter
    ) -> list[MultimodalHorizonForecast]:
        """Produce exact multi-horizon return, price, interval, and direction distributions."""
        self.eval()
        with torch.no_grad():
            x_num = torch.tensor(numeric_arr[np.newaxis, :], dtype=torch.float32)
            x_news = (
                torch.tensor(news_arr[np.newaxis, :], dtype=torch.float32)
                if news_arr is not None
                else None
            )
            d_mu_t, d_log_vol_t = self.forward(x_num, x_news)
            d_mu = d_mu_t.squeeze(0).numpy()
            d_log_vol = d_log_vol_t.squeeze(0).numpy()

        results: list[MultimodalHorizonForecast] = []
        t_dist = stats.t(df=degrees_of_freedom)
        q80_factor = float(t_dist.ppf(0.90))  # 80% central interval: [10%, 90%]
        q90_factor = float(t_dist.ppf(0.95))  # 90% central interval: [5%, 95%]

        for idx, h in enumerate(self.horizons):
            # 1. Location: base drift + learned non-linear residual
            mu_h = float(d_mu[idx])

            # 2. Volatility: HAR baseline * exp(learned residual)
            sigma_har_h = har_vol_daily * math.sqrt(h)
            sigma_h = float(sigma_har_h * math.exp(d_log_vol[idx]))

            # 3. Price reconstruction
            p_median = float(base_price * math.exp(mu_h))

            # 4. Student-t prediction intervals
            low_80 = float(base_price * math.exp(mu_h - q80_factor * sigma_h))
            high_80 = float(base_price * math.exp(mu_h + q80_factor * sigma_h))

            low_90 = float(base_price * math.exp(mu_h - q90_factor * sigma_h))
            high_90 = float(base_price * math.exp(mu_h + q90_factor * sigma_h))

            # 5. Direction probabilities with neutral band tau_h = 0.5 * sigma_h
            tau_h = 0.5 * sigma_h
            p_down = float(t_dist.cdf((-tau_h - mu_h) / max(sigma_h, 1e-6)))
            p_up = float(1.0 - t_dist.cdf((tau_h - mu_h) / max(sigma_h, 1e-6)))
            p_neutral = max(0.0, 1.0 - p_down - p_up)

            results.append(
                MultimodalHorizonForecast(
                    horizon=h,
                    expected_cumulative_return=round(mu_h, 6),
                    reconstructed_median_price=round(p_median, 2),
                    volatility_sigma=round(sigma_h, 6),
                    prediction_interval_80pct=(round(low_80, 2), round(high_80, 2)),
                    quantiles_90pct=(round(low_90, 2), round(high_90, 2)),
                    direction_probability_up=round(p_up, 4),
                    direction_probability_neutral=round(p_neutral, 4),
                    direction_probability_down=round(p_down, 4),
                )
            )

        return results
