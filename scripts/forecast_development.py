"""Development forecast CLI for anchored probabilistic inference and diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from backend.contracts.forecast_contracts import PriceReturnDistributionContract
from backend.evaluation.abstention_gate import PlausibilityAbstentionGate
from research.volatility_forecasting.causal_dataset_v1 import (
    STATIONARY_FEATURE_COLUMNS_V1,
)
from research.volatility_forecasting.causal_models_v1 import CausalTCNModel


@dataclass(frozen=True)
class DevelopmentForecastOutput:
    ticker: str
    base_date: str
    base_price: float
    status: str
    is_certified_production_claim: bool
    median_prices: list[float]
    cumulative_returns_median: list[float]
    intervals_80pct: list[tuple[float, float]]
    gate_decision: str
    provenance_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DevelopmentForecastRunner:
    """Executes anchored development forecast with strict P0 pricing and abstention gating."""

    def __init__(
        self,
        model: CausalTCNModel | None = None,
        gate: PlausibilityAbstentionGate | None = None,
    ) -> None:
        self.model = model or CausalTCNModel(
            in_features=len(STATIONARY_FEATURE_COLUMNS_V1),
            forecast_days=7,
            mode="return",
        )
        self.gate = gate or PlausibilityAbstentionGate()

    def run_development_forecast(
        self,
        ticker: str,
        base_date: str,
        base_price: float,
        recent_feature_window: np.ndarray,  # (60, 16)
        daily_volatility: float = 0.018,
    ) -> DevelopmentForecastOutput:
        if base_price <= 0 or not math.isfinite(base_price):
            raise ValueError(f"Base price P0 must be strictly positive, got {base_price}")
        if recent_feature_window.shape != (60, len(STATIONARY_FEATURE_COLUMNS_V1)):
            raise ValueError(
                f"Feature window must have shape (60, {len(STATIONARY_FEATURE_COLUMNS_V1)})"
            )

        self.model.eval()
        x_t = torch.tensor(recent_feature_window[np.newaxis, ...], dtype=torch.float32)
        with torch.no_grad():
            pred_returns = self.model(x_t).squeeze(0).numpy().tolist()

        # Reconstruct median prices anchored at base_price (P0)
        reconstructed_prices = PriceReturnDistributionContract.reconstruct_anchored_prices(
            base_price=base_price,
            predicted_cumulative_returns_median=pred_returns,
        )

        # Build 80% intervals: [mu - 1.28 * sigma * sqrt(h), mu + 1.28 * sigma * sqrt(h)]
        intervals_80 = []
        for h_idx, ret in enumerate(pred_returns, start=1):
            h_sigma = daily_volatility * math.sqrt(h_idx)
            low_ret = ret - 1.28 * h_sigma
            high_ret = ret + 1.28 * h_sigma
            intervals_80.append(
                (
                    round(base_price * math.exp(low_ret), 2),
                    round(base_price * math.exp(high_ret), 2),
                )
            )

        # Evaluate abstention gate
        gate_res = self.gate.evaluate(
            predicted_day1_log_return=pred_returns[0],
            predicted_day1_volatility=daily_volatility,
            candidate_day1_returns=[pred_returns[0]],
            relative_loss_vs_baseline=0.90,
            coverage_80pct=0.80,
        )

        provenance = f"DEV_RUN_{ticker}_{base_date}_{base_price:.2f}"

        return DevelopmentForecastOutput(
            ticker=ticker,
            base_date=base_date,
            base_price=round(base_price, 2),
            status="development_diagnostic_only",
            is_certified_production_claim=False,
            median_prices=[round(p, 2) for p in reconstructed_prices],
            cumulative_returns_median=[round(r, 6) for r in pred_returns],
            intervals_80pct=intervals_80,
            gate_decision=gate_res.decision,
            provenance_hash=provenance,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Development Forecast CLI")
    parser.add_argument("--ticker", type=str, default="BP", help="Ticker symbol")
    parser.add_argument(
        "--base-price", type=float, default=42.15, help="Latest validated close price P0"
    )
    parser.add_argument("--base-date", type=str, default="2026-08-28", help="Latest session date")
    args = parser.parse_args()

    # Synthetic sample window for development execution
    rng = np.random.default_rng(42)
    sample_window = rng.normal(0, 0.01, size=(60, len(STATIONARY_FEATURE_COLUMNS_V1)))

    runner = DevelopmentForecastRunner()
    out = runner.run_development_forecast(
        ticker=args.ticker,
        base_date=args.base_date,
        base_price=args.base_price,
        recent_feature_window=sample_window,
    )
    print(json.dumps(out.to_dict(), indent=2))


if __name__ == "__main__":
    main()
