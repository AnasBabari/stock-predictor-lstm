"""Operator-only model artifact preparation command."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence

from config import MAX_FORECAST_DAYS
from data_pipeline import fetch_data, prepare_return_data, preprocess
from model import load_or_train

MODEL_TYPES = ("lstm", "bilstm_attention_direction")
SUPPORTED_MODEL_TYPES = MODEL_TYPES + ("gru", "bilstm_attention_regression")


def normalise_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", ticker):
        raise argparse.ArgumentTypeError("ticker must match [A-Z0-9.\\-]{1,12}")
    return ticker


def _prepare_price(feature_df):
    return preprocess(feature_df, forecast_days=MAX_FORECAST_DAYS)


def _prepare_direction(feature_df):
    return prepare_return_data(feature_df, forecast_days=MAX_FORECAST_DAYS)


PREPARERS: dict[str, Callable] = {
    "lstm": _prepare_price,
    "gru": _prepare_price,
    "bilstm_attention_regression": _prepare_price,
    "bilstm_attention_direction": _prepare_direction,
}


def pretrain(tickers: Sequence[str], model_types: Sequence[str]) -> tuple[int, int]:
    """Prepare every requested artifact, continuing after independent failures."""
    successes = 0
    failures = 0
    for ticker in dict.fromkeys(tickers):
        try:
            feature_df, _prices, _dates, metadata = fetch_data(ticker)
        except Exception as err:
            failures += len(model_types)
            print(
                f"ERROR {ticker}: market data failed ({type(err).__name__}: {err})", file=sys.stderr
            )
            continue

        for model_type in dict.fromkeys(model_types):
            try:
                X_train, X_test, y_train, y_test, scaler, _train_dates, _test_dates = PREPARERS[
                    model_type
                ](feature_df)
                load_or_train(
                    ticker,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    scaler,
                    model_type,
                    feature_df,
                    metadata,
                    allow_stale_fallback=False,
                )
                successes += 1
                print(f"READY {ticker}/{model_type}")
            except Exception as err:
                failures += 1
                print(
                    f"ERROR {ticker}/{model_type}: {type(err).__name__}: {err}",
                    file=sys.stderr,
                )
    return successes, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare fresh model artifacts outside the public HTTP request path."
    )
    parser.add_argument(
        "--ticker",
        action="append",
        required=True,
        type=normalise_ticker,
        help="Approved ticker to prepare; repeat for more than one ticker.",
    )
    parser.add_argument(
        "--model-type",
        action="append",
        choices=SUPPORTED_MODEL_TYPES,
        help="Model type to prepare; defaults to both types and may be repeated.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_types = args.model_type or list(MODEL_TYPES)
    successes, failures = pretrain(args.ticker, model_types)
    print(f"SUMMARY ready={successes} failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
