"""Render runtime entrypoint that prepares missing hosted artifacts before serving."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from config import MODEL_DIR
from pretrain import MODEL_TYPES, normalise_ticker


def _requested_tickers() -> list[str]:
    raw = os.getenv("RENDER_PRETRAIN_TICKERS", "AAPL,MSFT,TSLA")
    tickers = [normalise_ticker(value) for value in raw.split(",") if value.strip()]
    if not tickers:
        raise RuntimeError("RENDER_PRETRAIN_TICKERS must contain at least one ticker.")
    return list(dict.fromkeys(tickers))


def _missing_pairs(tickers: list[str]) -> list[tuple[str, str]]:
    root = Path(MODEL_DIR)
    return [
        (ticker, model_type)
        for ticker in tickers
        for model_type in MODEL_TYPES
        if not (root / ticker / model_type / "current.json").is_file()
    ]


def prepare_missing_artifacts() -> None:
    tickers = _requested_tickers()
    missing = _missing_pairs(tickers)
    if not missing:
        print("READY hosted model artifacts already exist", flush=True)
        return
    args = [sys.executable, "backend/pretrain.py"]
    for ticker in dict.fromkeys(ticker for ticker, _ in missing):
        args.extend(("--ticker", ticker))
    for model_type in MODEL_TYPES:
        if any(model == model_type for _, model in missing):
            args.extend(("--model-type", model_type))
    print(f"Preparing {len(missing)} hosted model artifacts.", flush=True)
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Hosted artifact preparation failed with exit code {result.returncode}.")


def main() -> None:
    prepare_missing_artifacts()
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api:app",
            "--app-dir",
            "backend",
            "--host",
            "0.0.0.0",  # nosec B104 - Render requires binding to its public interface
            "--port",
            os.getenv("PORT", "8000"),
        ],
    )


if __name__ == "__main__":
    main()
