"""Prepare missing hosted artifacts when the existing Render start command is used."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from config import MODEL_DIR
from pretrain import MODEL_TYPES, normalise_ticker


def prepare_hosted_artifacts() -> None:
    """Prepare approved artifacts only for the production Render disk."""
    if not os.getenv("RENDER_PRETRAIN_TICKERS"):
        return
        return
    raw = os.getenv("RENDER_PRETRAIN_TICKERS", "AAPL,MSFT,TSLA")
    tickers = list(
        dict.fromkeys(normalise_ticker(value) for value in raw.split(",") if value.strip())
    )
    root = Path(MODEL_DIR)
    missing = [
        (ticker, model_type)
        for ticker in tickers
        for model_type in MODEL_TYPES
        if not (root / ticker / model_type / "current.json").is_file()
    ]
    if not missing:
        return
    args = [sys.executable, "backend/pretrain.py"]
    for ticker in dict.fromkeys(ticker for ticker, _ in missing):
        args.extend(("--ticker", ticker))
    for model_type in MODEL_TYPES:
        if any(model == model_type for _, model in missing):
            args.extend(("--model-type", model_type))
    print(f"Preparing {len(missing)} hosted model artifacts before API startup.", flush=True)
    result = subprocess.run(args, check=False)
    if result.returncode:
        raise RuntimeError(
            f"Hosted artifact preparation failed with exit code {result.returncode}."
        )
