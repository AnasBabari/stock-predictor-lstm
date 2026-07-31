"""Prepare missing hosted artifacts when the existing Render start command is used."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from config import MODEL_DIR
from pretrain import MODEL_TYPES, normalise_ticker


def prepare_hosted_artifacts() -> None:
    """Prepare approved artifacts only when boot-time preparation is explicitly enabled.

    Training must never run by accident in the 512MB free web instance.  Both opt-in flags
    are required; operators that want on-box training must set PREPARE_HOSTED_ARTIFACTS_ON_BOOT=1
    and RENDER_PRETRAIN_TICKERS together.
    """
    if os.getenv("PREPARE_HOSTED_ARTIFACTS_ON_BOOT", "0") != "1":
        return
    if not os.getenv("RENDER_PRETRAIN_TICKERS"):
        return
    raw = os.getenv("RENDER_PRETRAIN_TICKERS", "AAPL,MSFT,TSLA")
    model_types = [
        value.strip()
        for value in os.getenv("RENDER_PRETRAIN_MODEL_TYPES", "lstm").split(",")
        if value.strip()
    ]
    invalid = [value for value in model_types if value not in MODEL_TYPES]
    if invalid:
        raise RuntimeError(f"Unsupported hosted model types: {invalid}")
    tickers = list(
        dict.fromkeys(normalise_ticker(value) for value in raw.split(",") if value.strip())
    )
    root = Path(MODEL_DIR)
    missing = [
        (ticker, model_type)
        for ticker in tickers
        for model_type in model_types
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
