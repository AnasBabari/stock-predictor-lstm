#!/usr/bin/env python3
"""Print acquisition status and a written-permission request for OHLCV data.

This tool neither downloads data nor creates a certification manifest. Presence
of credentials proves access only; it never proves model-training or deployment
rights.
"""

from __future__ import annotations

import argparse
import json
import os

ALPACA_PERMISSION_TEMPLATE = """Subject: Request for written market-data permission

Please confirm whether historical daily US-equity OHLCV obtained through my
account may be used for academic machine-learning training, backtesting,
publication of derived aggregate metrics, distribution of trained model weights,
and public non-commercial inference, without redistributing raw market data.

Please identify the governing agreement, dataset/feed, permitted retention
period, attribution requirements, and any restrictions on derived models.
"""


def acquisition_status() -> dict[str, object]:
    """Report available acquisition routes without exposing credential values."""

    return {
        "artifact_role": "candidate_acquisition_status",
        "certification_eligible": False,
        "permission_status": "permission_pending",
        "routes": {
            "wrds_crsp": {
                "credentials_present": bool(os.environ.get("WRDS_USERNAME", "").strip()),
                "next": "Confirm institutional subscription and governing licence in writing.",
            },
            "alpaca": {
                "credentials_present": bool(
                    os.environ.get("ALPACA_API_KEY", "").strip()
                    and os.environ.get("ALPACA_API_SECRET", "").strip()
                ),
                "next": "Obtain written permission covering the exact intended uses.",
            },
            "alpha_vantage": {
                "credentials_present": bool(os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()),
                "next": "Obtain written permission covering the exact intended uses.",
            },
        },
        "next": "Acquire external rights evidence; do not relabel candidate data as licensed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permission-template", action="store_true")
    args = parser.parse_args()
    print(json.dumps(acquisition_status(), indent=2, sort_keys=True))
    if args.permission_template:
        print(ALPACA_PERMISSION_TEMPLATE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
