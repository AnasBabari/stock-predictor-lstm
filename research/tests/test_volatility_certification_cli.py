from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest
from volatility_forecasting.contracts import VolatilityForecastProtocol

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "certify_volatility_candidate.py"
SPEC = importlib.util.spec_from_file_location("certify_volatility_candidate", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _report() -> dict[str, object]:
    protocol = VolatilityForecastProtocol()
    folds = [{"best_epoch": value} for value in (5, 8, 9, 4, 8)]
    records = [{"seed": seed, "folds": folds} for seed in protocol.seeds]
    return {
        "certifiable": True,
        "mode": "full_development",
        "protocol": asdict(protocol),
        "architecture": {
            "feature_count": protocol.feature_count,
            "horizon_count": len(protocol.horizons),
            "window_size": protocol.window_size,
        },
        "seeds": records,
        "seed_consensus": {
            str(horizon): {"promoted_all_seeds": horizon in (1, 3, 5, 7, 14)}
            for horizon in protocol.horizons
        },
    }


def test_report_validation_freezes_seed_coverage_and_eligible_horizons() -> None:
    protocol = VolatilityForecastProtocol()
    records, architecture, eligible = MODULE.validate_development_report(_report(), protocol)
    assert tuple(records) == protocol.seeds
    assert architecture.encoder_family == "tcn"
    assert eligible == (1, 3, 5, 7, 14)


def test_report_validation_rejects_noncertifiable_and_missing_seed_evidence() -> None:
    protocol = VolatilityForecastProtocol()
    report = _report()
    report["certifiable"] = False
    with pytest.raises(ValueError, match="non-certifiable"):
        MODULE.validate_development_report(report, protocol)
    report = _report()
    report["seeds"] = report["seeds"][:-1]
    with pytest.raises(ValueError, match="frozen seeds"):
        MODULE.validate_development_report(report, protocol)
