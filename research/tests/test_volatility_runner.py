from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_volatility_research.py"
SPEC = importlib.util.spec_from_file_location("run_volatility_research", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_exposure_map_is_normalized_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "exposures.json"
    path.write_text(json.dumps({"nmm": {"Shipping_Disruption": 0.8}}), encoding="utf-8")
    assert RUNNER._load_exposure_map(path) == {"NMM": {"shipping_disruption": 0.8}}

    path.write_text(json.dumps({"NMM": {"oil_supply": 1.1}}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RUNNER._load_exposure_map(path)


def test_news_consensus_binds_reasons_to_the_correct_seed() -> None:
    records = [
        [
            {
                "promoted": True,
                "relative_qlike_to_market": 0.8,
                "reasons": (),
            }
        ],
        [
            {
                "promoted": False,
                "relative_qlike_to_market": 1.02,
                "reasons": ("no improvement",),
            }
        ],
    ]
    consensus = RUNNER._news_ablation_consensus(records, (41, 42), (7,))
    seven_day = consensus["7"]
    assert seven_day["promoted_all_seeds"] is False
    assert seven_day["reasons_by_seed"] == {"41": [], "42": ["no improvement"]}
