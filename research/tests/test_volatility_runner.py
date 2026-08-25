from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_volatility_research.py"
SPEC = importlib.util.spec_from_file_location("run_volatility_research", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


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


def test_news_gap_mask_excludes_the_causal_lookback_not_just_the_gap_day() -> None:
    dates = np.array(
        ["2025-06-13", "2025-06-14", "2025-06-18", "2025-06-22", "2025-06-30", "2025-07-02"],
        dtype="datetime64[D]",
    )
    mask, gaps = RUNNER._news_gap_exclusion_mask(
        dates,
        {"provenance": {"missing_archive_dates": ["2025-06-14"]}},
    )
    assert gaps == ("2025-06-14",)
    assert mask.tolist() == [False, True, True, False, False, False]
