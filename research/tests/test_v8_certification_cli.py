from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from certify_v8_candidate import _v8_fold_plan, _write_json_atomic  # noqa: E402


def test_one_shot_marker_precedes_any_sealed_example_load() -> None:
    source = (SCRIPTS / "certify_v8_candidate.py").read_text(encoding="utf-8")
    marker = source.index('_write_json_atomic(out / "v8-holdout-opened.json", marker)')
    cache_load = source.index("examples = load_example_cache", marker)
    panel_build = source.index("examples = build_volatility_panel_examples", marker)
    assert marker < cache_load
    assert marker < panel_build


def test_atomic_json_writer_emits_strict_json(tmp_path) -> None:
    target = tmp_path / "evidence.json"
    _write_json_atomic(target, {"status": "passed", "metric": 0.95})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "metric": 0.95,
        "status": "passed",
    }
    assert not list(tmp_path.glob(".evidence.json-*"))


def test_v8_fold_adapter_preserves_exact_reserve_identities() -> None:
    split = type(
        "Split",
        (),
        {
            "train_tickers": ("AAA",),
            "holdout_tickers": ("MSFT",),
            "temporal_test_indices": np.asarray([2, 3], dtype=np.int64),
            "asset_transfer_test_indices": np.asarray([4], dtype=np.int64),
            "pooled_test_indices": np.asarray([2, 3, 4], dtype=np.int64),
        },
    )()
    examples = type(
        "Examples",
        (),
        {
            "origin_dates": np.asarray(
                ["2024-01-01", "2024-01-02", "2025-01-03", "2025-01-04", "2025-01-03"],
                dtype="datetime64[D]",
            )
        },
    )()
    plan = _v8_fold_plan(split, examples)
    np.testing.assert_array_equal(plan.temporal_certification_indices, [2, 3])
    np.testing.assert_array_equal(plan.asset_transfer_certification_indices, [4])
    assert plan.certification_start == np.datetime64("2025-01-03")
