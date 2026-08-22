from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_global_pipeline import (  # noqa: E402
    PipelineConfig,
    generate_synthetic_universe,
    run_pipeline,
)


def test_pipeline_executes_cleanly_on_micro_panel(tmp_path: Path) -> None:
    tickers = ["MIC1", "MIC2", "MIC3", "MIC4"]
    universe = generate_synthetic_universe(tickers, n_sessions=350, seed=42)
    cfg = PipelineConfig(
        run_id="test_micro_run",
        horizons=[1, 5],
        folds=2,
        embargo=2,
        min_train_sessions=200,
        candidate_families=["persistence", "ridge_global"],
        seeds=[42],
        resamples=100,
    )
    run_dir = tmp_path / "micro_run"
    results = run_pipeline(
        config=cfg,
        run_dir=run_dir,
        universe_data=universe,
        open_locked_certification_holdout=False,
    )

    assert results["run_id"] == "test_micro_run"
    assert "snapshot" in results["stages"]
    assert "features" in results["stages"]
    assert "folds" in results["stages"]
    assert "selection" in results["stages"]
    assert results["stages"]["certification"]["status"] == "locked_untouched"

    manifest_file = run_dir / "pipeline_manifest.json"
    assert manifest_file.exists()
    saved = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert saved["config_digest"] == cfg.digest()


def test_locked_certification_holdout_opens_with_explicit_flag(tmp_path: Path) -> None:
    tickers = ["MIC1", "MIC2", "MIC3", "MIC4"]
    universe = generate_synthetic_universe(tickers, n_sessions=350, seed=7)
    cfg = PipelineConfig(
        run_id="test_cert_run",
        horizons=[1],
        folds=2,
        embargo=2,
        min_train_sessions=200,
        candidate_families=["persistence"],
        seeds=[7],
        resamples=50,
    )
    run_dir = tmp_path / "cert_run"
    results = run_pipeline(
        config=cfg,
        run_dir=run_dir,
        universe_data=universe,
        open_locked_certification_holdout=True,
    )

    assert results["stages"]["certification"]["status"] == "holdout_opened"
    assert len(results["stages"]["certification"]["holdout_tickers"]) >= 1
