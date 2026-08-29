"""Tests ensuring prospective/protocol contracts remain importable without PyTorch.

This test file prevents regressions in the nightly v7 maturity monitor and
other lightweight verification environments that do not install PyTorch.
"""

from __future__ import annotations

import subprocess
import sys


def test_prospective_and_contracts_import_without_torch() -> None:
    code = (
        "import sys\n"
        "from research.volatility_forecasting.prospective import ProspectiveCycleSettings, prospective_protocol\n"
        "from research.volatility_forecasting.contracts import VolatilityForecastProtocol, VolatilityLossWeights\n"
        "from research.volatility_forecasting.folds import build_prospective_certification_fold_plan\n"
        "from research.volatility_forecasting.cache import load_example_cache\n"
        "cycle = ProspectiveCycleSettings()\n"
        "protocol = prospective_protocol()\n"
        "weights = VolatilityLossWeights()\n"
        "assert 'torch' not in sys.modules, f'torch was unexpectedly imported: {sys.modules.get(\"torch\")}'\n"
        "print('TORCH_FREE_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "TORCH_FREE_OK" in result.stdout
