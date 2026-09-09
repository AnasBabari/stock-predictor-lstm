"""Provider interface: raw vendor output -> normalized strike frame."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class OptionsProvider(ABC):
    """Every provider yields NORMALIZED_COLUMNS grain, one date, >=1 ticker."""

    name: str = "base"

    @abstractmethod
    def read(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Return normalized strike records (see schemas.NORMALIZED_COLUMNS)."""
        raise NotImplementedError
