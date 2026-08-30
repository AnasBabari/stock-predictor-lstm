"""Stable point-in-time security identifier resolution handling historical ticker transitions and active index intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StableSecurityIdentity:
    security_id: str  # Primary permanent identifier e.g. "US.META"
    exchange_mic: str  # e.g. "XNAS"
    provider_aliases: tuple[str, ...]  # e.g. ("FB", "META")
    ticker_intervals: tuple[tuple[str, str, str], ...]  # ((ticker, start_date, end_date), ...)
    active_membership_intervals: tuple[tuple[str, str], ...]  # ((start_date, end_date), ...)

    def to_dict(self) -> dict[str, Any]:
        return {
            "security_id": self.security_id,
            "exchange_mic": self.exchange_mic,
            "provider_aliases": list(self.provider_aliases),
            "ticker_intervals": [list(t) for t in self.ticker_intervals],
            "active_membership_intervals": [list(t) for t in self.active_membership_intervals],
        }


class SecurityIdentityResolver:
    """Resolves point-in-time ticker symbols to stable security IDs with fail-closed semantics."""

    def __init__(self, identities: list[StableSecurityIdentity]) -> None:
        self._identities = {ident.security_id: ident for ident in identities}
        self._ticker_map: dict[str, list[tuple[str, str, str]]] = {}
        for ident in identities:
            for ticker, start_d, end_d in ident.ticker_intervals:
                self._ticker_map.setdefault(ticker.upper(), []).append(
                    (start_d, end_d, ident.security_id)
                )

    def resolve_ticker_to_security_id(self, ticker: str, date_str: str) -> str | None:
        """Returns stable security_id for ticker at date_str, or None if unresolved (fail closed)."""
        t_upper = ticker.upper()
        if t_upper in self._ticker_map:
            for start_d, end_d, sec_id in self._ticker_map[t_upper]:
                if start_d <= date_str <= end_d:
                    return sec_id
        return None

    def is_active_constituent(self, security_id: str, date_str: str) -> bool:
        """Checks if security_id was an active index member at date_str."""
        if security_id not in self._identities:
            return False
        ident = self._identities[security_id]
        return any(
            start_d <= date_str <= end_d for start_d, end_d in ident.active_membership_intervals
        )

    def get_identity(self, security_id: str) -> StableSecurityIdentity | None:
        return self._identities.get(security_id)
