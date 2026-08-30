"""Unit tests for SecurityIdentityResolver."""

from research.volatility_forecasting.stable_security_identity_v11 import (
    SecurityIdentityResolver,
    StableSecurityIdentity,
)


def test_stable_security_identity_resolution():
    meta_ident = StableSecurityIdentity(
        security_id="US.META",
        exchange_mic="XNAS",
        provider_aliases=("FB", "META"),
        ticker_intervals=(
            ("FB", "2012-05-18", "2022-06-08"),
            ("META", "2022-06-09", "2026-08-28"),
        ),
        active_membership_intervals=(("2012-05-18", "2026-08-28"),),
    )

    resolver = SecurityIdentityResolver([meta_ident])

    # 1. Historical FB ticker resolves to US.META
    assert resolver.resolve_ticker_to_security_id("FB", "2020-01-15") == "US.META"

    # 2. Modern META ticker resolves to US.META
    assert resolver.resolve_ticker_to_security_id("META", "2024-05-10") == "US.META"

    # 3. META before transition date fails closed (returns None)
    assert resolver.resolve_ticker_to_security_id("META", "2020-01-15") is None

    # 4. Unknown ticker returns None
    assert resolver.resolve_ticker_to_security_id("UNKNOWN", "2024-05-10") is None
