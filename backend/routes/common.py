"""Shared validation, rate limiting, and dependencies for API routers."""

from __future__ import annotations

import ipaddress
import re

from fastapi import HTTPException, Request
from slowapi import Limiter

from config import settings

VALID_MODEL_TYPES = {
    "lstm",
    "gru",
    "attention",
    "bilstm",
    "bilstm_attention_regression",
    "bilstm_attention_direction",
}

_trusted_proxy_ips = frozenset(settings.trusted_proxy_ips)


def _is_trusted_ip(value: str | None) -> bool:

    if not value:
        return False
    try:
        import api

        addr = ipaddress.ip_address(value.strip())
        trusted_set = set(getattr(api, "_trusted_proxy_ips", ())) | set(settings.trusted_proxy_ips)
        for trusted in trusted_set:
            trusted_str = str(trusted)
            if "/" in trusted_str:
                if addr in ipaddress.ip_network(trusted_str, strict=False):
                    return True
            elif str(addr) == trusted_str:
                return True
        return False
    except ValueError:
        return False


def _normalise_ip(value: str) -> str | None:
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def rate_limit_identity(request: Request) -> str:
    """Trust forwarding data only when the direct peer is explicitly configured."""
    peer = request.client.host if request.client is not None else "unknown"
    normalised_peer = _normalise_ip(peer)
    if normalised_peer is None or not _is_trusted_ip(normalised_peer):
        return normalised_peer or peer

    forwarded = request.headers.get("x-forwarded-for")
    if not forwarded:
        return normalised_peer
    hops = [_normalise_ip(value) for value in forwarded.split(",")]
    if any(hop is None for hop in hops):
        return normalised_peer
    for hop in reversed(hops):
        if hop is not None and not _is_trusted_ip(hop):
            return hop
    return normalised_peer


limiter = Limiter(key_func=rate_limit_identity)


def validate_ticker(ticker: str) -> str:
    """Sanitise and validate a ticker symbol (1.4)."""
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")
    return ticker


def validate_model_type(model_type: str) -> str:
    """Validate model type parameter to prevent path traversal."""
    model_type = model_type.strip().lower()
    if model_type not in VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Must be one of: {sorted(VALID_MODEL_TYPES)}",
        )
    return model_type
