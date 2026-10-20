"""API-key authentication dependency for service endpoints."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request


async def require_api_key(
    request: Request,
    api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Require a configured API key; allow open local development by default."""
    settings = request.app.state.settings
    expected = settings.api_key
    if expected is None:
        if settings.environment.lower() == "production":
            raise HTTPException(
                status_code=503, detail="API authentication is unavailable"
            )
        return
    if api_key is None or not hmac.compare_digest(api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
