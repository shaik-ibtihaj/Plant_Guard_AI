"""Small per-process fixed-window rate limiter for API requests."""

from __future__ import annotations

import threading
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Limit each client IP to a fixed number of requests per minute."""

    def __init__(self, app, requests_per_minute: int = 60) -> None:
        super().__init__(app)
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.requests_per_minute = requests_per_minute
        self._counts: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if not request.url.path.startswith("/api/v1/"):
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        window = int(time.time() // 60)
        with self._lock:
            current_window, count = self._counts.get(client_ip, (window, 0))
            if current_window != window:
                current_window, count = window, 0
            if count >= self.requests_per_minute:
                retry_after = max(1, 60 - int(time.time() % 60))
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )
            self._counts[client_ip] = (current_window, count + 1)
            if len(self._counts) > 10_000:
                self._counts = {
                    address: record
                    for address, record in self._counts.items()
                    if record[0] == window
                }
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        return response
