"""Outbound HTTP for MCP servers: cache, rate limit, retries, typed errors.

Every call goes through `OutboundClient.get`, which applies the PHI egress
filter to the free-text parameters before anything leaves the process.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol, cast

import httpx

from mcp_servers.common.phi import check_outbound

log = logging.getLogger(__name__)

USER_AGENT = "progenitor-mcp/0.1"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class UpstreamErrorKind(StrEnum):
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    HTTP_ERROR = "http_error"
    NOT_FOUND = "not_found"
    MALFORMED = "malformed"
    UNAVAILABLE = "unavailable"


class UpstreamError(Exception):
    def __init__(self, kind: UpstreamErrorKind, destination: str, detail: str = "") -> None:
        self.kind = kind
        self.destination = destination
        super().__init__(f"{destination}: {kind}{f' ({detail})' if detail else ''}")


class Cache(Protocol):
    async def get(self, key: str) -> bytes | str | None: ...
    async def set(self, key: str, value: str, ex: int | None = None) -> Any: ...


class RateLimiter:
    """Spaces requests at least 60/rate_per_min seconds apart (per process)."""

    def __init__(self, rate_per_min: float) -> None:
        self._interval = 60.0 / rate_per_min if rate_per_min > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        if not self._interval:
            return
        async with self._lock:
            now = time.monotonic()
            if now < self._next:
                await asyncio.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self._interval


def redis_cache_from_env() -> Cache | None:
    """Redis cache if REDIS_URL is set and MCP_CACHE_DISABLED is not; else no cache."""
    url = os.environ.get("REDIS_URL")
    if not url or os.environ.get("MCP_CACHE_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None
    import redis.asyncio as aioredis

    return cast(Cache, aioredis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2))


class OutboundClient:
    def __init__(
        self,
        destination: str,
        base_url: str,
        *,
        rate_per_min: float = 60.0,
        timeout: float = 30.0,
        max_attempts: int = 3,
        cache: Cache | None = None,
        cache_ttl: int = 24 * 3600,
        http: httpx.AsyncClient | None = None,
        backoff: float = 1.0,
    ) -> None:
        self.destination = destination
        self.base_url = base_url.rstrip("/")
        self.max_attempts = max_attempts
        self.cache = cache
        self.cache_ttl = cache_ttl
        self.backoff = backoff
        self._limiter = RateLimiter(rate_per_min)
        self._http = http or httpx.AsyncClient(timeout=timeout, headers={"User-Agent": USER_AGENT})

    async def aclose(self) -> None:
        await self._http.aclose()

    def _cache_key(self, path: str, params: Mapping[str, Any]) -> str:
        raw = json.dumps([self.destination, path, sorted(params.items())], default=str)
        return f"mcp:{self.destination}:{hashlib.sha256(raw.encode()).hexdigest()}"

    async def _cache_get(self, key: str) -> Any | None:
        if self.cache is None:
            return None
        try:
            hit = await self.cache.get(key)
        except Exception as exc:  # noqa: BLE001 - cache is best-effort
            log.debug("cache get failed (%s)", exc.__class__.__name__)
            return None
        return json.loads(hit) if hit else None

    async def _cache_set(self, key: str, value: Any) -> None:
        if self.cache is None:
            return
        try:
            await self.cache.set(key, json.dumps(value), ex=self.cache_ttl)
        except Exception as exc:  # noqa: BLE001
            log.debug("cache set failed (%s)", exc.__class__.__name__)

    async def get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        free_text: Sequence[str] = (),
        parse: str = "json",
    ) -> Any:
        """GET base_url+path. `free_text` names the params checked by the PHI filter.

        Structured params (ids, page tokens, sizes) must be validated by the caller.
        """
        params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        check_outbound({k: params[k] for k in free_text if k in params}, destination=self.destination)

        key = self._cache_key(path, params)
        cached = await self._cache_get(key)
        if cached is not None:
            return cached

        last: UpstreamError | None = None
        for attempt in range(1, self.max_attempts + 1):
            await self._limiter.wait()
            try:
                resp = await self._http.get(f"{self.base_url}{path}", params=params)
            except httpx.TimeoutException:
                last = UpstreamError(UpstreamErrorKind.TIMEOUT, self.destination)
            except httpx.TransportError as exc:
                last = UpstreamError(UpstreamErrorKind.UNAVAILABLE, self.destination, exc.__class__.__name__)
            else:
                if resp.status_code == 404:
                    raise UpstreamError(UpstreamErrorKind.NOT_FOUND, self.destination)
                if resp.status_code in RETRY_STATUSES:
                    kind = (
                        UpstreamErrorKind.RATE_LIMITED
                        if resp.status_code == 429
                        else UpstreamErrorKind.HTTP_ERROR
                    )
                    last = UpstreamError(kind, self.destination, str(resp.status_code))
                    retry_after = resp.headers.get("Retry-After", "")
                    if attempt < self.max_attempts:
                        delay = float(retry_after) if retry_after.isdigit() else self.backoff * attempt
                        await asyncio.sleep(min(delay, 30.0))
                    continue
                if resp.status_code >= 400:
                    raise UpstreamError(UpstreamErrorKind.HTTP_ERROR, self.destination, str(resp.status_code))
                try:
                    body: Any = resp.json() if parse == "json" else resp.text
                except ValueError as exc:
                    raise UpstreamError(
                        UpstreamErrorKind.MALFORMED, self.destination, "invalid JSON"
                    ) from exc
                await self._cache_set(key, body)
                return body
            if attempt < self.max_attempts:
                await asyncio.sleep(self.backoff * attempt)
        if last is None:  # unreachable: every failed attempt sets `last`
            raise UpstreamError(UpstreamErrorKind.UNAVAILABLE, self.destination)
        raise last
