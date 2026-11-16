"""Shared FastMCP helpers: health route, error mapping, runner."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Awaitable, Callable
from functools import wraps

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mcp_servers.common.http import UpstreamError
from mcp_servers.common.phi import PhiBlockedError


def add_health_route(mcp: FastMCP) -> None:
    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok", "server": mcp.name})


def tool_errors[**P, R](fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Turn upstream and PHI errors into ToolErrors with short, safe messages."""

    @wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await fn(*args, **kwargs)
        except PhiBlockedError as exc:
            raise ToolError(f"blocked: query contains identifiers ({', '.join(exc.types)})") from exc
        except UpstreamError as exc:
            raise ToolError(f"upstream {exc.kind}: {exc.destination}") from exc

    return wrapper


def require(pattern: str, value: str, field: str) -> str:
    """Validate a structured id (NCT id, RxCUI, PMID) so it can skip the free-text PHI check."""
    value = value.strip()
    if not re.fullmatch(pattern, value):
        raise ToolError(f"invalid {field}: expected pattern {pattern}")
    return value


def run(mcp: FastMCP, default_port: int) -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    mcp.run(
        transport="http",
        host=os.environ.get("MCP_HOST", "0.0.0.0"),  # noqa: S104  # nosec B104 - container-internal
        port=int(os.environ.get("MCP_PORT", str(default_port))),
        show_banner=False,
    )
