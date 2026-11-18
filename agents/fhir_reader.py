"""Read access to FHIR for workers: via the FHIR MCP tools (agents) or plain HTTP (evals, tests)."""

from __future__ import annotations

import json
from typing import Any, Protocol

import httpx

Resource = dict[str, Any]


class FhirReader(Protocol):
    async def read(self, resource_type: str, resource_id: str) -> Resource: ...

    async def search(self, resource_type: str, params: dict[str, str]) -> tuple[list[Resource], list[Resource]]:
        """Returns (matches, included). One page; callers pass `_count`."""
        ...


def split_bundle(bundle: Resource) -> tuple[list[Resource], list[Resource]]:
    matches: list[Resource] = []
    included: list[Resource] = []
    for entry in bundle.get("entry", []) or []:
        mode = (entry.get("search") or {}).get("mode", "match")
        (included if mode == "include" else matches).append(entry["resource"])
    return matches, included


class HttpFhirReader:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, headers={"Accept": "application/fhir+json"}
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def read(self, resource_type: str, resource_id: str) -> Resource:
        resp = await self._http.get(f"/{resource_type}/{resource_id}")
        resp.raise_for_status()
        body: Resource = resp.json()
        return body

    async def search(self, resource_type: str, params: dict[str, str]) -> tuple[list[Resource], list[Resource]]:
        resp = await self._http.get(f"/{resource_type}", params=params)
        resp.raise_for_status()
        return split_bundle(resp.json())


def _tool_json(result: Any) -> Resource:
    """langchain-mcp-adapters returns MCP content blocks; the FHIR server sends JSON text."""
    text = result[0]["text"] if isinstance(result, list) else result
    body: Resource = json.loads(text) if isinstance(text, str) else text
    return body


class McpFhirReader:
    """Uses the `fhir_read` and `fhir_search` tools from agents.tools.load_tools()."""

    def __init__(self, read_tool: Any, search_tool: Any) -> None:
        self._read = read_tool
        self._search = search_tool

    async def read(self, resource_type: str, resource_id: str) -> Resource:
        return _tool_json(await self._read.ainvoke({"type": resource_type, "id": resource_id}))

    async def search(self, resource_type: str, params: dict[str, str]) -> tuple[list[Resource], list[Resource]]:
        return split_bundle(_tool_json(await self._search.ainvoke({"type": resource_type, "searchParam": params})))
