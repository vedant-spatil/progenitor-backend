"""Minimal FHIR client for loaders: transactions and counts against local HAPI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from data.loaders.common import SOURCE_TAG_SYSTEM, USER_AGENT

FHIR_JSON = "application/fhir+json"


def default_base_url() -> str:
    return os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir").rstrip("/")


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 502, 503, 504}
    return False


@dataclass
class TransactionResult:
    created: int = 0
    updated: int = 0
    statuses: dict[str, int] = field(default_factory=dict)

    def add(self, other: TransactionResult) -> None:
        self.created += other.created
        self.updated += other.updated
        for k, v in other.statuses.items():
            self.statuses[k] = self.statuses.get(k, 0) + v


class FhirClient:
    def __init__(self, base_url: str | None = None, timeout: float = 600.0) -> None:
        self.base_url = (base_url or default_base_url()).rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Accept": FHIR_JSON, "Content-Type": FHIR_JSON, "User-Agent": USER_AGENT},
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> FhirClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @retry(
        retry=retry_if_exception(_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, max=30),
        reraise=True,
    )
    def _post_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        resp = self._http.post("", json=bundle)
        if resp.status_code >= 400:
            # Surface HAPI's OperationOutcome diagnostics, which say which entry failed.
            if resp.status_code in {429, 502, 503, 504}:
                resp.raise_for_status()
            raise RuntimeError(f"FHIR transaction failed ({resp.status_code}): {resp.text[:2000]}")
        body: dict[str, Any] = resp.json()
        return body

    def transaction(self, entries: list[dict[str, Any]]) -> TransactionResult:
        """POST a transaction bundle. Entries must carry their own `request`."""
        result = TransactionResult()
        if not entries:
            return result
        body = self._post_bundle({"resourceType": "Bundle", "type": "transaction", "entry": entries})
        for entry in body.get("entry", []):
            status = str(entry.get("response", {}).get("status", "")).split(" ")[0]
            result.statuses[status] = result.statuses.get(status, 0) + 1
            if status == "201":
                result.created += 1
            elif status == "200":
                result.updated += 1
        return result

    def count(self, resource_type: str, params: dict[str, str] | None = None) -> int:
        query = {"_summary": "count", **(params or {})}
        resp = self._http.get(f"/{resource_type}", params=query)
        resp.raise_for_status()
        return int(resp.json().get("total", 0))

    def count_by_source(self, resource_type: str, source: str) -> int:
        return self.count(resource_type, {"_tag": f"{SOURCE_TAG_SYSTEM}|{source}"})

    def ids(self, resource_type: str, params: dict[str, str]) -> list[str]:
        """All ids matching a search (follows paging)."""
        out: list[str] = []
        url: str | None = f"/{resource_type}"
        query: dict[str, str] | None = {"_elements": "id", "_count": "500", **params}
        while url:
            resp = self._http.get(url, params=query)
            resp.raise_for_status()
            body = resp.json()
            out += [e["resource"]["id"] for e in body.get("entry", [])]
            nxt = next((link["url"] for link in body.get("link", []) if link.get("relation") == "next"), None)
            url, query = nxt, None
        return out


def add_source_tag(resource: dict[str, Any], source: str) -> None:
    meta = resource.setdefault("meta", {})
    tags = meta.setdefault("tag", [])
    if not any(t.get("system") == SOURCE_TAG_SYSTEM for t in tags):
        tags.append({"system": SOURCE_TAG_SYSTEM, "code": source})


def put_entry(resource: dict[str, Any]) -> dict[str, Any]:
    """Idempotent upsert: PUT <type>/<id>. Rerunning with the same content adds nothing."""
    rtype, rid = resource["resourceType"], resource["id"]
    return {
        "fullUrl": f"{rtype}/{rid}",
        "resource": resource,
        "request": {"method": "PUT", "url": f"{rtype}/{rid}"},
    }
