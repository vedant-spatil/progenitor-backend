"""Terminology MCP server: RxNorm (RxNav), LOINC and ICD-10-CM (NLM Clinical Tables).

Tools: normalize_drug, drug_classes, lookup_loinc, lookup_icd10. All free, no keys.
"""

from __future__ import annotations

import os
from typing import Any

from fastmcp import FastMCP

from mcp_servers.common.http import OutboundClient, UpstreamError, UpstreamErrorKind, redis_cache_from_env
from mcp_servers.common.server import add_health_route, require, run, tool_errors

RXNAV_URL = "https://rxnav.nlm.nih.gov/REST"
TABLES_URL = "https://clinicaltables.nlm.nih.gov/api"

# Common labs used by trial criteria. Codes verified against NLM Clinical Tables (LOINC) on 2026-11-16.
COMMON_LOINC: dict[str, tuple[str, str]] = {
    "hba1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood"),
    "hemoglobin a1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood"),
    "creatinine": ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma"),
    "egfr": (
        "98979-8",
        "Glomerular filtration rate [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based "
        "formula (CKD-EPI 2021)/1.73 sq M",
    ),
    "ldl": ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation"),
    "bmi": ("39156-5", "Body mass index (BMI) [Ratio]"),
    "systolic bp": ("8480-6", "Systolic blood pressure"),
    "diastolic bp": ("8462-4", "Diastolic blood pressure"),
    "hemoglobin": ("718-7", "Hemoglobin [Mass/volume] in Blood"),
    "platelets": ("777-3", "Platelets [#/volume] in Blood by Automated count"),
}


def _tables_rows(data: Any) -> list[tuple[str, str]]:
    """Clinical Tables returns [total, codes, extra, [[code, name], ...]]."""
    if not isinstance(data, list) or len(data) < 4 or not isinstance(data[3], list):
        raise UpstreamError(UpstreamErrorKind.MALFORMED, "clinicaltables.nlm.nih.gov", "unexpected shape")
    return [(str(r[0]), str(r[1])) for r in data[3] if isinstance(r, list) and len(r) >= 2]


def build_server(rxnav: OutboundClient | None = None, tables: OutboundClient | None = None) -> FastMCP:
    rate = float(os.environ.get("NLM_RATE_PER_MIN", "120"))
    rxnav = rxnav or OutboundClient(
        "rxnav.nlm.nih.gov", RXNAV_URL, rate_per_min=rate, cache=redis_cache_from_env()
    )
    tables = tables or OutboundClient(
        "clinicaltables.nlm.nih.gov", TABLES_URL, rate_per_min=rate, cache=redis_cache_from_env()
    )
    mcp = FastMCP(
        "terminology", instructions="Normalize drugs (RxNorm), labs (LOINC), and conditions (ICD-10-CM)."
    )
    add_health_route(mcp)

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
    @tool_errors
    async def normalize_drug(name: str, max_results: int = 5) -> dict[str, Any]:
        """Map a drug name (brand or generic, misspellings ok) to RxNorm concepts (rxcui, name, score)."""
        data = await rxnav.get(
            "/approximateTerm.json",
            {"term": name, "maxEntries": max(1, min(max_results, 20))},
            free_text=("term",),
        )
        if not isinstance(data, dict):
            raise UpstreamError(UpstreamErrorKind.MALFORMED, rxnav.destination)
        seen: dict[str, dict[str, Any]] = {}
        for c in data.get("approximateGroup", {}).get("candidate", []) or []:
            rxcui = c.get("rxcui")
            if not rxcui:
                continue
            entry = seen.setdefault(rxcui, {"rxcui": rxcui, "name": None, "score": float(c.get("score", 0))})
            if c.get("source") == "RXNORM" and c.get("name"):
                entry["name"] = c["name"]
            elif entry["name"] is None and c.get("name"):
                entry["name"] = c["name"]
        candidates = sorted(seen.values(), key=lambda e: -e["score"])[:max_results]
        return {"query": name, "candidates": candidates}

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
    @tool_errors
    async def drug_classes(rxcui: str, source: str = "ATC") -> dict[str, Any]:
        """Drug classes for an RxCUI, e.g. to test 'no insulin' (ATC A10A). source: ATC, VA, MESHPA, EPC."""
        rxcui = require(r"\d{1,9}", rxcui, "rxcui")
        source = require(r"[A-Z]{2,10}", source.upper(), "source")
        data = await rxnav.get("/rxclass/class/byRxcui.json", {"rxcui": rxcui, "relaSource": source})
        if not isinstance(data, dict):
            raise UpstreamError(UpstreamErrorKind.MALFORMED, rxnav.destination)
        infos = (data.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo", []) or []
        classes = {
            (i["rxclassMinConceptItem"]["classId"], i["rxclassMinConceptItem"]["className"])
            for i in infos
            if i.get("rxclassMinConceptItem")
        }
        return {
            "rxcui": rxcui,
            "source": source,
            "classes": [{"class_id": cid, "name": name} for cid, name in sorted(classes)],
        }

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
    @tool_errors
    async def lookup_loinc(name: str, max_results: int = 5) -> dict[str, Any]:
        """LOINC codes for a lab or vital sign name (e.g. 'hba1c', 'egfr', 'creatinine')."""
        key = name.strip().lower()
        common = COMMON_LOINC.get(key)
        data = await tables.get(
            "/loinc_items/v3/search",
            {"terms": name, "df": "LOINC_NUM,LONG_COMMON_NAME", "maxList": max(1, min(max_results, 20))},
            free_text=("terms",),
        )
        matches = [{"code": c, "name": n} for c, n in _tables_rows(data)]
        return {
            "query": name,
            "preferred": {"code": common[0], "name": common[1]} if common else None,
            "matches": matches,
        }

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
    @tool_errors
    async def lookup_icd10(text: str, max_results: int = 5) -> dict[str, Any]:
        """ICD-10-CM codes for a condition description (e.g. 'type 2 diabetes')."""
        data = await tables.get(
            "/icd10cm/v3/search",
            {"sf": "code,name", "terms": text, "maxList": max(1, min(max_results, 20))},
            free_text=("terms",),
        )
        return {"query": text, "matches": [{"code": c, "name": n} for c, n in _tables_rows(data)]}

    return mcp


if __name__ == "__main__":
    run(build_server(), default_port=8102)
