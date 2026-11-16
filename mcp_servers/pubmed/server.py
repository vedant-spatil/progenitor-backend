"""PubMed MCP server (NCBI E-utilities): search_pubmed, get_abstracts.

Our own server instead of mcp-simple-pubmed (see docs/decisions/0002-own-pubmed-mcp.md):
it pins fastmcp 2.x, runs as an unrecordable stdio subprocess, and its queries
would bypass our PHI egress filter.
"""

from __future__ import annotations

import os
from typing import Any

from defusedxml import ElementTree
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from mcp_servers.common.http import OutboundClient, UpstreamError, UpstreamErrorKind, redis_cache_from_env
from mcp_servers.common.server import add_health_route, require, run, tool_errors

EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DESTINATION = "eutils.ncbi.nlm.nih.gov"
PMID = r"\d{1,9}"


def _ncbi_params() -> dict[str, str]:
    """NCBI asks for tool + email; email is a project contact (PUBMED_EMAIL), never a personal one."""
    params = {"tool": "progenitor"}
    if email := os.environ.get("PUBMED_EMAIL"):
        params["email"] = email
    if key := os.environ.get("PUBMED_API_KEY"):
        params["api_key"] = key
    return params


def build_server(client: OutboundClient | None = None) -> FastMCP:
    # NCBI allows 3 req/s without a key, 10 with one.
    default_rate = "540" if os.environ.get("PUBMED_API_KEY") else "170"
    client = client or OutboundClient(
        DESTINATION,
        EUTILS_URL,
        rate_per_min=float(os.environ.get("PUBMED_RATE_PER_MIN", default_rate)),
        cache=redis_cache_from_env(),
    )
    mcp = FastMCP("pubmed", instructions="Search PubMed and read abstracts (NCBI E-utilities).")
    add_health_route(mcp)

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
    @tool_errors
    async def search_pubmed(query: str, max_results: int = 10, min_year: int | None = None) -> dict[str, Any]:
        """Search PubMed with condition/intervention terms (never patient identifiers).

        Returns PMIDs with title, journal, and publication year, newest first.
        """
        params: dict[str, Any] = {
            **_ncbi_params(),
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max(1, min(max_results, 50)),
            "sort": "pub_date",
        }
        if min_year:
            params.update({"datetype": "pdat", "mindate": str(int(min_year)), "maxdate": "3000"})
        found = await client.get("/esearch.fcgi", params, free_text=("term",))
        ids = (found.get("esearchresult") or {}).get("idlist") if isinstance(found, dict) else None
        if ids is None:
            raise UpstreamError(UpstreamErrorKind.MALFORMED, DESTINATION, "no esearchresult")
        if not ids:
            return {"query": query, "count": 0, "articles": []}
        summary = await client.get(
            "/esummary.fcgi", {**_ncbi_params(), "db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        )
        result = summary.get("result", {}) if isinstance(summary, dict) else {}
        articles = [
            {
                "pmid": pmid,
                "title": result.get(pmid, {}).get("title"),
                "journal": result.get(pmid, {}).get("fulljournalname") or result.get(pmid, {}).get("source"),
                "year": str(result.get(pmid, {}).get("pubdate", ""))[:4] or None,
            }
            for pmid in ids
            if pmid in result
        ]
        return {"query": query, "count": int(found["esearchresult"].get("count", 0)), "articles": articles}

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
    @tool_errors
    async def get_abstracts(pmids: list[str]) -> dict[str, Any]:
        """Title and abstract text for up to 20 PMIDs."""
        if not pmids or len(pmids) > 20:
            raise ToolError("pmids: give 1 to 20 ids")
        ids = [require(PMID, p, "pmid") for p in pmids]
        xml = await client.get(
            "/efetch.fcgi",
            {**_ncbi_params(), "db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"},
            parse="text",
        )
        try:
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError as exc:
            raise UpstreamError(UpstreamErrorKind.MALFORMED, DESTINATION, "invalid XML") from exc
        articles = []
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//MedlineCitation/PMID")
            title_node = art.find(".//ArticleTitle")
            title = "".join(title_node.itertext()) if title_node is not None else None
            parts = []
            for node in art.iter("AbstractText"):
                label = node.get("Label")
                text = "".join(node.itertext()).strip()
                parts.append(f"{label}: {text}" if label else text)
            articles.append({"pmid": pmid, "title": title, "abstract": "\n".join(parts) or None})
        return {"articles": articles}

    return mcp


if __name__ == "__main__":
    run(build_server(), default_port=8103)
