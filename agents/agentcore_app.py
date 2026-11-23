"""AgentCore Runtime entrypoint (ADR 0008). Run in the runtime container: `python -m agents.agentcore_app`.

Public network mode: AgentCore Memory holds checkpoints (IAM), the LLM is reached over the internet
(`LLM_PROVIDER`, e.g. groq), and the MCP servers on EC2 through CloudFront `/internal/mcp/*` with
AGENT_SERVICE_TOKEN (agents.tools.connections_from_env). The runtime never reaches Postgres or Redis:
it streams events and the Celery caller writes the rows (apps.runs.stream).

Env: AGENTCORE_MEMORY_ID, AWS_REGION, AGENT_SERVICE_TOKEN, CT_MCP_URL, TERM_MCP_URL, PUBMED_MCP_URL,
FHIR_MCP_URL, LLM_PROVIDER and the provider's key/model settings.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langgraph_checkpoint_aws import AgentCoreMemorySaver
from pydantic import ValidationError

from agents.events import EventChannel
from agents.graph import Deps
from agents.live_deps import live_deps
from agents.runtime_step import StepRequest, run_step

app = BedrockAgentCoreApp()

# Every checkpoint read and write needs an actor id; all runs belong to the backend service.
ACTOR_ID = "progenitor-backend"


def _saver() -> AgentCoreMemorySaver:
    return AgentCoreMemorySaver(os.environ["AGENTCORE_MEMORY_ID"], region_name=os.environ.get("AWS_REGION"))


@asynccontextmanager
async def _deps(req: StepRequest, ch: EventChannel) -> AsyncIterator[Deps]:
    async with live_deps(req.run_id, req.patient_fhir_id, ch.emit, ch) as deps:
        yield deps


async def invoke(payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    try:
        req = StepRequest.model_validate(payload)
    except ValidationError:
        yield {"type": "error", "error": "BadRequest"}
        return
    async for event in run_step(req, _deps, _saver(), {"actor_id": ACTOR_ID}):
        yield event


# Registered without decorator syntax so mypy stays strict when the agentcore group is not installed (CI).
app.entrypoint(invoke)

if __name__ == "__main__":
    app.run()
