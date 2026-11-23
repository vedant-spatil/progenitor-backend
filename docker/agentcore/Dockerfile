# AgentCore Runtime image for the agent graph (ADR 0008). Build from backend/:
#   docker buildx build --platform linux/arm64 -f docker/agentcore/Dockerfile -t progenitor-agent .
# AgentCore Runtime needs linux/arm64, 0.0.0.0:8080, POST /invocations and GET /ping
# (BedrockAgentCoreApp serves both). No Django, Postgres or Redis in this image's code path.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    # AgentCore Observability via ADOT. Never record prompt or completion text in spans (CLAUDE.md, ADR 0008).
    AGENT_OBSERVABILITY_ENABLED=true \
    OTEL_PYTHON_DISTRO=aws_distro \
    OTEL_PYTHON_CONFIGURATOR=aws_configurator \
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false

COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group agentcore --no-install-project

# Only the code the runtime needs: the graph, its workers, and the shared PHI filter.
COPY agents/ agents/
COPY mcp_servers/__init__.py mcp_servers/__init__.py
COPY mcp_servers/common/ mcp_servers/common/

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8080

CMD ["opentelemetry-instrument", "python", "-m", "agents.agentcore_app"]
