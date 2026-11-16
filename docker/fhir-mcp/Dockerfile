# wso2 FHIR MCP server, pinned. No pinned upstream image exists, so build from PyPI.
FROM python:3.12-slim
RUN pip install --no-cache-dir fhir-mcp-server==0.11.0 && useradd --create-home --uid 10001 app
USER app
ENV FHIR_MCP_HOST=0.0.0.0 FHIR_MCP_PORT=8000
EXPOSE 8000
CMD ["fhir-mcp-server", "--transport", "streamable-http"]
