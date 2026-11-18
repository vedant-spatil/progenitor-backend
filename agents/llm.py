"""LLM access for agents: provider switch, structured output, cache, bounded retries.

Providers (LLM_PROVIDER):
- bedrock: Claude on Amazon Bedrock (bedrock-mantle endpoint) through the official
  `anthropic` SDK. Structured output via `messages.parse(output_format=<Pydantic model>)`.
  Default model Claude Opus 5.5. Thinking is always on and sampling parameters are not
  accepted, so there is no temperature setting; depth is set with `effort`.
  Auth: AWS_BEARER_TOKEN_BEDROCK (Bedrock API key) or the AWS credential chain.
- bedrock-openai: non-Claude Bedrock models (gpt-oss, Kimi, GLM, Mistral) via Bedrock's
  OpenAI-compatible endpoint with the Bedrock API key; LLM_MODEL is required.
- groq, ollama: OpenAI-compatible chat API. Structured output by forcing one tool whose
  parameters are the Pydantic JSON schema. Temperature 0.

At most 3 provider calls per request; a reply that fails schema validation is re-asked
once (OpenAI-compatible path). Prompts and replies are never logged at INFO.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from agents.telemetry import emit

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Groq model list checked 2026-11-16: llama-3.3-70b-versatile retired; qwen3.8-27b passes the structured-output probe.
DEFAULT_MODELS = {"groq": "qwen/qwen3.8-27b", "ollama": "qwen2.5:14b", "bedrock": "anthropic.claude-opus-5-5"}
# Client-side refusal fallback on Bedrock (server-side `fallbacks` is not offered there).
DEFAULT_BEDROCK_FALLBACK = "anthropic.claude-opus-4-8"
BASE_URLS = {"groq": "https://api.groq.com/openai/v1", "ollama": "http://localhost:11434/v1"}


class LlmError(RuntimeError):
    pass


class LlmCache(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...
    def set(self, key: str, value: dict[str, Any]) -> None: ...


class StructuredLLM(Protocol):
    model: str
    usage: Usage

    def complete(self, *, task: str, system: str, user: str, schema: type[T]) -> T: ...


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


def cache_key(provider: str, model: str, system: str, user: str, schema: dict[str, Any]) -> str:
    raw = json.dumps([provider, model, system, user, schema], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


class TokenPacer:
    """Client-side tokens-per-minute budget, so free-tier limits are not hit (fewer 429s).

    Estimates a request's input tokens from its size (~3 chars per token) and waits
    until the last 60 s of usage plus this request fits the budget.
    """

    def __init__(self, tokens_per_min: int, clock: Any = time.monotonic, sleep: Any = time.sleep) -> None:
        self.budget = tokens_per_min
        self.clock = clock
        self.sleep = sleep
        self.window: deque[tuple[float, int]] = deque()

    def acquire(self, tokens: int) -> None:
        if self.budget <= 0:
            return
        tokens = min(tokens, self.budget)
        while True:
            now = self.clock()
            while self.window and now - self.window[0][0] >= 60:
                self.window.popleft()
            used = sum(t for _, t in self.window)
            if used + tokens <= self.budget:
                self.window.append((now, tokens))
                return
            self.sleep(60 - (now - self.window[0][0]) + 0.5)


@dataclass
class ChatLLM:
    """OpenAI-compatible chat completions with a forced tool call (Groq, Ollama)."""

    provider: str
    model: str
    base_url: str
    api_key: str = ""
    max_tokens: int = 2048
    timeout: float = 120.0
    cache: LlmCache | None = None
    http: httpx.Client | None = None
    sleep: Any = time.sleep
    usage: Usage = field(default_factory=Usage)
    pacer: TokenPacer | None = None

    def _body(
        self, messages: list[dict[str, str]], name: str, schema: dict[str, Any], max_tokens: int
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "Return the answer in this structure.",
                        "parameters": schema,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": name}},
        }

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """One logical call: up to 3 provider attempts with backoff."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        client = self.http or httpx.Client(timeout=self.timeout)
        last: Exception | None = None
        estimate = len(json.dumps(body)) // 3 + int(body.get("max_tokens", 0))
        try:
            for attempt in range(3):
                if self.pacer:
                    self.pacer.acquire(estimate)
                try:
                    resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
                except httpx.TransportError as exc:
                    last = exc
                    self.sleep(2.0 * (attempt + 1))
                    continue
                self.usage.calls += 1
                if resp.is_success:
                    payload: dict[str, Any] = resp.json()
                    usage = payload.get("usage") or {}
                    self.usage.tokens_in += int(usage.get("prompt_tokens", 0))
                    self.usage.tokens_out += int(usage.get("completion_tokens", 0))
                    return payload
                if resp.status_code in (429, 500, 502, 503, 504):
                    last = LlmError(f"HTTP {resp.status_code}")
                    retry_after = resp.headers.get("retry-after", "")
                    wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 0.0
                    # Groq free tier (~tokens/min per model) clears within about a minute.
                    self.sleep(max(wait, 20.0 * (attempt + 1) if resp.status_code == 429 else 2.0 * (attempt + 1)))
                    continue
                if resp.status_code == 400 and "tool_use_failed" in resp.text:
                    # Usually a reply cut off at the token limit: retry with 1.5x budget.
                    body = {**body, "max_tokens": int(body["max_tokens"] * 1.5)}
                    last = LlmError("tool_use_failed")
                    continue
                raise LlmError(f"rejected: HTTP {resp.status_code}: {resp.text[:300]}")
        finally:
            if self.http is None:
                client.close()
        raise LlmError(f"failed after 3 attempts: {last}")

    @staticmethod
    def _tool_args(payload: dict[str, Any], name: str) -> str:
        for choice in payload.get("choices", []):
            for call in choice.get("message", {}).get("tool_calls") or []:
                fn = call.get("function", {})
                if fn.get("name") == name:
                    return str(fn.get("arguments") or "{}")
        raise LlmError(f"no tool call for {name}")

    def complete(self, *, task: str, system: str, user: str, schema: type[T]) -> T:
        name = schema.__name__
        json_schema = schema.model_json_schema()
        key = cache_key(self.provider, self.model, system, user, json_schema)
        started = time.perf_counter()
        if self.cache and (hit := self.cache.get(key)) is not None:
            self.usage.cache_hits += 1
            emit("llm_call", task=task, model=self.model, cache_hit=True, duration_ms=0)
            return schema.model_validate(hit)

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        result: T | None = None
        error = ""
        for reask in range(2):
            raw = self._tool_args(self._post(self._body(messages, name, json_schema, self.max_tokens)), name)
            try:
                result = schema.model_validate_json(raw)
                break
            except ValidationError as exc:
                error = exc.__class__.__name__
                if reask == 0:
                    # Re-ask once with the validation errors (field paths and messages only).
                    detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:10])
                    messages += [
                        {"role": "assistant", "content": raw[:4000]},
                        {
                            "role": "user",
                            "content": f"That reply did not match the schema ({detail}). Return it again, fixed.",
                        },
                    ]
        emit(
            "llm_call",
            task=task,
            model=self.model,
            cache_hit=False,
            status="ok" if result else "schema_error",
            duration_ms=round((time.perf_counter() - started) * 1000),
        )
        if result is None:
            raise LlmError(f"{task}: reply failed schema validation twice ({error})")
        if self.cache:
            self.cache.set(key, result.model_dump(mode="json"))
        return result


@dataclass
class BedrockLLM:
    """Claude on Amazon Bedrock via the anthropic SDK's Mantle client, with Pydantic structured output."""

    model: str
    client: Any
    effort: str = "medium"
    max_tokens: int = 16000
    cache: LlmCache | None = None
    provider: str = "bedrock"
    usage: Usage = field(default_factory=Usage)

    def complete(self, *, task: str, system: str, user: str, schema: type[T]) -> T:
        key = cache_key(self.provider, f"{self.model}@{self.effort}", system, user, schema.model_json_schema())
        started = time.perf_counter()
        if self.cache and (hit := self.cache.get(key)) is not None:
            self.usage.cache_hits += 1
            emit("llm_call", task=task, model=self.model, cache_hit=True, duration_ms=0)
            return schema.model_validate(hit)

        import anthropic

        status = "ok"
        try:
            resp = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": self.effort},
            )
        except anthropic.APIStatusError as exc:
            status = f"http_{exc.status_code}"
            raise LlmError(f"{task}: Bedrock rejected the request (HTTP {exc.status_code})") from exc
        except anthropic.APIConnectionError as exc:
            status = "connection_error"
            raise LlmError(f"{task}: cannot reach Bedrock") from exc
        finally:
            if status != "ok":
                emit("llm_call", task=task, model=self.model, cache_hit=False, status=status)
        self.usage.calls += 1
        self.usage.tokens_in += int(getattr(resp.usage, "input_tokens", 0) or 0)
        self.usage.tokens_out += int(getattr(resp.usage, "output_tokens", 0) or 0)
        result = resp.parsed_output
        if resp.stop_reason in ("refusal", "max_tokens") or not isinstance(result, schema):
            emit("llm_call", task=task, model=self.model, cache_hit=False, status=str(resp.stop_reason))
            raise LlmError(f"{task}: no structured reply (stop_reason={resp.stop_reason})")
        emit(
            "llm_call",
            task=task,
            model=self.model,
            cache_hit=False,
            status="ok",
            duration_ms=round((time.perf_counter() - started) * 1000),
        )
        if self.cache:
            self.cache.set(key, result.model_dump(mode="json"))
        return result


def bedrock_client(region: str, fallback_model: str) -> Any:
    from anthropic import AnthropicBedrockMantle, BetaRefusalFallbackMiddleware

    middleware = [BetaRefusalFallbackMiddleware([{"model": fallback_model}])] if fallback_model else []
    # Credentials: AWS_BEARER_TOKEN_BEDROCK takes priority, else the AWS credential chain.
    return AnthropicBedrockMantle(aws_region=region, middleware=middleware, max_retries=3)


def get_llm(cache: LlmCache | None = None) -> StructuredLLM:
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if provider == "bedrock":
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
        if not region:
            raise LlmError("AWS_REGION is not set")
        has_creds = (
            os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            or os.environ.get("AWS_ACCESS_KEY_ID")
            or os.environ.get("AWS_PROFILE")
        )
        if not has_creds:
            raise LlmError("Bedrock credentials not set: AWS_BEARER_TOKEN_BEDROCK (Bedrock API key) or AWS credentials")
        fallback = os.environ.get("LLM_FALLBACK_MODEL", DEFAULT_BEDROCK_FALLBACK)
        return BedrockLLM(
            model=os.environ.get("LLM_MODEL") or DEFAULT_MODELS["bedrock"],
            client=bedrock_client(region, fallback),
            effort=os.environ.get("LLM_EFFORT", "medium"),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS_BEDROCK", "16000")),
            cache=cache,
        )
    if provider == "bedrock-openai":
        # Non-Claude Bedrock models (gpt-oss, Kimi, GLM, Mistral, ...) through Bedrock's OpenAI-compatible
        # endpoint, authenticated with the Bedrock API key. Same forced-tool structured output as Groq.
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
        token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
        if not (region and token):
            raise LlmError("bedrock-openai needs AWS_REGION and AWS_BEARER_TOKEN_BEDROCK")
        model = os.environ.get("LLM_MODEL", "")
        if not model:
            raise LlmError("bedrock-openai needs LLM_MODEL (e.g. global.moonshotai.kimi-k3.1)")
        return ChatLLM(
            provider=provider,
            model=model,
            base_url=f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
            api_key=token,
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
            cache=cache,
        )
    if provider not in BASE_URLS:
        raise LlmError(f"unknown LLM_PROVIDER {provider!r}")
    api_key = os.environ.get("GROQ_API_KEY", "") if provider == "groq" else ""
    if provider == "groq" and not api_key:
        raise LlmError("GROQ_API_KEY is not set")
    base = os.environ.get("OLLAMA_BASE_URL", "").rstrip("/") + "/v1" if provider == "ollama" else BASE_URLS["groq"]
    return ChatLLM(
        provider=provider,
        model=os.environ.get("LLM_MODEL") or DEFAULT_MODELS[provider],
        base_url=base,
        api_key=api_key,
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2048")),
        cache=cache,
        pacer=TokenPacer(int(os.environ.get("LLM_TOKENS_PER_MIN", "0"))),
    )
