"""LLM access for agents: provider switch, structured output, cache, bounded retries.

Providers (LLM_PROVIDER):
- bedrock: Claude on Amazon Bedrock (bedrock-mantle endpoint) through the official
  `anthropic` SDK. Structured output via `messages.parse(output_format=<Pydantic model>)`.
  Default model Claude Opus 5.5. Thinking is always on and sampling parameters are not
  accepted, so there is no temperature setting; depth is set with `effort`.
  Auth: AWS_BEARER_TOKEN_BEDROCK (Bedrock API key) or the AWS credential chain.
- bedrock-openai: non-Claude Bedrock models (gpt-oss, Kimi, GLM, Mistral) via Bedrock's
  OpenAI-compatible endpoint with the Bedrock API key; LLM_MODEL is required.
- groq, ollama: OpenAI-compatible chat API. Groq accepts several keys (GROQ_API_KEY_1, _2, ...);
  a KeyRouter sends each call to the key with quota left and fails over on 429.
  Structured output by forcing one tool whose parameters are the Pydantic JSON schema. Temperature 0.

At most 3 failed provider calls per request (rate-limited calls on a multi-key Groq
router wait instead); a reply that fails schema validation is re-asked
once (OpenAI-compatible path). Prompts and replies are never logged at INFO.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
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


_DURATION = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")
_TRY_AGAIN = re.compile(r"try again in ((?:\d+(?:\.\d+)?(?:h|ms|m|s))+)", re.IGNORECASE)
_DAILY = re.compile(r"per day|\(TPD\)|\(RPD\)", re.IGNORECASE)


def parse_duration(text: str | None) -> float | None:
    """Groq durations: '7.66s', '2m59.56s', '1h2m3s', '120ms'; also plain seconds ('20')."""
    if not text:
        return None
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    m = _DURATION.match(text)
    if not m or not any(m.groups()):
        return None
    h, mi, s, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def groq_keys_from_env(environ: Any = None) -> list[str]:
    """GROQ_API_KEY_1, GROQ_API_KEY_2, ... in numeric order, then GROQ_API_KEY; blanks and duplicates dropped."""
    env = os.environ if environ is None else environ
    numbered = sorted(
        (int(k.rsplit("_", 1)[1]), v.strip())
        for k, v in env.items()
        if re.fullmatch(r"GROQ_API_KEY_\d+", k) and v and v.strip()
    )
    keys = [v for _, v in numbered] + [env.get("GROQ_API_KEY", "").strip()]
    return list(dict.fromkeys(k for k in keys if k))


@dataclass
class _KeyState:
    cooldown_until: float = 0.0  # per-minute limits (TPM/RPM)
    exhausted_until: float = 0.0  # per-day limits (TPD/RPD)
    remaining_tokens: float = float("inf")  # unknown until the first response: try unused keys first
    last_used: float = 0.0


class KeyRouter:
    """Spreads calls over several Groq keys by remaining quota; fails over on 429 without waiting.

    Keys are never logged: they appear as key#1, key#2, ... (order of GROQ_API_KEY_<n>).
    """

    def __init__(self, keys: list[str], clock: Any = time.monotonic, sleep: Any = time.sleep) -> None:
        if not keys:
            raise LlmError("no Groq API keys configured")
        self.keys = keys
        self.state = [_KeyState() for _ in keys]
        self.clock = clock
        self.sleep = sleep
        self._lock = threading.Lock()

    def label(self, i: int) -> str:
        return f"key#{i + 1}"

    def pick(self) -> int:
        """Index of the best available key; waits for the earliest key if all are cooling down."""
        while True:
            with self._lock:
                now = self.clock()
                free = [i for i, s in enumerate(self.state) if s.cooldown_until <= now and s.exhausted_until <= now]
                if free:
                    best = max(free, key=lambda i: (self.state[i].remaining_tokens, -self.state[i].last_used))
                    self.state[best].last_used = now
                    return best
                if all(s.exhausted_until > now for s in self.state):
                    wait = min(s.exhausted_until for s in self.state) - now
                    raise LlmError(f"all Groq keys hit their daily limit; next frees in {wait / 60:.0f} min")
                wait = min(max(s.cooldown_until, s.exhausted_until) for s in self.state) - now
            self.sleep(max(wait, 0.5))

    def observe(self, i: int, resp: httpx.Response) -> None:
        with self._lock:
            s = self.state[i]
            remaining = resp.headers.get("x-ratelimit-remaining-tokens")
            if remaining is not None and remaining.replace(".", "", 1).isdigit():
                s.remaining_tokens = float(remaining)
            if resp.status_code != 429:
                return
            now = self.clock()
            body = resp.text
            retry = parse_duration(resp.headers.get("retry-after"))
            hint = _TRY_AGAIN.search(body)
            wait = parse_duration(hint.group(1)) if hint else None
            if _DAILY.search(body):
                reset = wait or parse_duration(resp.headers.get("x-ratelimit-reset-requests")) or 3600.0
                s.exhausted_until = now + reset
                reason = "daily_limit"
            else:
                reset = retry or wait or parse_duration(resp.headers.get("x-ratelimit-reset-tokens")) or 20.0
                s.cooldown_until = now + reset
                reason = "minute_limit"
            s.remaining_tokens = 0.0
        log.info("groq_key_limited key=%s reason=%s reset_s=%.0f", self.label(i), reason, reset)


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
    router: KeyRouter | None = None  # several Groq keys: picks one per attempt, fails over on 429
    rate_limit_wait_s: float = 900.0  # with a router: longest wait for any key to free up, per call

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
        """One logical call: up to 3 failed provider attempts with backoff.

        With a key router, rate-limited (429) answers do not use up attempts: the router moves to a key with
        quota left, or waits for the earliest one, for at most `rate_limit_wait_s`.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        client = self.http or httpx.Client(timeout=self.timeout)
        last: Exception | None = None
        estimate = len(json.dumps(body)) // 3 + int(body.get("max_tokens", 0))
        failures = 0
        deadline = time.monotonic() + self.rate_limit_wait_s
        try:
            while failures < 3:
                key_index = None
                if self.router:
                    key_index = self.router.pick()
                    headers["Authorization"] = f"Bearer {self.router.keys[key_index]}"
                if self.pacer:
                    self.pacer.acquire(estimate)
                try:
                    resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
                except httpx.TransportError as exc:
                    last = exc
                    failures += 1
                    self.sleep(2.0 * failures)
                    continue
                if resp.status_code == 413 or (resp.status_code == 429 and "request too large" in resp.text.lower()):
                    # A per-request cap of this tier (e.g. Groq free OTPM < max_tokens): retrying cannot help.
                    raise LlmError(
                        "request too large for this model's per-minute limit on this tier: "
                        "lower LLM_MAX_TOKENS or the prompt, or use another model or tier"
                    )
                if self.router and key_index is not None:
                    self.router.observe(key_index, resp)
                    if resp.status_code == 429:
                        last = LlmError("HTTP 429")
                        if time.monotonic() > deadline:
                            raise LlmError(f"rate limited on every key for over {self.rate_limit_wait_s:.0f}s")
                        continue  # the router skips this key now; no sleep while another key is free
                self.usage.calls += 1
                if resp.is_success:
                    payload: dict[str, Any] = resp.json()
                    usage = payload.get("usage") or {}
                    self.usage.tokens_in += int(usage.get("prompt_tokens", 0))
                    self.usage.tokens_out += int(usage.get("completion_tokens", 0))
                    return payload
                failures += 1
                if resp.status_code in (429, 500, 502, 503, 504):
                    last = LlmError(f"HTTP {resp.status_code}")
                    retry_after = resp.headers.get("retry-after", "")
                    wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 0.0
                    # Groq free tier (~tokens/min per model) clears within about a minute.
                    self.sleep(max(wait, 20.0 * failures if resp.status_code == 429 else 2.0 * failures))
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
        budget = self.max_tokens
        for reask in range(2):
            payload = self._post(self._body(messages, name, json_schema, budget))
            out_tokens = int((payload.get("usage") or {}).get("completion_tokens", 0))
            truncated = out_tokens >= budget
            raw = self._tool_args(payload, name)
            try:
                result = schema.model_validate_json(_decode_string_fields(raw))
                break
            except ValidationError as exc:
                fields = "; ".join(f"{'.'.join(map(str, e['loc']))} ({e['type']})" for e in exc.errors()[:3])
                error = f"ValidationError at {fields}"
                if truncated:
                    error += f"; reply cut off at max_tokens {budget}"
                if truncated:
                    # Reasoning models spend part of max_tokens thinking: retry with a bigger budget, not a re-ask.
                    budget = min(budget * 2, 32768)
                elif reask == 0:
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


def _decode_string_fields(raw: str) -> str:
    """Some models send a list or object field as a JSON string ("criteria": "[{...}]"): decode those."""
    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(data, dict):
        return raw
    changed = False
    for k, v in data.items():
        if isinstance(v, str) and v.lstrip()[:1] in ("[", "{"):
            try:
                data[k] = json.loads(v)
                changed = True
            except ValueError:
                pass
    return json.dumps(data) if changed else raw


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
            # Reasoning models (e.g. Kimi) count thinking tokens in max_tokens; billing is per token used.
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "16000")),
            cache=cache,
        )
    if provider not in BASE_URLS:
        raise LlmError(f"unknown LLM_PROVIDER {provider!r}")
    keys = groq_keys_from_env() if provider == "groq" else []
    if provider == "groq" and not keys:
        raise LlmError("GROQ_API_KEY_1 (or GROQ_API_KEY) is not set")
    api_key = keys[0] if keys else ""
    router = KeyRouter(keys) if len(keys) > 1 else None
    base = os.environ.get("OLLAMA_BASE_URL", "").rstrip("/") + "/v1" if provider == "ollama" else BASE_URLS["groq"]
    return ChatLLM(
        provider=provider,
        model=os.environ.get("LLM_MODEL") or DEFAULT_MODELS[provider],
        base_url=base,
        api_key=api_key,
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2048")),
        cache=cache,
        pacer=TokenPacer(int(os.environ.get("LLM_TOKENS_PER_MIN", "0"))),
        router=router,
    )
