"""LLM client: forced tool call, temperature 0, schema re-ask, bounded retries, cache."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from agents.llm import ChatLLM, KeyRouter, LlmError, get_llm, groq_keys_from_env, parse_duration


class Answer(BaseModel):
    label: str
    score: int


def reply(args: dict[str, Any] | str, name: str = "Answer") -> httpx.Response:
    arguments = args if isinstance(args, str) else json.dumps(args)
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def make_llm(responses: list[httpx.Response], cache: Any = None) -> tuple[ChatLLM, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    queue = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return next(queue)

    llm = ChatLLM(
        provider="groq",
        model="m",
        base_url="https://llm.test",
        api_key="k",
        cache=cache,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda s: None,
    )
    return llm, sent


def complete(llm: ChatLLM) -> Answer:
    return llm.complete(task="t", system="sys", user="u", schema=Answer)


def test_forces_tool_and_temperature_zero() -> None:
    llm, sent = make_llm([reply({"label": "a", "score": 1})])
    assert complete(llm) == Answer(label="a", score=1)
    body = sent[0]
    assert body["temperature"] == 0
    assert body["tool_choice"] == {"type": "function", "function": {"name": "Answer"}}
    assert body["tools"][0]["function"]["parameters"]["required"] == ["label", "score"]
    assert (llm.usage.tokens_in, llm.usage.tokens_out) == (10, 5)


def test_schema_error_reasked_once_with_errors() -> None:
    llm, sent = make_llm([reply({"label": "a"}), reply({"label": "a", "score": 2})])
    assert complete(llm).score == 2
    followup = sent[1]["messages"][-1]["content"]
    assert "score" in followup
    assert "did not match the schema" in followup


def test_schema_error_twice_raises() -> None:
    llm, _ = make_llm([reply({"label": "a"}), reply("not json")])
    with pytest.raises(LlmError, match="schema validation twice"):
        complete(llm)


def test_429_retried_then_ok() -> None:
    llm, sent = make_llm([httpx.Response(429), reply({"label": "a", "score": 1})])
    assert complete(llm).label == "a"
    assert len(sent) == 2


def test_at_most_three_provider_calls() -> None:
    llm, sent = make_llm([httpx.Response(503)] * 3)
    with pytest.raises(LlmError, match="failed after 3 attempts"):
        complete(llm)
    assert len(sent) == 3


def test_tool_use_failed_retries_with_bigger_budget() -> None:
    llm, sent = make_llm(
        [httpx.Response(400, text='{"error":{"code":"tool_use_failed"}}'), reply({"label": "a", "score": 1})]
    )
    complete(llm)
    assert sent[1]["max_tokens"] == int(sent[0]["max_tokens"] * 1.5)


def test_client_error_not_retried() -> None:
    llm, sent = make_llm([httpx.Response(401, text="bad key")])
    with pytest.raises(LlmError, match="rejected: HTTP 401"):
        complete(llm)
    assert len(sent) == 1


def test_missing_tool_call_raises() -> None:
    llm, _ = make_llm([httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})])
    with pytest.raises(LlmError, match="no tool call"):
        complete(llm)


def test_cache_hit_skips_provider() -> None:
    class Mem:
        def __init__(self) -> None:
            self.d: dict[str, dict[str, Any]] = {}

        def get(self, key: str) -> dict[str, Any] | None:
            return self.d.get(key)

        def set(self, key: str, value: dict[str, Any]) -> None:
            self.d[key] = value

    cache = Mem()
    llm, sent = make_llm([reply({"label": "a", "score": 1})], cache=cache)
    assert complete(llm) == complete(llm)
    assert len(sent) == 1
    assert llm.usage.cache_hits == 1


def _clear_groq_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for k in [k for k in os.environ if k.startswith("GROQ_API_KEY")]:
        monkeypatch.delenv(k)


def test_get_llm_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_groq_keys(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(LlmError, match="GROQ_API_KEY"):
        get_llm()


def test_get_llm_bedrock_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    for var in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(LlmError, match="credentials"):
        get_llm()


def test_get_llm_bedrock_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.llm import BedrockLLM

    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_EFFORT", raising=False)
    llm = get_llm()
    assert isinstance(llm, BedrockLLM)
    assert (llm.model, llm.effort) == ("anthropic.claude-opus-5-5", "medium")


def test_get_llm_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    llm = get_llm()
    assert isinstance(llm, ChatLLM)
    assert (llm.base_url, llm.model, llm.api_key) == ("http://localhost:11434/v1", "qwen2.5:14b", "")


_: Callable[..., Any] = make_llm


def test_token_pacer_waits_for_window() -> None:
    from agents.llm import TokenPacer

    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    pacer = TokenPacer(1000, clock=lambda: now[0], sleep=sleep)
    pacer.acquire(600)
    pacer.acquire(300)
    assert slept == []
    pacer.acquire(500)  # 1400 > 1000 within 60 s: must wait for the first entry to expire
    assert slept
    assert now[0] >= 60
    pacer.acquire(5000)  # larger than the budget: capped, never waits forever


class FakeParsed:
    def __init__(self, parsed: Any, stop_reason: str = "end_turn") -> None:
        self.parsed_output = parsed
        self.stop_reason = stop_reason
        self.usage = type("U", (), {"input_tokens": 100, "output_tokens": 20})()


class FakeBedrock:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Messages:
            def parse(self, **kw: Any) -> Any:
                outer.calls.append(kw)
                r = outer.replies.pop(0)
                if isinstance(r, Exception):
                    raise r
                return r

        self.beta = type("B", (), {"messages": _Messages()})()


def test_bedrock_structured_output_and_effort() -> None:
    from agents.llm import BedrockLLM

    fake = FakeBedrock([FakeParsed(Answer(label="a", score=1))])
    llm = BedrockLLM(model="anthropic.claude-opus-5-5", client=fake, effort="medium")
    assert llm.complete(task="t", system="sys", user="u", schema=Answer) == Answer(label="a", score=1)
    call = fake.calls[0]
    assert call["output_format"] is Answer
    assert call["output_config"] == {"effort": "medium"}
    assert call["system"] == "sys"
    assert "temperature" not in call  # not accepted by current Claude models
    assert (llm.usage.calls, llm.usage.tokens_in, llm.usage.tokens_out) == (1, 100, 20)


def test_bedrock_refusal_raises() -> None:
    from agents.llm import BedrockLLM

    llm = BedrockLLM(model="m", client=FakeBedrock([FakeParsed(None, stop_reason="refusal")]))
    with pytest.raises(LlmError, match="refusal"):
        llm.complete(task="t", system="s", user="u", schema=Answer)


def test_bedrock_http_error_wrapped() -> None:
    import anthropic
    import httpx2

    from agents.llm import BedrockLLM

    err = anthropic.APIStatusError(
        "denied", response=httpx2.Response(403, request=httpx2.Request("POST", "https://x")), body=None
    )
    llm = BedrockLLM(model="m", client=FakeBedrock([err]))
    with pytest.raises(LlmError, match="HTTP 403"):
        llm.complete(task="t", system="s", user="u", schema=Answer)


def test_bedrock_cache_hit_skips_call() -> None:
    from agents.llm import BedrockLLM

    class Mem:
        def __init__(self) -> None:
            self.d: dict[str, dict[str, Any]] = {}

        def get(self, key: str) -> dict[str, Any] | None:
            return self.d.get(key)

        def set(self, key: str, value: dict[str, Any]) -> None:
            self.d[key] = value

    fake = FakeBedrock([FakeParsed(Answer(label="a", score=1))])
    llm = BedrockLLM(model="m", client=fake, cache=Mem())
    llm.complete(task="t", system="s", user="u", schema=Answer)
    llm.complete(task="t", system="s", user="u", schema=Answer)
    assert len(fake.calls) == 1


def test_get_llm_bedrock_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "bedrock-openai")
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "tok")
    monkeypatch.setenv("LLM_MODEL", "zai.glm-5")
    llm = get_llm()
    assert isinstance(llm, ChatLLM)
    assert llm.base_url == "https://bedrock-runtime.ap-south-1.amazonaws.com/openai/v1"
    assert (llm.model, llm.api_key) == ("zai.glm-5", "tok")
    monkeypatch.setenv("LLM_MODEL", "")
    with pytest.raises(LlmError, match="LLM_MODEL"):
        get_llm()


# ---------------- Groq key router ----------------


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def limited(daily: bool = False, retry: str = "7.5") -> httpx.Response:
    what = "tokens per day (TPD): Limit 500000, Used 499990" if daily else "tokens per minute (TPM): Limit 6000"
    body = {"error": {"message": f"Rate limit reached on {what}. Please try again in 2m30s.", "code": "rate_limit"}}
    return httpx.Response(429, headers={"retry-after": retry}, json=body)


def router_llm(responses: list[httpx.Response], n_keys: int = 3) -> tuple[ChatLLM, list[str], Clock]:
    used: list[str] = []
    queue = iter(responses)
    clock = Clock()

    def handler(request: httpx.Request) -> httpx.Response:
        used.append(request.headers["authorization"].removeprefix("Bearer "))
        return next(queue)

    router = KeyRouter([f"secret-{i}" for i in range(1, n_keys + 1)], clock=clock, sleep=clock.sleep)
    llm = ChatLLM(
        provider="groq",
        model="m",
        base_url="https://llm.test",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=clock.sleep,
        router=router,
    )
    return llm, used, clock


def ok(remaining: int) -> httpx.Response:
    r = reply({"label": "a", "score": 1})
    r.headers["x-ratelimit-remaining-tokens"] = str(remaining)
    return r


def test_parse_duration() -> None:
    assert parse_duration("7.66s") == pytest.approx(7.66)
    assert parse_duration("2m59.56s") == pytest.approx(179.56)
    assert parse_duration("1h2m3s") == pytest.approx(3723)
    assert parse_duration("120ms") == pytest.approx(0.12)
    assert parse_duration("20") == 20.0
    assert parse_duration("") is None
    assert parse_duration("soon") is None


def test_keys_from_env_numbered_order_then_plain() -> None:
    env = {"GROQ_API_KEY_10": "c", "GROQ_API_KEY_2": "b", "GROQ_API_KEY_1": " a ", "GROQ_API_KEY": "a", "X": "y"}
    assert groq_keys_from_env(env) == ["a", "b", "c"]
    assert groq_keys_from_env({"GROQ_API_KEY_1": ""}) == []


def test_picks_key_with_most_remaining_tokens() -> None:
    llm, used, _ = router_llm([ok(100), ok(5000), ok(9000), ok(9000)])
    for _ in range(4):
        complete(llm)
    # First round tries each unused key once; then the one with the most tokens left (key 3, then 2 after 3 drops).
    assert used[:3] == ["secret-1", "secret-2", "secret-3"]
    assert used[3] == "secret-3"


def test_minute_429_fails_over_without_sleeping() -> None:
    llm, used, clock = router_llm([limited(), reply({"label": "b", "score": 2})])
    assert complete(llm).label == "b"
    assert used == ["secret-1", "secret-2"]
    assert clock.slept == []
    assert llm.router is not None
    assert llm.router.state[0].cooldown_until == pytest.approx(clock.t + 7.5)


def test_daily_429_skips_key_and_raises_when_all_exhausted() -> None:
    llm, used, _ = router_llm([limited(daily=True), limited(daily=True)], n_keys=2)
    with pytest.raises(LlmError, match="daily limit"):
        complete(llm)
    assert used == ["secret-1", "secret-2"]
    assert llm.router is not None
    assert all(s.exhausted_until > 0 for s in llm.router.state)


def test_waits_for_earliest_key_when_all_cooling() -> None:
    llm, used, clock = router_llm([limited(retry="10"), limited(retry="30"), reply({"label": "c", "score": 3})], 2)
    assert complete(llm).label == "c"
    assert used == ["secret-1", "secret-2", "secret-1"]
    assert sum(clock.slept) == pytest.approx(10)


def test_router_logs_never_contain_keys(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.INFO)
    llm, _, _ = router_llm([limited(), limited(daily=True), reply({"label": "d", "score": 4})])
    complete(llm)
    assert "key#1" in caplog.text
    assert "secret" not in caplog.text


def test_get_llm_builds_router_for_several_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_groq_keys(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY_1", "k1")
    llm = get_llm()
    assert isinstance(llm, ChatLLM)
    assert llm.router is None
    assert llm.api_key == "k1"
    monkeypatch.setenv("GROQ_API_KEY_2", "k2")
    llm = get_llm()
    assert isinstance(llm, ChatLLM)
    assert llm.router is not None
    assert llm.router.keys == ["k1", "k2"]


def test_request_too_large_fails_at_once() -> None:
    body = {"error": {"message": "Request too large for model on output tokens per minute (OTPM): Limit 1000"}}
    llm, used, _ = router_llm([httpx.Response(429, json=body)])
    with pytest.raises(LlmError, match="too large"):
        complete(llm)
    assert len(used) == 1


def test_many_minute_429s_do_not_use_up_attempts() -> None:
    llm, used, _ = router_llm([limited(retry="5")] * 8 + [reply({"label": "e", "score": 5})], n_keys=2)
    assert complete(llm).label == "e"
    assert len(used) == 9
