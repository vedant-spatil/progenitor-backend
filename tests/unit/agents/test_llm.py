"""LLM client: forced tool call, temperature 0, schema re-ask, bounded retries, cache."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from agents.llm import ChatLLM, LlmError, get_llm


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


def test_get_llm_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
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
