from __future__ import annotations

from dataclasses import dataclass

import llm_batch_py.token_estimation as token_estimation
from llm_batch_py.jobs import AnthropicConfig
from llm_batch_py.token_estimation import estimate_anthropic_batch_tokens, estimate_openai_tokens


def test_estimate_openai_tokens_for_chat_payload() -> None:
    estimate = estimate_openai_tokens(
        "gpt-4o-mini",
        {
            "messages": [
                {"role": "system", "content": "Return JSON only"},
                {"role": "user", "content": "Label alpha"},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "labels", "schema": {"type": "object"}},
            },
        },
    )

    assert estimate > 0


def test_estimate_anthropic_batch_tokens_uses_sdk_client() -> None:
    calls = []

    @dataclass
    class Response:
        input_tokens: int

    class Messages:
        def count_tokens(self, **kwargs):
            calls.append(kwargs)
            return Response(input_tokens=42)

    class Client:
        messages = Messages()

    estimates = estimate_anthropic_batch_tokens(
        Client(),
        AnthropicConfig(model="claude-3-5-haiku-latest", api_key="test"),
        [{"messages": [{"role": "user", "content": "Hello"}], "system": "Return JSON"}],
    )

    assert estimates == [42]
    assert calls[0]["model"] == "claude-3-5-haiku-latest"


def test_encoding_lookup_is_cached_per_model(monkeypatch) -> None:
    token_estimation._encoding_for_model.cache_clear()
    calls = []

    class Encoding:
        def encode(self, value: str) -> list[int]:
            return list(range(len(value)))

    def fake_encoding_for_model(model: str):
        calls.append(model)
        return Encoding()

    monkeypatch.setattr(token_estimation.tiktoken, "encoding_for_model", fake_encoding_for_model)
    monkeypatch.setattr(
        token_estimation.tiktoken,
        "get_encoding",
        lambda name: (_ for _ in ()).throw(AssertionError(f"unexpected fallback: {name}")),
    )

    first = estimate_openai_tokens("gpt-4o-mini", {"messages": [{"role": "user", "content": "a"}]})
    second = estimate_openai_tokens(
        "gpt-4o-mini",
        {"messages": [{"role": "user", "content": "bb"}]},
    )

    assert first > 0
    assert second > 0
    assert calls == ["gpt-4o-mini"]
