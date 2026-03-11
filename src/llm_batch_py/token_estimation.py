from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import cache
from typing import Any

import orjson
import tiktoken

from llm_batch_py.jobs import EmbeddingJob, StructuredOutputJob

_OPENAI_CHAT_OVERHEAD: dict[str, tuple[int, int]] = {
    "gpt-4o": (3, 1),
    "gpt-4o-mini": (3, 1),
}


@cache
def _encoding_for_model(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def estimate_openai_tokens(model: str, body: Mapping[str, Any]) -> int:
    encoding = _encoding_for_model(model)
    if "input" in body and isinstance(body["input"], str):
        return len(encoding.encode(body["input"]))

    messages = body.get("messages", [])
    tokens_per_message, tokens_per_name = _OPENAI_CHAT_OVERHEAD.get(model, (3, 1))
    total = 0
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            if key == "content":
                if isinstance(value, str):
                    total += len(encoding.encode(value))
                else:
                    total += len(encoding.encode(str(value)))
            elif key == "name" and value:
                total += tokens_per_name + len(encoding.encode(str(value)))
            else:
                total += len(encoding.encode(str(value)))
    response_format = body.get("response_format")
    if response_format is not None:
        total += len(encoding.encode(str(response_format)))
    return total + 3


def estimate_openai_batch_tokens(model: str, bodies: Iterable[Mapping[str, Any]]) -> list[int]:
    return [estimate_openai_tokens(model, body) for body in bodies]


def estimate_anthropic_tokens(model: str, body: Mapping[str, Any]) -> int:
    encoding = _encoding_for_model(model)
    total = 0

    system = body.get("system")
    if system is not None:
        total += len(encoding.encode(_stringify_token_payload(system))) + 3

    for message in body.get("messages", []):
        total += 4
        total += len(encoding.encode(_stringify_token_payload(message.get("role", ""))))
        total += len(encoding.encode(_stringify_token_payload(message.get("content", ""))))

    tools = body.get("tools")
    if tools is not None:
        total += len(encoding.encode(_stringify_token_payload(tools))) + 8

    tool_choice = body.get("tool_choice")
    if tool_choice is not None:
        total += len(encoding.encode(_stringify_token_payload(tool_choice))) + 4

    return total + 3


def estimate_anthropic_batch_tokens(
    client_or_model: Any,
    config_or_bodies: Any,
    bodies: Iterable[Mapping[str, Any]] | None = None,
) -> list[int]:
    if bodies is None:
        model = client_or_model
        return [estimate_anthropic_tokens(model, body) for body in config_or_bodies]

    client = client_or_model
    config = config_or_bodies
    estimates: list[int] = []
    for body in bodies:
        response = client.messages.count_tokens(
            model=config.model,
            messages=body["messages"],
            system=body.get("system"),
        )
        estimates.append(int(response.input_tokens))
    return estimates


def estimate_job_output_tokens(
    job: StructuredOutputJob[Any] | EmbeddingJob, avg_output_tokens: int
) -> int:
    if job.endpoint_kind == "embeddings":
        return 0
    return avg_output_tokens


def _stringify_token_payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    return orjson.dumps(value).decode("utf-8")
