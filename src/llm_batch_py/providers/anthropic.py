from __future__ import annotations

import re
from typing import Any

import anthropic
import orjson

from llm_batch_py._core_wrapper import canonical_json
from llm_batch_py.jobs import AnthropicConfig, StructuredOutputJob, output_model_json_schema
from llm_batch_py.prompt_cache import (
    emit_prompt_cache_diagnostic_once,
    prompt_cache_enabled,
)
from llm_batch_py.providers.base import (
    BatchSnapshot,
    PreparedRequest,
    ProviderAdapter,
    ProviderResult,
    RowRequest,
    SubmittedBatch,
    parse_json_response_text,
)
from llm_batch_py.token_estimation import estimate_anthropic_batch_tokens


class AnthropicStructuredAdapter(ProviderAdapter):
    provider_name = "anthropic"
    request_cap = 100_000
    byte_cap = 256 * 1024 * 1024

    def __init__(self, config: AnthropicConfig) -> None:
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key, timeout=config.timeout)

    def prepare_requests(
        self,
        job: StructuredOutputJob[Any],
        requests: list[RowRequest],
        *,
        input_token_estimation: str = "exact",
    ) -> list[PreparedRequest]:
        transport_records = [self.render_transport_record(job, request) for request in requests]
        estimated_tokens: list[int | None]
        if input_token_estimation == "exact":
            estimated_tokens = estimate_anthropic_batch_tokens(
                self.config.model,
                (record["params"] for record in transport_records),
            )
        else:
            estimated_tokens = [None] * len(transport_records)
        return [
            PreparedRequest(
                request=request,
                transport_record=transport_record,
                estimated_input_tokens=estimated_tokens[index],
            )
            for index, (request, transport_record) in enumerate(
                zip(requests, transport_records, strict=True)
            )
        ]

    def render_transport_record(
        self,
        job: StructuredOutputJob[Any],
        request: RowRequest,
    ) -> dict[str, Any]:
        return {
            "custom_id": request.custom_id,
            "params": self._request_params(job, request.payload),
        }

    def submit_batch(
        self,
        job: StructuredOutputJob[Any],
        request_artifact_path: str,
        request_payload: bytes,
        transport_records: list[dict[str, Any]],
    ) -> SubmittedBatch:
        response = self.client.messages.batches.create(requests=transport_records)
        return SubmittedBatch(
            provider_batch_id=response.id,
            status=response.processing_status,
            raw_payload=response.model_dump(mode="json"),
        )

    def poll_batch(
        self,
        job: StructuredOutputJob[Any],
        provider_batch_id: str,
    ) -> BatchSnapshot:
        response = self.client.messages.batches.retrieve(provider_batch_id)
        return BatchSnapshot(
            status=response.processing_status,
            raw_payload=response.model_dump(mode="json"),
        )

    def fetch_results(
        self,
        job: StructuredOutputJob[Any],
        batch_snapshot: BatchSnapshot,
        provider_batch_id: str,
    ) -> list[ProviderResult]:
        results: list[ProviderResult] = []
        for item in self.client.messages.batches.results(provider_batch_id):
            raw = item.model_dump(mode="json")
            result = raw["result"]
            if result["type"] == "succeeded":
                try:
                    parsed = _anthropic_message_payload(result["message"])
                except (KeyError, TypeError, orjson.JSONDecodeError):
                    results.append(
                        ProviderResult(
                            custom_id=item.custom_id,
                            status="failed",
                            parsed_output=None,
                            raw_payload=raw,
                            error_code="invalid_json_response",
                        )
                    )
                    continue
                results.append(
                    ProviderResult(
                        custom_id=item.custom_id,
                        status="completed",
                        parsed_output=parsed,
                        raw_payload=raw,
                        raw_output_text=_anthropic_message_raw_output_text(result["message"]),
                        input_tokens=result["message"]["usage"].get("input_tokens"),
                        output_tokens=result["message"]["usage"].get("output_tokens"),
                    )
                )
                continue
            error_code = _anthropic_error_code(result)
            results.append(
                ProviderResult(
                    custom_id=item.custom_id,
                    status="failed",
                    parsed_output=None,
                    raw_payload=raw,
                    error_code=error_code,
                )
            )
        return results

    def _request_params(
        self,
        job: StructuredOutputJob[Any],
        payload: dict[str, Any] | str,
    ) -> dict[str, Any]:
        normalized = _normalize_structured_payload(payload)
        prompt_cache = job.prompt_cache
        output_tool = _output_tool_schema(job)
        tools = [*(normalized.get("tools") or []), output_tool]
        system = normalized.get("system")
        messages = normalized["messages"]
        cache_request_fields: dict[str, Any] = {}
        if prompt_cache_enabled(prompt_cache):
            system, messages, tools = _apply_prompt_cache_breakpoint(
                system=system,
                messages=messages,
                tools=tools,
            )
            cache_request_fields = {"cache_control": {"type": "ephemeral"}}
        params: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": normalized.get("max_tokens", self.config.max_output_tokens),
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "tool", "name": output_tool["name"]},
        }
        if system is not None:
            params["system"] = system
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if prompt_cache_enabled(prompt_cache):
            emit_prompt_cache_diagnostic_once(
                provider_name="anthropic",
                model=self.config.model,
                config=prompt_cache,
                payload=_cache_diagnostic_payload(
                    {"system": system, "messages": messages, "tools": tools}
                ),
                boundary="explicit",
                note=(
                    "llm-batch-py inserts one Anthropic cache breakpoint automatically on the "
                    "estimated reusable prompt prefix."
                ),
                request_fields=cache_request_fields,
            )
        return params


def _normalize_structured_payload(payload: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(payload, str):
        return {"messages": [{"role": "user", "content": payload}]}
    if "messages" in payload:
        return dict(payload)
    raise ValueError("Structured prompt builders must return a string or a dict with messages")


def _anthropic_error_code(result: dict[str, Any]) -> str:
    error_obj = result.get("error")
    if isinstance(error_obj, dict):
        nested_error = error_obj.get("error")
        if isinstance(nested_error, dict):
            nested_type = nested_error.get("type")
            if isinstance(nested_type, str) and nested_type:
                return nested_type
        error_type = error_obj.get("type")
        if isinstance(error_type, str) and error_type:
            return error_type
    result_type = result.get("type")
    if isinstance(result_type, str) and result_type:
        return result_type
    return "unknown_error"


def _anthropic_message_text(message: dict[str, Any]) -> str:
    chunks = []
    for block in message.get("content", []):
        if block.get("type") == "text":
            chunks.append(block.get("text", ""))
    return "".join(chunks)


def _anthropic_message_payload(message: dict[str, Any]) -> dict[str, Any]:
    for block in message.get("content", []):
        if block.get("type") == "tool_use":
            tool_input = block.get("input")
            if isinstance(tool_input, dict):
                return tool_input
    text = _anthropic_message_text(message)
    return parse_json_response_text(text)


def _anthropic_message_raw_output_text(message: dict[str, Any]) -> str | None:
    for block in message.get("content", []):
        if block.get("type") == "tool_use":
            tool_input = block.get("input")
            if isinstance(tool_input, dict):
                return canonical_json(tool_input)
    text = _anthropic_message_text(message)
    return text or None


def _output_tool_schema(job: StructuredOutputJob[Any]) -> dict[str, Any]:
    schema = output_model_json_schema(job.output_model)
    tool_name = _tool_name(job.name)
    return {
        "name": tool_name,
        "description": f"Return structured output for {job.name}.",
        "input_schema": schema,
    }


def _tool_name(job_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", job_name).strip("_") or "structured_output"
    return f"{normalized[:57]}_output"


def _cache_diagnostic_payload(normalized: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"messages": normalized["messages"]}
    if normalized.get("system") is not None:
        payload["system"] = normalized["system"]
    if normalized.get("tools") is not None:
        payload["tools"] = normalized["tools"]
    return payload


def _apply_prompt_cache_breakpoint(
    *,
    system: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    tools_copy = [dict(tool) for tool in tools]
    system_copy = _normalize_system_blocks(system)
    messages_copy = [_normalize_message_for_cache(message) for message in messages]

    candidates: list[tuple[str, int, int | None]] = []
    for tool_index in range(len(tools_copy)):
        candidates.append(("tool", tool_index, None))
    for block_index in range(len(system_copy)):
        candidates.append(("system", block_index, None))
    for message_index, message in enumerate(messages_copy):
        content = message.get("content")
        if isinstance(content, list):
            for block_index in range(len(content)):
                candidates.append(("message", message_index, block_index))

    if not candidates:
        return system, messages_copy, tools_copy

    breakpoint_target = candidates[-2] if len(candidates) > 1 else candidates[-1]
    kind, outer_index, inner_index = breakpoint_target
    if kind == "tool":
        tools_copy[outer_index]["cache_control"] = {"type": "ephemeral"}
    elif kind == "system":
        system_copy[outer_index]["cache_control"] = {"type": "ephemeral"}
    else:
        message_content = messages_copy[outer_index]["content"]
        assert isinstance(message_content, list)
        message_content[inner_index]["cache_control"] = {"type": "ephemeral"}

    normalized_system: Any = system_copy if system_copy else None
    return normalized_system, messages_copy, tools_copy


def _normalize_system_blocks(system: Any) -> list[dict[str, Any]]:
    if system is None:
        return []
    if isinstance(system, str):
        return [{"type": "text", "text": system}]
    if isinstance(system, list):
        return [dict(block) for block in system]
    raise ValueError("Anthropic structured system prompts must be a string or a list of blocks")


def _normalize_message_for_cache(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    content = normalized.get("content")
    if isinstance(content, str):
        normalized["content"] = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        normalized["content"] = [
            dict(block) if isinstance(block, dict) else block for block in content
        ]
    return normalized
