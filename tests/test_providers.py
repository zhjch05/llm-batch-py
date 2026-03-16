from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import orjson
import polars as pl
from pydantic import BaseModel

from llm_batch_py import (
    AnthropicConfig,
    OpenAIConfig,
    PromptCacheConfig,
    ResultCacheStoreConfig,
    StructuredOutputJob,
    structured_template,
)
from llm_batch_py.prompt_cache import _reset_prompt_cache_diagnostics
from llm_batch_py.providers.anthropic import AnthropicStructuredAdapter
from llm_batch_py.providers.base import BatchSnapshot
from llm_batch_py.providers.openai import OpenAIBatchAdapter
from llm_batch_py.runner import Runner


@dataclass
class FakeAnthropicItem:
    custom_id: str
    payload: dict[str, object]

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        assert mode == "json"
        return self.payload


class FakeAnthropicBatches:
    def __init__(self, items: list[FakeAnthropicItem]) -> None:
        self._items = items

    def results(self, provider_batch_id: str) -> list[FakeAnthropicItem]:
        assert provider_batch_id == "provider-1"
        return list(self._items)


class FakeAnthropicMessages:
    def __init__(self, items: list[FakeAnthropicItem]) -> None:
        self.batches = FakeAnthropicBatches(items)


class FakeAnthropicClient:
    def __init__(self, items: list[FakeAnthropicItem]) -> None:
        self.messages = FakeAnthropicMessages(items)


def test_anthropic_fetch_results_extracts_nested_error_code(structured_job) -> None:
    adapter = AnthropicStructuredAdapter.__new__(AnthropicStructuredAdapter)
    adapter.config = structured_job.provider
    adapter.client = FakeAnthropicClient(
        [
            FakeAnthropicItem(
                custom_id="too-long",
                payload={
                    "custom_id": "too-long",
                    "result": {
                        "type": "errored",
                        "error": {
                            "type": "error",
                            "error": {
                                "details": {"error_visibility": "user_facing"},
                                "type": "invalid_request_error",
                                "message": "prompt is too long: 201148 tokens > 200000 maximum",
                            },
                            "request_id": "workerreq_123",
                        },
                    },
                },
            )
        ]
    )

    results = adapter.fetch_results(
        structured_job,
        BatchSnapshot(status="ended", raw_payload={}),
        "provider-1",
    )

    assert [result.status for result in results] == ["failed"]
    assert results[0].error_code == "invalid_request_error"
    assert results[0].parsed_output is None


def test_openai_prepare_requests_builds_structured_transport_records(structured_job) -> None:
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = structured_job.provider
    requests = Runner()._build_requests(structured_job, structured_job.materialized_input())

    prepared = adapter.prepare_requests(structured_job, requests)

    assert [item.transport_record["custom_id"] for item in prepared] == [
        request.custom_id for request in requests
    ]
    assert prepared[0].transport_record["body"]["messages"][0]["content"] == "Label alpha"
    assert prepared[0].transport_record["body"]["response_format"]["json_schema"]["schema"][
        "type"
    ] == "object"
    assert prepared[0].estimated_input_tokens is not None


def test_openai_prepare_requests_builds_embedding_transport_records(embedding_job) -> None:
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = embedding_job.provider
    requests = Runner()._build_requests(embedding_job, embedding_job.materialized_input())

    prepared = adapter.prepare_requests(embedding_job, requests)

    assert prepared[0].transport_record["url"] == "/v1/embeddings"
    assert prepared[0].transport_record["body"]["input"] == "Embed alpha"
    assert prepared[0].estimated_input_tokens is not None


def test_openai_parser_marks_only_malformed_rows_failed(structured_job) -> None:
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = structured_job.provider
    lines = [
        orjson.dumps(
            {
                "custom_id": "good",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": '{"label":"ALPHA"}'}}],
                        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                    },
                },
            }
        ).decode("utf-8"),
        orjson.dumps(
            {
                "custom_id": "bad",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": "not-json"}}],
                        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                    },
                },
            }
        ).decode("utf-8"),
    ]

    results = adapter._parse_result_lines(structured_job, lines)

    assert [result.status for result in results] == ["completed", "failed"]
    assert results[0].parsed_output == {"label": "ALPHA"}
    assert results[0].raw_output_text == '{"label":"ALPHA"}'
    assert results[1].raw_output_text is None
    assert results[1].error_code == "invalid_json_response"


def test_openai_parser_accepts_fenced_json(structured_job) -> None:
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = structured_job.provider
    lines = [
        orjson.dumps(
            {
                "custom_id": "fenced",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {"message": {"content": "```json\n{\"label\":\"ALPHA\"}\n```"}}
                        ],
                        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                    },
                },
            }
        ).decode("utf-8"),
    ]

    results = adapter._parse_result_lines(structured_job, lines)

    assert [result.status for result in results] == ["completed"]
    assert results[0].parsed_output == {"label": "ALPHA"}
    assert results[0].raw_output_text == "```json\n{\"label\":\"ALPHA\"}\n```"


def test_openai_parser_surfaces_provider_declared_error_codes(structured_job) -> None:
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = structured_job.provider
    lines = [
        orjson.dumps(
            {
                "custom_id": "rate-limited",
                "error": {"code": "rate_limit_exceeded"},
                "response": {"status_code": 200, "body": {}},
            }
        ).decode("utf-8")
    ]

    results = adapter._parse_result_lines(structured_job, lines)

    assert [result.status for result in results] == ["failed"]
    assert results[0].error_code == "rate_limit_exceeded"


def test_openai_parser_marks_http_errors_failed(structured_job) -> None:
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = structured_job.provider
    lines = [
        orjson.dumps(
            {
                "custom_id": "server-error",
                "response": {
                    "status_code": 429,
                    "body": {"error": {"code": "server_error"}},
                },
            }
        ).decode("utf-8")
    ]

    results = adapter._parse_result_lines(structured_job, lines)

    assert [result.status for result in results] == ["failed"]
    assert results[0].error_code == "server_error"


def test_openai_fetch_results_preserves_mixed_output_and_error_artifacts(structured_job) -> None:
    class ResponseText:
        def __init__(self, text: str) -> None:
            self.text = text

    class Files:
        def __init__(self, contents: dict[str, str]) -> None:
            self.contents = contents

        def content(self, file_id: str) -> ResponseText:
            return ResponseText(self.contents[file_id])

    class Client:
        def __init__(self, contents: dict[str, str]) -> None:
            self.files = Files(contents)

    output_line = orjson.dumps(
        {
            "custom_id": "good",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": '{"label":"ALPHA"}'}}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                },
            },
        }
    ).decode("utf-8")
    error_line = orjson.dumps(
        {
            "custom_id": "bad",
            "response": {
                "status_code": 500,
                "body": {"error": {"code": "server_error"}},
            },
        }
    ).decode("utf-8")
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = structured_job.provider
    adapter.client = Client({"output-file": output_line, "error-file": error_line})

    results = adapter.fetch_results(
        structured_job,
        BatchSnapshot(
            status="completed",
            raw_payload={},
            output_path="output-file",
            error_path="error-file",
        ),
        "provider-1",
    )

    assert [result.custom_id for result in results] == ["good", "bad"]
    assert [result.status for result in results] == ["completed", "failed"]
    assert results[0].parsed_output == {"label": "ALPHA"}
    assert results[1].error_code == "server_error"


def test_openai_embedding_parser_marks_invalid_embedding_response_failed(embedding_job) -> None:
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = embedding_job.provider
    lines = [
        orjson.dumps(
            {
                "custom_id": "bad-embedding",
                "response": {
                    "status_code": 200,
                    "body": {"data": [], "usage": {"prompt_tokens": 8}},
                },
            }
        ).decode("utf-8")
    ]

    results = adapter._parse_result_lines(embedding_job, lines)

    assert [result.status for result in results] == ["failed"]
    assert results[0].error_code == "invalid_embedding_response"


def test_openai_parser_marks_non_text_message_content_failed(structured_job) -> None:
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = structured_job.provider
    lines = [
        orjson.dumps(
            {
                "custom_id": "bad-content",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": {"type": "text", "text": "ignored"}}}],
                        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                    },
                },
            }
        ).decode("utf-8")
    ]

    results = adapter._parse_result_lines(structured_job, lines)

    assert [result.status for result in results] == ["failed"]
    assert results[0].error_code == "invalid_json_response"


def test_anthropic_parser_marks_only_malformed_rows_failed(structured_job) -> None:
    adapter = AnthropicStructuredAdapter.__new__(AnthropicStructuredAdapter)
    adapter.config = structured_job.provider
    adapter.client = FakeAnthropicClient(
        [
            FakeAnthropicItem(
                custom_id="good",
                payload={
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [{"type": "text", "text": '{"label":"ALPHA"}'}],
                            "usage": {"input_tokens": 8, "output_tokens": 4},
                        },
                    }
                },
            ),
            FakeAnthropicItem(
                custom_id="bad",
                payload={
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [{"type": "text", "text": "not-json"}],
                            "usage": {"input_tokens": 8, "output_tokens": 4},
                        },
                    }
                },
            ),
        ]
    )

    results = adapter.fetch_results(
        structured_job,
        BatchSnapshot(status="completed", raw_payload={}),
        "provider-1",
    )

    assert [result.status for result in results] == ["completed", "failed"]
    assert results[0].parsed_output == {"label": "ALPHA"}
    assert results[0].raw_output_text == '{"label":"ALPHA"}'
    assert results[1].raw_output_text is None
    assert results[1].error_code == "invalid_json_response"


def test_anthropic_parser_surfaces_provider_failure_type(structured_job) -> None:
    adapter = AnthropicStructuredAdapter.__new__(AnthropicStructuredAdapter)
    adapter.config = structured_job.provider
    adapter.client = FakeAnthropicClient(
        [
            FakeAnthropicItem(
                custom_id="errored",
                payload={
                    "result": {
                        "type": "errored",
                        "error": {"type": "overloaded_error"},
                    }
                },
            )
        ]
    )

    results = adapter.fetch_results(
        structured_job,
        BatchSnapshot(status="completed", raw_payload={}),
        "provider-1",
    )

    assert [result.status for result in results] == ["failed"]
    assert results[0].error_code == "overloaded_error"


def test_anthropic_parser_accepts_fenced_json(structured_job) -> None:
    adapter = AnthropicStructuredAdapter.__new__(AnthropicStructuredAdapter)
    adapter.config = structured_job.provider
    adapter.client = FakeAnthropicClient(
        [
            FakeAnthropicItem(
                custom_id="fenced",
                payload={
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "```json\n{\"label\":\"ALPHA\"}\n```",
                                }
                            ],
                            "usage": {"input_tokens": 8, "output_tokens": 4},
                        },
                    }
                },
            )
        ]
    )

    results = adapter.fetch_results(
        structured_job,
        BatchSnapshot(status="completed", raw_payload={}),
        "provider-1",
    )

    assert [result.status for result in results] == ["completed"]
    assert results[0].parsed_output == {"label": "ALPHA"}
    assert results[0].raw_output_text == "```json\n{\"label\":\"ALPHA\"}\n```"


def test_anthropic_parser_prefers_tool_use_input(structured_job) -> None:
    adapter = AnthropicStructuredAdapter.__new__(AnthropicStructuredAdapter)
    adapter.config = structured_job.provider
    adapter.client = FakeAnthropicClient(
        [
            FakeAnthropicItem(
                custom_id="tool-use",
                payload={
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "labels_output",
                                    "input": {"label": "ALPHA"},
                                }
                            ],
                            "usage": {"input_tokens": 8, "output_tokens": 4},
                        },
                    }
                },
            )
        ]
    )

    results = adapter.fetch_results(
        structured_job,
        BatchSnapshot(status="completed", raw_payload={}),
        "provider-1",
    )

    assert [result.status for result in results] == ["completed"]
    assert results[0].parsed_output == {"label": "ALPHA"}
    assert results[0].raw_output_text == '{"label":"ALPHA"}'


def test_anthropic_parser_marks_unusable_content_blocks_failed(structured_job) -> None:
    adapter = AnthropicStructuredAdapter.__new__(AnthropicStructuredAdapter)
    adapter.config = structured_job.provider
    adapter.client = FakeAnthropicClient(
        [
            FakeAnthropicItem(
                custom_id="bad-content",
                payload={
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [{"type": "tool_use", "name": "noop"}],
                            "usage": {"input_tokens": 8, "output_tokens": 4},
                        },
                    }
                },
            )
        ]
    )

    results = adapter.fetch_results(
        structured_job,
        BatchSnapshot(status="completed", raw_payload={}),
        "provider-1",
    )

    assert [result.status for result in results] == ["failed"]
    assert results[0].error_code == "invalid_json_response"


def test_anthropic_fetch_results_preserves_mixed_success_and_failure(structured_job) -> None:
    adapter = AnthropicStructuredAdapter.__new__(AnthropicStructuredAdapter)
    adapter.config = structured_job.provider
    adapter.client = FakeAnthropicClient(
        [
            FakeAnthropicItem(
                custom_id="good",
                payload={
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": [{"type": "text", "text": '{"label":"ALPHA"}'}],
                            "usage": {"input_tokens": 8, "output_tokens": 4},
                        },
                    }
                },
            ),
            FakeAnthropicItem(
                custom_id="failed",
                payload={
                    "result": {
                        "type": "errored",
                        "error": {"type": "rate_limit_error"},
                    }
                },
            ),
        ]
    )

    results = adapter.fetch_results(
        structured_job,
        BatchSnapshot(status="completed", raw_payload={}),
        "provider-1",
    )

    assert [result.custom_id for result in results] == ["good", "failed"]
    assert [result.status for result in results] == ["completed", "failed"]
    assert results[0].parsed_output == {"label": "ALPHA"}
    assert results[1].error_code == "rate_limit_error"


def test_anthropic_prepare_requests_builds_transport_records(structured_job) -> None:
    adapter = AnthropicStructuredAdapter.__new__(AnthropicStructuredAdapter)
    adapter.config = structured_job.provider.__class__(
        model="claude-3-5-haiku-latest",
        api_key="test",
    )
    requests = Runner()._build_requests(structured_job, structured_job.materialized_input())

    prepared = adapter.prepare_requests(structured_job, requests)

    assert prepared[0].transport_record["params"]["messages"][0]["content"] == "Label alpha"
    assert prepared[0].transport_record["params"]["tool_choice"] == {
        "type": "tool",
        "name": "labels_output",
    }
    assert prepared[0].transport_record["params"]["tools"][0]["name"] == "labels_output"
    assert prepared[0].estimated_input_tokens is not None
    assert prepared[0].estimated_input_tokens > 0
    label_schema = prepared[0].transport_record["params"]["tools"][0]["input_schema"][
        "properties"
    ]["label"]
    assert label_schema["type"] == "string"


def test_anthropic_request_params_always_include_generated_output_tool(structured_job) -> None:
    adapter = AnthropicStructuredAdapter.__new__(AnthropicStructuredAdapter)
    adapter.config = AnthropicConfig(model="claude-3-5-haiku-latest", api_key="test")
    job = replace(
        structured_job,
        provider=adapter.config,
        prompt_builder=structured_template(
            system="Return JSON only.",
            messages="Label {{ row.text }}",
            tools=[
                {
                    "name": "lookup",
                    "description": "Fetch extra context.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            version="v1",
        ),
    )

    params = adapter._request_params(job, job.prompt_builder({"id": 1, "text": "alpha"}))

    assert [tool["name"] for tool in params["tools"]] == ["lookup", "labels_output"]
    assert params["tool_choice"] == {"type": "tool", "name": "labels_output"}


def test_anthropic_prompt_cache_marks_cacheable_blocks_not_top_level(structured_job) -> None:
    adapter = AnthropicStructuredAdapter.__new__(AnthropicStructuredAdapter)
    adapter.config = AnthropicConfig(model="claude-3-5-haiku-latest", api_key="test")
    job = replace(
        structured_job,
        provider=adapter.config,
        prompt_cache=PromptCacheConfig(mode="auto"),
    )
    payload = {
        "system": "Return JSON only.",
        "messages": [{"role": "user", "content": "Label alpha"}],
    }

    params = adapter._request_params(job, payload)

    assert "cache_control" not in params
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["messages"][0]["content"] == [{"type": "text", "text": "Label alpha"}]


def test_openai_prompt_cache_auto_leaves_request_body_shape_unchanged(structured_job) -> None:
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = structured_job.provider
    job = replace(structured_job, prompt_cache=PromptCacheConfig(mode="auto"))

    body = adapter._request_body(job, {"messages": [{"role": "user", "content": "Label alpha"}]})

    assert "prompt_cache_key" not in body
    assert body["messages"][0]["content"] == "Label alpha"


def test_openai_prompt_cache_off_matches_default_request_body(structured_job) -> None:
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = structured_job.provider
    default_job = structured_job
    uncached_job = replace(structured_job, prompt_cache=PromptCacheConfig(mode="off"))

    default_body = adapter._request_body(
        default_job, {"messages": [{"role": "user", "content": "Label alpha"}]}
    )
    uncached_body = adapter._request_body(
        uncached_job, {"messages": [{"role": "user", "content": "Label alpha"}]}
    )

    assert uncached_body == default_body


def test_openai_request_body_preserves_structured_prompt_fields() -> None:
    class Output(BaseModel):
        label: str

    job = StructuredOutputJob(
        name="labels",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "text": ["alpha"]}),
        prompt_builder=structured_template(
            messages=[{"role": "user", "content": "Label {{ row.text }}"}],
            system="Return JSON only.",
            tools=[{"type": "function", "function": {"name": "noop"}}],
            max_tokens=77,
            version="v1",
        ),
        output_model=Output,
        provider=OpenAIConfig(model="gpt-4o-mini", api_key="test"),
        result_cache=ResultCacheStoreConfig(root_uri="memory://catalog"),
    )
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = job.provider
    payload = job.prompt_builder({"id": 1, "text": "alpha"})

    body = adapter._request_body(job, payload)

    assert body["messages"][0] == {"role": "system", "content": "Return JSON only."}
    assert body["messages"][1]["content"] == "Label alpha"
    assert body["tools"] == [{"type": "function", "function": {"name": "noop"}}]
    assert body["max_completion_tokens"] == 77


cache_verbose_prompt = structured_template(
    messages=[
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": "Label this company: {{ row.text }}"},
    ],
    name="cache_verbose_prompt",
    version="v1",
)


def test_openai_prompt_cache_verbose_logs_cached_and_uncached_candidates(
    structured_job,
    caplog,
) -> None:
    _reset_prompt_cache_diagnostics()
    caplog.set_level(logging.INFO, logger="llm_batch_py.prompt_cache")
    adapter = OpenAIBatchAdapter.__new__(OpenAIBatchAdapter)
    adapter.config = structured_job.provider
    job = replace(
        structured_job,
        prompt_builder=cache_verbose_prompt,
        prompt_cache=PromptCacheConfig(mode="auto", verbose=True),
    )
    requests = Runner()._build_requests(job, job.materialized_input())

    adapter._request_body(job, requests[0].payload)
    adapter._request_body(job, requests[0].payload)

    messages = [record.getMessage() for record in caplog.records]

    assert len(messages) == 1
    assert "Prompt cache estimated analysis" in messages[0]
    assert "provider=openai" in messages[0]
    assert "boundary=provider-managed" in messages[0]
    assert "estimated_cached_locations=['messages[0]']" in messages[0]
    assert "estimated_uncached_locations=['messages[1]']" in messages[0]
    assert "Return JSON only." in messages[0]
    assert "Label this company: alpha" in messages[0]
    assert "Estimated analysis only. OpenAI decides the actual cache boundary" in messages[0]


def test_anthropic_prompt_cache_verbose_logs_explicit_breakpoint(structured_job, caplog) -> None:
    _reset_prompt_cache_diagnostics()
    caplog.set_level(logging.INFO, logger="llm_batch_py.prompt_cache")
    adapter = AnthropicStructuredAdapter.__new__(AnthropicStructuredAdapter)
    adapter.config = AnthropicConfig(model="claude-3-5-haiku-latest", api_key="test")
    job = replace(
        structured_job,
        provider=adapter.config,
        prompt_cache=PromptCacheConfig(mode="auto", verbose=True),
    )
    payload = {
        "system": [{"type": "text", "text": "Return JSON only."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Label this company: alpha"}]},
        ],
    }

    adapter._request_params(job, payload)
    adapter._request_params(job, payload)

    messages = [record.getMessage() for record in caplog.records]

    assert len(messages) == 1
    assert "Prompt cache estimated analysis" in messages[0]
    assert "provider=anthropic" in messages[0]
    assert "boundary=explicit" in messages[0]
    assert "breakpoints=['system.content[0]']" in messages[0]
    assert "estimated_cached_locations=['tools[0]', 'system.content[0]']" in messages[0]
    assert "estimated_uncached_locations=['messages[0].content[0]']" in messages[0]
    assert "Label this company: alpha" in messages[0]
    assert "llm-batch-py inserts one Anthropic cache breakpoint automatically" in messages[0]
