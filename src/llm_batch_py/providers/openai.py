from __future__ import annotations

from typing import Any

import openai
import orjson

from llm_batch_py.jobs import (
    EmbeddingJob,
    OpenAIConfig,
    StructuredOutputJob,
    output_model_json_schema,
)
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
from llm_batch_py.token_estimation import estimate_openai_batch_tokens


class OpenAIBatchAdapter(ProviderAdapter):
    provider_name = "openai"
    request_cap = 50_000
    byte_cap = 200 * 1024 * 1024
    completion_window = "24h"

    def __init__(self, config: OpenAIConfig) -> None:
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key,
            timeout=config.timeout,
            organization=config.organization,
            base_url=config.base_url,
        )

    def prepare_requests(
        self,
        job: StructuredOutputJob[Any] | EmbeddingJob,
        requests: list[RowRequest],
        *,
        input_token_estimation: str = "exact",
    ) -> list[PreparedRequest]:
        del input_token_estimation
        transport_records = [self.render_transport_record(job, request) for request in requests]

        estimated_tokens = estimate_openai_batch_tokens(
            self.config.model,
            (record["body"] for record in transport_records),
        )
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
        job: StructuredOutputJob[Any] | EmbeddingJob,
        request: RowRequest,
    ) -> dict[str, Any]:
        schema = None
        response_name = None
        if not isinstance(job, EmbeddingJob):
            schema = output_model_json_schema(job.output_model)
            response_name = job.name.replace("-", "_")

        return {
            "custom_id": request.custom_id,
            "method": "POST",
            "url": self._endpoint(job),
            "body": self._request_body(
                job,
                request.payload,
                schema=schema,
                response_name=response_name,
            ),
        }

    def submit_batch(
        self,
        job: StructuredOutputJob[Any] | EmbeddingJob,
        request_artifact_path: str,
        request_payload: bytes,
        transport_records: list[dict[str, Any]],
    ) -> SubmittedBatch:
        uploaded = self.client.files.create(
            file=("requests.jsonl", request_payload, "application/jsonl"),
            purpose="batch",
        )
        batch = self.client.batches.create(
            completion_window=self.completion_window,
            endpoint=self._endpoint(job),
            input_file_id=uploaded.id,
        )
        return SubmittedBatch(
            provider_batch_id=batch.id,
            status=batch.status,
            raw_payload=batch.model_dump(mode="json"),
        )

    def poll_batch(
        self,
        job: StructuredOutputJob[Any] | EmbeddingJob,
        provider_batch_id: str,
    ) -> BatchSnapshot:
        batch = self.client.batches.retrieve(provider_batch_id)
        output_path = None
        error_path = None
        if batch.output_file_id:
            output_path = batch.output_file_id
        if batch.error_file_id:
            error_path = batch.error_file_id
        return BatchSnapshot(
            status=batch.status,
            raw_payload=batch.model_dump(mode="json"),
            output_path=output_path,
            error_path=error_path,
        )

    def fetch_results(
        self,
        job: StructuredOutputJob[Any] | EmbeddingJob,
        batch_snapshot: BatchSnapshot,
        provider_batch_id: str,
    ) -> list[ProviderResult]:
        results: list[ProviderResult] = []
        if batch_snapshot.output_path:
            output_lines = self.client.files.content(batch_snapshot.output_path).text.splitlines()
            results.extend(self._parse_result_lines(job, output_lines))
        if batch_snapshot.error_path:
            error_lines = self.client.files.content(batch_snapshot.error_path).text.splitlines()
            results.extend(self._parse_result_lines(job, error_lines, default_status="failed"))
        return results

    def _parse_result_lines(
        self,
        job: StructuredOutputJob[Any] | EmbeddingJob,
        lines: list[str],
        default_status: str = "completed",
    ) -> list[ProviderResult]:
        parsed: list[ProviderResult] = []
        for line in lines:
            if not line:
                continue
            entry = orjson.loads(line)
            custom_id = entry["custom_id"]
            response = entry.get("response", {})
            body = response.get("body", {})
            if entry.get("error") or response.get("status_code", 200) >= 400:
                parsed.append(
                    ProviderResult(
                        custom_id=custom_id,
                        status="failed",
                        parsed_output=None,
                        raw_payload=entry,
                        error_code=(entry.get("error") or body.get("error") or {}).get("code"),
                    )
                )
                continue
            if isinstance(job, EmbeddingJob):
                try:
                    embedding = body["data"][0]["embedding"]
                except (KeyError, IndexError, TypeError):
                    parsed.append(
                        ProviderResult(
                            custom_id=custom_id,
                            status="failed",
                            parsed_output=None,
                            raw_payload=entry,
                            error_code="invalid_embedding_response",
                        )
                    )
                    continue
                usage = body.get("usage", {})
                parsed.append(
                    ProviderResult(
                        custom_id=custom_id,
                        status=default_status,
                        parsed_output={"embedding": embedding, "embedding_dim": len(embedding)},
                        raw_payload=entry,
                        input_tokens=usage.get("prompt_tokens"),
                    )
                )
                continue
            try:
                text = _message_content_text(body["choices"][0]["message"]["content"])
                parsed_output = parse_json_response_text(text)
            except (KeyError, IndexError, TypeError, orjson.JSONDecodeError):
                parsed.append(
                    ProviderResult(
                        custom_id=custom_id,
                        status="failed",
                        parsed_output=None,
                        raw_payload=entry,
                        error_code="invalid_json_response",
                    )
                )
                continue
            usage = body.get("usage", {})
            parsed.append(
                ProviderResult(
                    custom_id=custom_id,
                    status=default_status,
                    parsed_output=parsed_output,
                    raw_payload=entry,
                    raw_output_text=text,
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                )
            )
        return parsed

    def _endpoint(self, job: StructuredOutputJob[Any] | EmbeddingJob) -> str:
        return "/v1/embeddings" if isinstance(job, EmbeddingJob) else "/v1/chat/completions"

    def _request_body(
        self,
        job: StructuredOutputJob[Any] | EmbeddingJob,
        payload: dict[str, Any] | str,
        *,
        schema: dict[str, Any] | None = None,
        response_name: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(job, EmbeddingJob):
            if not isinstance(payload, str):
                raise ValueError("Embedding text builders must return a string")
            body: dict[str, Any] = {"model": self.config.model, "input": payload}
            if self.config.dimensions is not None:
                body["dimensions"] = self.config.dimensions
            return body

        normalized = _normalize_structured_payload(payload)
        prompt_cache = job.prompt_cache
        messages = list(normalized["messages"])
        system = normalized.get("system")
        if system is not None:
            messages = [{"role": "system", "content": system}, *messages]
        body = {
            "model": self.config.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_name or job.name.replace("-", "_"),
                    "strict": True,
                    "schema": (
                        schema if schema is not None else job.output_model.model_json_schema()
                    ),
                },
            },
        }
        if normalized.get("tools") is not None:
            body["tools"] = normalized["tools"]
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        max_output_tokens = normalized.get("max_tokens", self.config.max_output_tokens)
        if max_output_tokens is not None:
            body["max_completion_tokens"] = max_output_tokens
        if prompt_cache_enabled(prompt_cache):
            emit_prompt_cache_diagnostic_once(
                provider_name="openai",
                model=self.config.model,
                config=prompt_cache,
                payload={
                    "messages": messages,
                    **(
                        {"tools": normalized["tools"]}
                        if normalized.get("tools") is not None
                        else {}
                    ),
                },
                boundary="provider-managed",
                note=(
                    "Estimated analysis only. OpenAI decides the actual cache boundary; "
                    "llm-batch-py reports hints only."
                ),
                request_fields={},
            )
        return body


def _normalize_structured_payload(payload: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(payload, str):
        return {"messages": [{"role": "user", "content": payload}]}
    if "messages" in payload:
        return dict(payload)
    raise ValueError("Structured prompt builders must return a string or a dict with messages")


def _message_content_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(chunk.get("text", "") for chunk in content if isinstance(chunk, dict))
    if isinstance(content, str):
        return content
    raise TypeError("missing message content")
