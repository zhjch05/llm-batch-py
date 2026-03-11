from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import orjson

from llm_batch_py.jobs import EmbeddingJob, ResolvedInputTokenEstimation, StructuredOutputJob


@dataclass(frozen=True)
class RowRequest:
    request_id: str
    custom_id: str
    row_key: dict[str, Any]
    cache_key: str
    payload: dict[str, Any] | str
    payload_json: str
    payload_bytes: int
    prompt_version: str


@dataclass(frozen=True)
class PreparedRequest:
    request: RowRequest
    transport_record: dict[str, Any]
    estimated_input_tokens: int | None = None


@dataclass(frozen=True)
class SubmittedBatch:
    provider_batch_id: str
    status: str
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class BatchSnapshot:
    status: str
    raw_payload: dict[str, Any]
    output_path: str | None = None
    error_path: str | None = None


@dataclass(frozen=True)
class ProviderResult:
    custom_id: str
    status: str
    parsed_output: dict[str, Any] | None
    raw_payload: dict[str, Any]
    raw_output_text: str | None = None
    error_code: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class ProviderAdapter(Protocol):
    provider_name: str
    request_cap: int
    byte_cap: int

    def render_transport_record(
        self,
        job: StructuredOutputJob[Any] | EmbeddingJob,
        request: RowRequest,
    ) -> dict[str, Any]: ...

    def prepare_requests(
        self,
        job: StructuredOutputJob[Any] | EmbeddingJob,
        requests: list[RowRequest],
        *,
        input_token_estimation: ResolvedInputTokenEstimation = "exact",
    ) -> list[PreparedRequest]: ...

    def submit_batch(
        self,
        job: StructuredOutputJob[Any] | EmbeddingJob,
        request_artifact_path: str,
        request_payload: bytes,
        transport_records: list[dict[str, Any]],
    ) -> SubmittedBatch: ...

    def poll_batch(
        self,
        job: StructuredOutputJob[Any] | EmbeddingJob,
        provider_batch_id: str,
    ) -> BatchSnapshot: ...

    def fetch_results(
        self,
        job: StructuredOutputJob[Any] | EmbeddingJob,
        batch_snapshot: BatchSnapshot,
        provider_batch_id: str,
    ) -> list[ProviderResult]: ...


def parse_json_response_text(text: str) -> dict[str, Any]:
    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        fenced = _strip_code_fence(text)
        if fenced == text:
            raise
        return orjson.loads(fenced)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) < 3:
        return text
    if not lines[-1].strip().startswith("```"):
        return text
    return "\n".join(lines[1:-1]).strip()
