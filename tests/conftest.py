from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl
import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_batch_py import (
    BatchConfig,
    EmbeddingJob,
    LockConfig,
    OpenAIConfig,
    ResultCacheStoreConfig,
    StructuredOutputJob,
    prompt_udf,
)
from llm_batch_py.providers.base import (
    BatchSnapshot,
    PreparedRequest,
    ProviderResult,
    SubmittedBatch,
)


class LabelOutput(BaseModel):
    label: str


@prompt_udf(version="v1")
def structured_prompt(row):
    return {
        "messages": [{"role": "user", "content": f"Label {row.text}"}],
        "expected": {"label": row.text.upper()},
    }


@prompt_udf(version="v1")
def embedding_text(row):
    return f"Embed {row.text}"


@dataclass
class FakeBatch:
    status: str
    requests: list[dict[str, Any]]
    results: list[ProviderResult]


FakeOutcomeName = Literal["completed", "failed", "malformed", "invalid_embedding"]


@dataclass(frozen=True)
class FakeOutcome:
    kind: FakeOutcomeName
    error_code: str | None = None
    parsed_output: dict[str, Any] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class FakeAdapter:
    provider_name = "openai"
    request_cap = 50_000
    byte_cap = 200 * 1024 * 1024

    def __init__(
        self,
        mode: str = "structured",
        *,
        submit_failures: list[Exception] | None = None,
    ) -> None:
        self.mode = mode
        self.submit_failures = list(submit_failures or [])
        self.submissions: list[str] = []
        self.submit_attempts = 0
        self.max_submit_inflight = 0
        self._submit_inflight = 0
        self.batches: dict[str, FakeBatch] = {}

    def prepare_requests(self, job, requests, *, input_token_estimation="exact"):
        del input_token_estimation
        return [
            PreparedRequest(
                request=request,
                transport_record=self.render_transport_record(job, request),
                estimated_input_tokens=max(1, request.payload_bytes // 4),
            )
            for request in requests
        ]

    def render_transport_record(self, job, request):
        del job
        return {"custom_id": request.custom_id, "payload": request.payload}

    def submit_batch(self, job, request_artifact_path, request_payload, transport_records):
        self.submit_attempts += 1
        self._submit_inflight += 1
        self.max_submit_inflight = max(self.max_submit_inflight, self._submit_inflight)
        try:
            if self.submit_failures:
                raise self.submit_failures.pop(0)
            provider_batch_id = f"provider_{len(self.batches) + 1}"
            self.submissions.append(provider_batch_id)
            self.batches[provider_batch_id] = FakeBatch(
                status="submitted",
                requests=transport_records,
                results=[],
            )
            return SubmittedBatch(
                provider_batch_id=provider_batch_id,
                status="submitted",
                raw_payload={"id": provider_batch_id, "status": "submitted"},
            )
        finally:
            self._submit_inflight -= 1

    def poll_batch(self, job, provider_batch_id):
        batch = self.batches[provider_batch_id]
        return BatchSnapshot(
            status=batch.status, raw_payload={"id": provider_batch_id, "status": batch.status}
        )

    def fetch_results(self, job, batch_snapshot, provider_batch_id):
        return list(self.batches[provider_batch_id].results)

    def complete_all(
        self,
        *,
        fail_custom_ids: set[str] | None = None,
        outcomes: dict[str, FakeOutcome | FakeOutcomeName] | None = None,
    ) -> None:
        failures = fail_custom_ids or set()
        resolved_outcomes = outcomes or {}
        for _provider_batch_id, batch in self.batches.items():
            results: list[ProviderResult] = []
            for request in batch.requests:
                custom_id = request["custom_id"]
                payload = request["payload"]
                outcome = resolved_outcomes.get(custom_id)
                if custom_id in failures and outcome is None:
                    outcome = FakeOutcome(kind="failed", error_code="invalid_request")
                if outcome is None:
                    outcome = "completed"
                results.append(self._result_from_outcome(custom_id, payload, outcome))
            batch.results = results
            batch.status = "completed"

    def _result_from_outcome(
        self,
        custom_id: str,
        payload: dict[str, Any] | str,
        outcome: FakeOutcome | FakeOutcomeName,
    ) -> ProviderResult:
        if isinstance(outcome, str):
            outcome = FakeOutcome(kind=outcome)

        if outcome.kind == "failed":
            return ProviderResult(
                custom_id=custom_id,
                status="failed",
                parsed_output=None,
                raw_payload={"custom_id": custom_id},
                error_code=outcome.error_code or "invalid_request",
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
            )

        if self.mode == "embeddings":
            text = payload
            if outcome.kind == "invalid_embedding":
                parsed_output = outcome.parsed_output or {"embedding": None, "embedding_dim": None}
            else:
                embedding = [float(len(text)), 1.0]
                parsed_output = outcome.parsed_output or {
                    "embedding": embedding,
                    "embedding_dim": len(embedding),
                }
            return ProviderResult(
                custom_id=custom_id,
                status="completed",
                parsed_output=parsed_output,
                raw_payload={"custom_id": custom_id},
                raw_output_text=None,
                input_tokens=outcome.input_tokens or len(text.split()),
                output_tokens=outcome.output_tokens,
            )

        if outcome.kind == "malformed":
            parsed_output = outcome.parsed_output or {"unexpected": "value"}
        else:
            parsed_output = outcome.parsed_output or payload["expected"]
        return ProviderResult(
            custom_id=custom_id,
            status="completed",
            parsed_output=parsed_output,
            raw_payload={"custom_id": custom_id},
            raw_output_text=f'{{"label":"{parsed_output["label"]}"}}'
            if isinstance(parsed_output, dict) and "label" in parsed_output
            else None,
            input_tokens=outcome.input_tokens or 8,
            output_tokens=outcome.output_tokens or 4,
        )


@pytest.fixture
def input_df() -> pl.DataFrame:
    return pl.DataFrame({"id": [1, 2], "text": ["alpha", "beta"]})


@pytest.fixture
def structured_job(tmp_path, input_df) -> StructuredOutputJob[LabelOutput]:
    return StructuredOutputJob(
        name="labels",
        key_cols=["id"],
        input_df=input_df,
        prompt_builder=structured_prompt,
        output_model=LabelOutput,
        provider=OpenAIConfig(model="gpt-4o-mini", api_key="test"),
        result_cache=ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")),
        lock=LockConfig(),
        batch=BatchConfig(),
    )


@pytest.fixture
def embedding_job(tmp_path, input_df) -> EmbeddingJob:
    return EmbeddingJob(
        name="embeddings",
        key_cols=["id"],
        input_df=input_df,
        text_builder=embedding_text,
        provider=OpenAIConfig(model="text-embedding-3-small", api_key="test"),
        result_cache=ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog-emb")),
        lock=LockConfig(),
    )
