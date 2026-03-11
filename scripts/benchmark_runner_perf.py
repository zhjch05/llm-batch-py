from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from time import perf_counter

import polars as pl
from pydantic import BaseModel

from llm_batch_py import BatchConfig, CacheStoreConfig, OpenAIConfig, StructuredOutputJob, prompt_udf
from llm_batch_py._core_wrapper import canonical_json, jsonl_dump_bytes
from llm_batch_py.catalog import MANIFEST_BATCHES, MANIFEST_REQUESTS, MANIFEST_RESULTS, ParquetCatalog
from llm_batch_py.providers.base import PreparedRequest
from llm_batch_py.runner import Runner


class LabelOutput(BaseModel):
    label: str


@prompt_udf(version="perf-v1")
def structured_prompt(row):
    return {
        "messages": [{"role": "user", "content": f"Label {row.text}"}],
        "expected": {"label": row.text.upper()},
    }


@dataclass(frozen=True)
class PhaseResult:
    name: str
    seconds: float


class FakeAdapter:
    provider_name = "openai"
    request_cap = 50_000
    byte_cap = 200 * 1024 * 1024

    def prepare_requests(self, job, requests):
        return [
            PreparedRequest(
                request=request,
                transport_record={"custom_id": request.custom_id, "payload": request.payload},
                estimated_input_tokens=max(1, request.payload_bytes // 4),
            )
            for request in requests
        ]


def benchmark(rows: int) -> list[PhaseResult]:
    input_df = pl.DataFrame(
        {
            "id": list(range(rows)),
            "text": [f"row-{index}" for index in range(rows)],
        }
    )
    with tempfile.TemporaryDirectory(prefix="llm-batch-py-bench-") as tempdir:
        job = StructuredOutputJob(
            name="labels",
            key_cols=["id"],
            input_df=input_df,
            prompt_builder=structured_prompt,
            output_model=LabelOutput,
            provider=OpenAIConfig(model="gpt-4o-mini", api_key="test"),
            cache_store=CacheStoreConfig(root_uri=tempdir),
            batch=BatchConfig(),
        )
        runner = Runner()
        adapter = FakeAdapter()
        catalog = ParquetCatalog(job.cache_store)
        phases: list[PhaseResult] = []

        started = perf_counter()
        requests = runner._build_requests(job, input_df)
        phases.append(PhaseResult("request_building", perf_counter() - started))

        started = perf_counter()
        prepared_requests = adapter.prepare_requests(job, requests)
        phases.append(PhaseResult("token_preparation", perf_counter() - started))

        started = perf_counter()
        request_payload = jsonl_dump_bytes(
            [prepared_request.transport_record for prepared_request in prepared_requests]
        )
        phases.append(PhaseResult("jsonl_serialization", perf_counter() - started))

        started = perf_counter()
        catalog.append_manifest(
            MANIFEST_BATCHES,
            [
                {
                    "event_at": "2025-01-01T00:00:00+00:00",
                    "job_name": job.name,
                    "batch_id": "bench-batch",
                    "provider_batch_id": "provider-1",
                    "provider": job.provider.provider_name,
                    "model": job.provider.model,
                    "endpoint_kind": job.endpoint_kind,
                    "status": "submitted",
                    "request_count": len(prepared_requests),
                    "artifact_uri": "memory://requests.jsonl",
                    "raw_json": "{}",
                }
            ],
        )
        catalog.append_manifest(
            MANIFEST_REQUESTS,
            [
                {
                    "event_at": "2025-01-01T00:00:00+00:00",
                    "job_name": job.name,
                    "request_id": prepared_request.request.request_id,
                    "batch_id": "bench-batch",
                    "custom_id": prepared_request.request.custom_id,
                    "cache_key": prepared_request.request.cache_key,
                    "provider": job.provider.provider_name,
                    "model": job.provider.model,
                    "endpoint_kind": job.endpoint_kind,
                    "row_key_json": canonical_json(prepared_request.request.row_key),
                    "payload_json": prepared_request.request.payload_json,
                    "prompt_version": prepared_request.request.prompt_version,
                    "status": "submitted",
                }
                for prepared_request in prepared_requests
            ],
        )
        catalog.append_manifest(
            MANIFEST_RESULTS,
            [
                {
                    "event_at": "2025-01-01T00:01:00+00:00",
                    "job_name": job.name,
                    "batch_id": "bench-batch",
                    "request_id": prepared_request.request.request_id,
                    "custom_id": prepared_request.request.custom_id,
                    "cache_key": prepared_request.request.cache_key,
                    "provider": job.provider.provider_name,
                    "model": job.provider.model,
                    "endpoint_kind": job.endpoint_kind,
                    "status": "completed",
                    "error_code": None,
                    "row_key_json": canonical_json(prepared_request.request.row_key),
                    "parsed_json": f'{{"label":"ROW-{index}"}}',
                    "raw_json": "{}",
                    "input_tokens": 8,
                    "output_tokens": 4,
                }
                for index, prepared_request in enumerate(prepared_requests)
            ],
        )
        _ = request_payload
        phases.append(PhaseResult("manifest_writes", perf_counter() - started))

        completed_results = runner._completed_results(catalog, job.name)
        row_states = {
            request.cache_key: runner._completed_row_state(
                job,
                request,
                completed_results[request.cache_key],
                cached=False,
            )
            for request in requests
        }
        started = perf_counter()
        runner._materialize(job, input_df, requests, row_states)
        phases.append(PhaseResult("result_materialization", perf_counter() - started))
        return phases


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark llm-batch-py local throughput hot paths.")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of rows to benchmark.")
    args = parser.parse_args()

    results = benchmark(args.rows)
    print(f"rows={args.rows}")
    for result in results:
        print(f"{result.name}: {result.seconds:.3f}s")


if __name__ == "__main__":
    main()
