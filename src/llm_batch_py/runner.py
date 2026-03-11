from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

import orjson
import polars as pl
from pydantic import ValidationError

from llm_batch_py._core_wrapper import canonical_json, jsonl_dump_bytes, stable_hash
from llm_batch_py.catalog import (
    MANIFEST_BATCHES,
    MANIFEST_REQUESTS,
    MANIFEST_RESULTS,
    ParquetCatalog,
    new_batch_id,
    new_run_id,
    utc_now_iso,
)
from llm_batch_py.jobs import (
    AnthropicConfig,
    EmbeddingJob,
    Job,
    ResolvedInputTokenEstimation,
    RunSummary,
    StreamRunSummary,
    output_model_json_schema,
    resolve_result_metadata_columns,
    structured_output_result_column_map,
    validate_job_input,
    validate_job_input_columns,
)
from llm_batch_py.providers.anthropic import AnthropicStructuredAdapter
from llm_batch_py.providers.base import (
    PreparedRequest,
    ProviderAdapter,
    ProviderResult,
    RowRequest,
)
from llm_batch_py.providers.openai import OpenAIBatchAdapter
from llm_batch_py.token_estimation import estimate_job_output_tokens

STATUS_COMPLETED = "completed"
STATUS_PENDING = "pending"
STATUS_SUBMITTED = "submitted"
STATUS_FAILED = "failed"
STATUS_CACHED = "cached"
STATUS_SUBMIT_FAILED_RETRYABLE = "submit_failed_retryable"
STATUS_SUBMIT_FAILED = "submit_failed"

RETRYABLE_ERROR_CODES = {
    "rate_limit_error",
    "rate_limit_exceeded",
    "overloaded_error",
    "server_error",
    "timeout",
    "expired",
}

ACTIVE_BATCH_STATUSES = {"in_progress", "validating", "finalizing", "queued", "submitted"}
TERMINAL_BATCH_STATUSES = {"ended", "completed"}
INLINE_SUBMIT_RETRY_ATTEMPTS = 2


@dataclass(frozen=True)
class PendingState:
    batch_id: str
    provider_batch_id: str | None
    status: str


@dataclass(frozen=True)
class SubmitFailure:
    batch_id: str | None
    retryable: bool
    error_code: str
    raw_payload: dict[str, Any]


class Runner:
    def __init__(self) -> None:
        self.last_summary: RunSummary | None = None
        self.last_stream_summary: StreamRunSummary | None = None

    def run(
        self,
        job: Job,
        *,
        dry_run: bool = False,
        avg_output_tokens: int = 0,
        metadata_columns: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        input_df = validate_job_input(job)
        self.last_stream_summary = None
        resolved_metadata_columns = resolve_result_metadata_columns(metadata_columns)
        result_df, summary = self._run_validated_chunk(
            job,
            input_df,
            dry_run=dry_run,
            avg_output_tokens=avg_output_tokens,
            metadata_columns=resolved_metadata_columns,
        )
        self.last_summary = summary
        return result_df

    def run_stream(
        self,
        job: Job,
        *,
        input_batch_rows: int | None = None,
        order_by: Sequence[str] | None = None,
        input_batches: Iterable[pl.DataFrame] | None = None,
        dry_run: bool = False,
        avg_output_tokens: int = 0,
        metadata_columns: Sequence[str] | None = None,
    ) -> Iterator[pl.DataFrame]:
        if input_batches is None and input_batch_rows is None:
            raise ValueError("input_batch_rows is required when input_batches is not provided")
        if input_batches is not None and input_batch_rows is not None:
            raise ValueError("Pass only one of input_batch_rows or input_batches")
        if input_batch_rows is not None and input_batch_rows <= 0:
            raise ValueError("input_batch_rows must be greater than 0")
        self.last_summary = None
        self.last_stream_summary = StreamRunSummary(
            chunk_count=0,
            total_rows=0,
            result_cache_hits=0,
            inflight_rows=0,
            submitted_rows=0,
            completed_rows=0,
            failed_rows=0,
            estimated_input_tokens=0,
            estimated_output_tokens=0,
            estimated_cost_usd=0.0,
            batches_submitted=0,
            dry_run=dry_run,
        )
        resolved_metadata_columns = resolve_result_metadata_columns(metadata_columns)

        chunk_iter = self._validated_stream_chunks(
            job,
            input_batch_rows=input_batch_rows,
            order_by=order_by,
            input_batches=input_batches,
        )

        def stream() -> Iterator[pl.DataFrame]:
            catalog = ParquetCatalog(job.result_cache, job.lock)
            adapter = self._adapter(job)
            aggregate_cost = 0.0
            aggregate_cost_unknown = False
            chunk_count = 0
            total_rows = 0
            result_cache_hits = 0
            inflight_rows = 0
            submitted_rows = 0
            completed_rows = 0
            failed_rows = 0
            estimated_input_tokens = 0
            estimated_output_tokens = 0
            batches_submitted = 0

            for chunk_df in chunk_iter:
                run_id = new_run_id()
                lock = catalog.acquire_lock(job.name, run_id)
                catalog.record_run(job.name, run_id, dry_run)
                try:
                    self._poll_batches(catalog, job, adapter)
                    result_df, summary = self._run_validated_chunk_with_context(
                        job,
                        chunk_df,
                        catalog=catalog,
                        adapter=adapter,
                        dry_run=dry_run,
                        avg_output_tokens=avg_output_tokens,
                        metadata_columns=resolved_metadata_columns,
                        run_id=run_id,
                    )
                finally:
                    catalog.release_lock(lock)

                chunk_count += 1
                total_rows += summary.total_rows
                result_cache_hits += summary.result_cache_hits
                inflight_rows += summary.inflight_rows
                submitted_rows += summary.submitted_rows
                completed_rows += summary.completed_rows
                failed_rows += summary.failed_rows
                estimated_input_tokens += summary.estimated_input_tokens
                estimated_output_tokens += summary.estimated_output_tokens
                batches_submitted += summary.batches_submitted
                if summary.estimated_cost_usd is None:
                    aggregate_cost_unknown = True
                elif not aggregate_cost_unknown:
                    aggregate_cost += summary.estimated_cost_usd
                self.last_summary = summary
                self.last_stream_summary = StreamRunSummary(
                    chunk_count=chunk_count,
                    total_rows=total_rows,
                    result_cache_hits=result_cache_hits,
                    inflight_rows=inflight_rows,
                    submitted_rows=submitted_rows,
                    completed_rows=completed_rows,
                    failed_rows=failed_rows,
                    estimated_input_tokens=estimated_input_tokens,
                    estimated_output_tokens=estimated_output_tokens,
                    estimated_cost_usd=None if aggregate_cost_unknown else aggregate_cost,
                    batches_submitted=batches_submitted,
                    dry_run=dry_run,
                )
                yield result_df

        return stream()

    def _run_validated_chunk(
        self,
        job: Job,
        input_df: pl.DataFrame,
        *,
        dry_run: bool,
        avg_output_tokens: int,
        metadata_columns: Sequence[str],
    ) -> tuple[pl.DataFrame, RunSummary]:
        catalog = ParquetCatalog(job.result_cache, job.lock)
        adapter = self._adapter(job)
        run_id = new_run_id()
        lock = catalog.acquire_lock(job.name, run_id)
        catalog.record_run(job.name, run_id, dry_run)

        try:
            self._poll_batches(catalog, job, adapter)
            return self._run_validated_chunk_with_context(
                job,
                input_df,
                catalog=catalog,
                adapter=adapter,
                dry_run=dry_run,
                avg_output_tokens=avg_output_tokens,
                metadata_columns=metadata_columns,
                run_id=run_id,
            )
        finally:
            catalog.release_lock(lock)

    def _run_validated_chunk_with_context(
        self,
        job: Job,
        input_df: pl.DataFrame,
        *,
        catalog: ParquetCatalog,
        adapter: ProviderAdapter,
        dry_run: bool,
        avg_output_tokens: int,
        metadata_columns: Sequence[str],
        run_id: str,
    ) -> tuple[pl.DataFrame, RunSummary]:
        requests = self._build_requests(job, input_df)

        completed_results = self._completed_results(catalog, job.name)
        pending_requests = self._pending_requests(catalog, job.name)
        retryable_submit_batches = self._retryable_submit_batches(catalog, job.name)
        retryable_submit_cache_keys = {
            row["cache_key"] for batch in retryable_submit_batches for row in batch["request_rows"]
        }
        new_requests: list[RowRequest] = []
        row_states: dict[str, dict[str, Any]] = {}
        result_cache_hits = 0
        inflight_rows = 0
        failed_rows = 0

        retry_counts = Counter(self._provider_request_attempts(catalog, job.name))

        for request in requests:
            result = completed_results.get(request.cache_key)
            if result and self._is_terminal_success(result):
                row_states[request.cache_key] = self._completed_row_state(
                    job, adapter, request, result, cached=True
                )
                result_cache_hits += 1
                continue
            if result and self._should_hold_failure(
                result, retry_counts[request.cache_key], job.batch.max_retries
            ):
                row_states[request.cache_key] = self._failed_row_state(
                    job, adapter, request, result
                )
                failed_rows += 1
                continue
            pending = pending_requests.get(request.cache_key)
            if pending is not None:
                row_states[request.cache_key] = self._pending_row_state(
                    job,
                    adapter,
                    request,
                    pending,
                )
                inflight_rows += 1
                continue
            if request.cache_key in retryable_submit_cache_keys:
                continue
            new_requests.append(request)

        estimated_input_tokens = 0
        request_groups: list[list[PreparedRequest]] = []
        oversized_requests: list[PreparedRequest] = []
        input_token_estimation = self._resolve_input_token_estimation(job, dry_run=dry_run)
        if new_requests:
            prepared_requests = adapter.prepare_requests(
                job,
                new_requests,
                input_token_estimation=input_token_estimation,
            )
            batch_size = self._choose_batch_size(job, adapter)
            request_groups, oversized_requests = self._chunk_prepared_requests(
                prepared_requests,
                max_requests=batch_size,
                byte_cap=adapter.byte_cap,
            )
            estimated_tokens = [
                prepared_request.estimated_input_tokens or 0
                for request_group in request_groups
                for prepared_request in request_group
            ]
            estimated_input_tokens = sum(estimated_tokens)

        estimated_output_tokens = sum(len(request_group) for request_group in request_groups) * (
            estimate_job_output_tokens(job, avg_output_tokens)
        )
        estimated_cost = self._estimate_cost(
            job,
            estimated_input_tokens,
            estimated_output_tokens,
            input_token_estimation=input_token_estimation,
        )
        submitted_batches = 0
        submitted_rows = 0

        if oversized_requests:
            oversized_failure = {
                "type": "RequestTooLarge",
                "message": (
                    "Request transport record exceeds the provider batch byte cap and cannot be "
                    "submitted."
                ),
                "byte_cap": adapter.byte_cap,
            }
            self._persist_local_failures(
                catalog,
                job,
                oversized_requests,
                error_code="request_too_large",
                raw_payload=oversized_failure,
            )
            for prepared_request in oversized_requests:
                row_states[prepared_request.request.cache_key] = self._failed_row_state(
                    job,
                    adapter,
                    prepared_request.request,
                    {
                        "batch_id": None,
                        "provider": job.provider.provider_name,
                        "model": job.provider.model,
                        "error_code": "request_too_large",
                        "raw_json": canonical_json(oversized_failure),
                        "event_at": utc_now_iso(),
                    },
                )
                failed_rows += 1

        blocked_submission = False
        if retryable_submit_batches:
            recovery = self._recover_retryable_submit_batches(
                catalog,
                job,
                adapter,
                requests,
                retryable_submit_batches,
                dry_run=dry_run,
            )
            row_states.update(recovery["row_states"])
            inflight_rows += recovery["inflight_rows"]
            failed_rows += recovery["failed_rows"]
            submitted_batches += recovery["submitted_batches"]
            submitted_rows += recovery["submitted_rows"]
            blocked_submission = recovery["blocked"]

        if not dry_run and request_groups and not blocked_submission:
            for request_group in request_groups:
                submission = self._submit_new_batch(catalog, job, adapter, request_group)
                if submission["pending_state"] is None:
                    row_states.update(submission["row_states"])
                    inflight_rows += submission["inflight_rows"]
                    failed_rows += submission["failed_rows"]
                    blocked_submission = submission["blocked"]
                    break
                submitted_batches += 1
                submitted_rows += len(request_group)
                pending_state = submission["pending_state"]
                for prepared_request in request_group:
                    request = prepared_request.request
                    row_states[request.cache_key] = self._pending_row_state(
                        job,
                        adapter,
                        request,
                        pending_state,
                    )
        result_df = self._materialize(
            job,
            adapter,
            input_df,
            requests,
            row_states,
            metadata_columns=metadata_columns,
        )
        completed_rows = len(
            [
                row
                for row in row_states.values()
                if row["llm_batch_py_status"] in {STATUS_COMPLETED, STATUS_CACHED}
            ]
        )
        summary = RunSummary(
            run_id=run_id,
            total_rows=len(requests),
            result_cache_hits=result_cache_hits,
            inflight_rows=inflight_rows,
            submitted_rows=submitted_rows,
            completed_rows=completed_rows,
            failed_rows=failed_rows,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            estimated_cost_usd=estimated_cost,
            input_token_estimation=input_token_estimation,
            batches_submitted=submitted_batches,
            dry_run=dry_run,
        )
        return result_df, summary

    def _adapter(self, job: Job) -> ProviderAdapter:
        if isinstance(job, EmbeddingJob):
            return OpenAIBatchAdapter(job.provider)
        if job.provider.provider_name == "anthropic":
            return AnthropicStructuredAdapter(job.provider)
        return OpenAIBatchAdapter(job.provider)

    def _build_requests(self, job: Job, input_df: pl.DataFrame) -> list[RowRequest]:
        requests: list[RowRequest] = []
        builder = job.text_builder if isinstance(job, EmbeddingJob) else job.prompt_builder
        schema = (
            None if isinstance(job, EmbeddingJob) else output_model_json_schema(job.output_model)
        )
        cache_key_context = {
            "job_name": job.name,
            "prompt_version": builder.version,
            "provider": job.provider.provider_name,
            "model": job.provider.model,
            "endpoint_kind": job.endpoint_kind,
            "schema": schema,
            "temperature": getattr(job.provider, "temperature", None),
            "max_output_tokens": getattr(job.provider, "max_output_tokens", None),
            "dimensions": getattr(job.provider, "dimensions", None),
            "base_url": getattr(job.provider, "base_url", None),
            "organization": getattr(job.provider, "organization", None),
            "prompt_cache": (
                asdict(job.prompt_cache)
                if not isinstance(job, EmbeddingJob) and job.prompt_cache is not None
                else None
            ),
        }
        for row in input_df.iter_rows(named=True):
            row_key = {column: row[column] for column in job.key_cols}
            payload = builder(row)
            if isinstance(job, EmbeddingJob) and not isinstance(payload, str):
                raise ValueError("Embedding jobs require text_builder to return a string")
            payload_json = canonical_json(payload)
            request_id = f"req_{uuid4().hex}"
            cache_key = stable_hash({"payload": payload, **cache_key_context})
            requests.append(
                RowRequest(
                    request_id=request_id,
                    custom_id=request_id,
                    row_key=row_key,
                    cache_key=cache_key,
                    payload=payload,
                    payload_json=payload_json,
                    payload_bytes=len(payload_json.encode("utf-8")),
                    prompt_version=builder.version,
                )
            )
        return requests

    def _poll_batches(
        self, catalog: ParquetCatalog, job: Job, adapter: ProviderAdapter
    ) -> list[ProviderResult]:
        latest_batches = catalog.latest_batches(job.name)
        if not len(latest_batches):
            return []
        requests_df = catalog.latest_requests(job.name)
        request_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        if len(requests_df):
            for row in requests_df.iter_rows(named=True):
                request_lookup[(row["batch_id"], row["custom_id"])] = row

        results: list[ProviderResult] = []
        for batch in latest_batches.iter_rows(named=True):
            status = batch["status"]
            needs_ingestion = bool(
                status in TERMINAL_BATCH_STATUSES and not batch.get("results_ingested_at")
            )
            if status not in ACTIVE_BATCH_STATUSES and not needs_ingestion:
                continue
            if status in ACTIVE_BATCH_STATUSES:
                snapshot = adapter.poll_batch(job, batch["provider_batch_id"])
                catalog.append_manifest(
                    MANIFEST_BATCHES,
                    [
                        {
                            **batch,
                            "event_at": utc_now_iso(),
                            "status": snapshot.status,
                            "raw_json": canonical_json(snapshot.raw_payload),
                            "output_artifact": snapshot.output_path,
                            "error_artifact": snapshot.error_path,
                        }
                    ],
                )
            else:
                snapshot = adapter.poll_batch(job, batch["provider_batch_id"])
            if snapshot.status not in TERMINAL_BATCH_STATUSES:
                continue
            if batch.get("results_ingested_at"):
                continue
            fetched = adapter.fetch_results(job, snapshot, batch["provider_batch_id"])
            request_rows = [
                row
                for (request_batch_id, _custom_id), row in request_lookup.items()
                if request_batch_id == batch["batch_id"]
            ]
            if not self._is_complete_batch_result_set(request_rows, fetched):
                result_rows = self._result_rows_for_batch(
                    job,
                    batch_id=batch["batch_id"],
                    request_lookup=request_lookup,
                    fetched=fetched,
                )
                if result_rows:
                    catalog.append_manifest(MANIFEST_RESULTS, result_rows)
                continue
            results.extend(fetched)
            result_rows = self._result_rows_for_batch(
                job,
                batch_id=batch["batch_id"],
                request_lookup=request_lookup,
                fetched=fetched,
            )
            if result_rows:
                catalog.append_manifest(MANIFEST_RESULTS, result_rows)
            catalog.append_manifest(
                MANIFEST_BATCHES,
                [
                    {
                        **batch,
                        "event_at": utc_now_iso(),
                        "status": snapshot.status,
                        "results_ingested_at": utc_now_iso(),
                        "raw_json": canonical_json(snapshot.raw_payload),
                        "output_artifact": snapshot.output_path,
                        "error_artifact": snapshot.error_path,
                    }
                ],
            )
        return results

    def _submit_batch(
        self,
        catalog: ParquetCatalog,
        job: Job,
        adapter: ProviderAdapter,
        requests: list[PreparedRequest],
        *,
        batch_id: str | None = None,
        artifact_uri: str | None = None,
        request_rows: list[dict[str, Any]] | None = None,
        submit_attempts: int = 1,
    ) -> PendingState | SubmitFailure:
        batch_id = batch_id or new_batch_id()
        transport_records = [request.transport_record for request in requests]
        transport_record_json_by_request_id = {
            request.request.request_id: canonical_json(
                request.transport_record
            )
            for request in requests
        }
        request_payload = jsonl_dump_bytes(transport_records)
        if artifact_uri is None:
            artifact_rel = (
                f"{job.name}/{job.provider.provider_name}/{job.provider.model}/"
                f"{job.endpoint_kind}/{batch_id}/requests.jsonl"
            )
            artifact_uri = catalog.write_artifact(artifact_rel, request_payload)
        submission_or_failure = self._submit_to_provider(
            adapter, job, artifact_uri, request_payload, transport_records
        )
        if isinstance(submission_or_failure, SubmitFailure):
            event_at = utc_now_iso()
            created_at = event_at
            batch_status = (
                STATUS_SUBMIT_FAILED_RETRYABLE
                if submission_or_failure.retryable
                else STATUS_SUBMIT_FAILED
            )
            batch_row = {
                "event_at": event_at,
                "created_at": created_at,
                "job_name": job.name,
                "batch_id": batch_id,
                "provider_batch_id": None,
                "provider": job.provider.provider_name,
                "model": job.provider.model,
                "endpoint_kind": job.endpoint_kind,
                "status": batch_status,
                "request_count": len(requests),
                "artifact_uri": artifact_uri,
                "raw_json": canonical_json(submission_or_failure.raw_payload),
                "submit_attempts": submit_attempts,
            }
            if request_rows is not None:
                batch_row["created_at"] = request_rows[0].get("created_at", batch_row["created_at"])
            catalog.append_manifest(MANIFEST_BATCHES, [batch_row])
            if request_rows is None:
                request_rows = [
                    {
                        "event_at": event_at,
                        "created_at": batch_row["created_at"],
                        "job_name": job.name,
                        "request_id": request.request.request_id,
                        "batch_id": batch_id,
                        "custom_id": request.request.custom_id,
                        "cache_key": request.request.cache_key,
                        "provider": job.provider.provider_name,
                        "model": job.provider.model,
                        "endpoint_kind": job.endpoint_kind,
                        "row_key_json": canonical_json(request.request.row_key),
                        "payload_json": request.request.payload_json,
                        "transport_record_json": transport_record_json_by_request_id[
                            request.request.request_id
                        ],
                        "prompt_version": request.request.prompt_version,
                        "status": batch_status,
                    }
                    for request in requests
                ]
            else:
                request_rows = [
                    {
                        **row,
                        "event_at": event_at,
                        "transport_record_json": row.get("transport_record_json")
                        or transport_record_json_by_request_id.get(row["request_id"]),
                        "status": batch_status,
                    }
                    for row in request_rows
                ]
            catalog.append_manifest(MANIFEST_REQUESTS, request_rows)
            if not submission_or_failure.retryable:
                self._append_submit_failure_results(
                    catalog,
                    job,
                    batch_id,
                    request_rows,
                    error_code=submission_or_failure.error_code,
                    raw_payload=submission_or_failure.raw_payload,
                    event_at=event_at,
                )
            return SubmitFailure(
                batch_id=batch_id,
                retryable=submission_or_failure.retryable,
                error_code=submission_or_failure.error_code,
                raw_payload=submission_or_failure.raw_payload,
            )
        batch_row = {
            "event_at": utc_now_iso(),
            "created_at": utc_now_iso(),
            "job_name": job.name,
            "batch_id": batch_id,
            "provider_batch_id": submission_or_failure.provider_batch_id,
            "provider": job.provider.provider_name,
            "model": job.provider.model,
            "endpoint_kind": job.endpoint_kind,
            "status": submission_or_failure.status,
            "request_count": len(requests),
            "artifact_uri": artifact_uri,
            "raw_json": canonical_json(submission_or_failure.raw_payload),
            "submit_attempts": submit_attempts,
        }
        if request_rows is not None:
            batch_row["created_at"] = request_rows[0].get("created_at", batch_row["created_at"])
        catalog.append_manifest(MANIFEST_BATCHES, [batch_row])
        if request_rows is None:
            request_rows = [
                {
                    "event_at": utc_now_iso(),
                    "created_at": batch_row["created_at"],
                    "job_name": job.name,
                    "request_id": request.request.request_id,
                    "batch_id": batch_id,
                    "custom_id": request.request.custom_id,
                    "cache_key": request.request.cache_key,
                    "provider": job.provider.provider_name,
                    "model": job.provider.model,
                    "endpoint_kind": job.endpoint_kind,
                    "row_key_json": canonical_json(request.request.row_key),
                    "payload_json": request.request.payload_json,
                    "transport_record_json": transport_record_json_by_request_id[
                        request.request.request_id
                    ],
                    "prompt_version": request.request.prompt_version,
                    "status": STATUS_SUBMITTED,
                }
                for request in requests
            ]
        else:
            request_rows = [
                {
                    **row,
                    "event_at": utc_now_iso(),
                    "transport_record_json": row.get("transport_record_json")
                    or transport_record_json_by_request_id.get(row["request_id"]),
                    "status": STATUS_SUBMITTED,
                }
                for row in request_rows
            ]
        catalog.append_manifest(MANIFEST_REQUESTS, request_rows)
        return PendingState(
            batch_id=batch_id,
            provider_batch_id=submission_or_failure.provider_batch_id,
            status=submission_or_failure.status,
        )

    def _completed_results(
        self, catalog: ParquetCatalog, job_name: str
    ) -> dict[str, dict[str, Any]]:
        results_df = catalog.latest_results(job_name)
        if not len(results_df):
            return {}
        return {row["cache_key"]: row for row in results_df.iter_rows(named=True)}

    def _pending_requests(self, catalog: ParquetCatalog, job_name: str) -> dict[str, PendingState]:
        batches_df = catalog.latest_batches(job_name)
        requests_df = catalog.latest_requests(job_name)
        if not len(batches_df) or not len(requests_df):
            return {}
        batch_statuses = {
            row["batch_id"]: row
            for row in batches_df.iter_rows(named=True)
            if row["status"] in ACTIVE_BATCH_STATUSES
            or (
                row["status"] in TERMINAL_BATCH_STATUSES
                and not row.get("results_ingested_at")
            )
        }
        pending: dict[str, PendingState] = {}
        for row in requests_df.iter_rows(named=True):
            batch = batch_statuses.get(row["batch_id"])
            if batch is None:
                continue
            pending[row["cache_key"]] = PendingState(
                batch_id=row["batch_id"],
                provider_batch_id=batch["provider_batch_id"],
                status=batch["status"],
            )
        return pending

    def _materialize(
        self,
        job: Job,
        adapter: ProviderAdapter,
        input_df: pl.DataFrame,
        requests: list[RowRequest],
        row_states: dict[str, dict[str, Any]],
        *,
        metadata_columns: Sequence[str],
    ) -> pl.DataFrame:
        output_columns = tuple(self._empty_output(job).keys())
        output_rows: list[dict[str, Any]] = []
        for request in requests:
            state = row_states.get(request.cache_key)
            if state is None:
                state = self._pending_row_state(
                    job,
                    adapter,
                    request,
                    PendingState(batch_id="", provider_batch_id="", status=STATUS_PENDING),
                )
            output_row = dict(request.row_key)
            output_row.update({column: state.get(column) for column in output_columns})
            output_row.update({column: state.get(column) for column in metadata_columns})
            output_rows.append(output_row)

        output_df = pl.DataFrame(output_rows)
        joined = input_df.join(output_df, on=list(job.key_cols), how="left")
        return joined.select([*input_df.columns, *output_columns, *metadata_columns])

    def _completed_row_state(
        self,
        job: Job,
        adapter: ProviderAdapter,
        request: RowRequest,
        result_row: dict[str, Any],
        *,
        cached: bool,
    ) -> dict[str, Any]:
        payload = orjson.loads(result_row["parsed_json"]) if result_row.get("parsed_json") else {}
        if isinstance(job, EmbeddingJob):
            output = {
                "embedding": payload.get("embedding"),
                "embedding_dim": payload.get("embedding_dim"),
            }
        else:
            try:
                model = job.output_model.model_validate(payload)
            except ValidationError:
                return self._failed_row_state(
                    job,
                    adapter,
                    request,
                    {
                        **result_row,
                        "error_code": "schema_validation_error",
                    },
                )
            output = self._structured_output_for_result(job, model.model_dump(mode="json"))
        return {
            **output,
            **self._request_raw_metadata(job, adapter, request),
            "llm_batch_py_status": STATUS_CACHED if cached else STATUS_COMPLETED,
            "llm_batch_py_batch_id": result_row.get("batch_id"),
            "llm_batch_py_provider": job.provider.provider_name,
            "llm_batch_py_model": job.provider.model,
            "llm_batch_py_input_tokens": result_row.get("input_tokens"),
            "llm_batch_py_output_tokens": result_row.get("output_tokens"),
            "llm_batch_py_output_raw_json": result_row.get("raw_json"),
            "llm_batch_py_output_raw_text": result_row.get("raw_output_text"),
            "llm_batch_py_error_code": None,
            "llm_batch_py_result_cached": cached,
            "llm_batch_py_cached": cached,
            "llm_batch_py_updated_at": result_row.get("event_at"),
        }

    def _pending_row_state(
        self,
        job: Job,
        adapter: ProviderAdapter,
        request: RowRequest,
        pending: PendingState,
    ) -> dict[str, Any]:
        return {
            **self._empty_output(job),
            **self._request_raw_metadata(job, adapter, request),
            "llm_batch_py_status": pending.status,
            "llm_batch_py_batch_id": pending.batch_id,
            "llm_batch_py_provider": job.provider.provider_name,
            "llm_batch_py_model": job.provider.model,
            "llm_batch_py_input_tokens": None,
            "llm_batch_py_output_tokens": None,
            "llm_batch_py_output_raw_json": None,
            "llm_batch_py_output_raw_text": None,
            "llm_batch_py_error_code": None,
            "llm_batch_py_result_cached": False,
            "llm_batch_py_cached": False,
            "llm_batch_py_updated_at": utc_now_iso(),
        }

    def _failed_row_state(
        self,
        job: Job,
        adapter: ProviderAdapter,
        request: RowRequest,
        result_row: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            **self._request_raw_metadata(job, adapter, request),
            "llm_batch_py_status": STATUS_FAILED,
            "llm_batch_py_batch_id": result_row.get("batch_id"),
            "llm_batch_py_provider": result_row.get("provider"),
            "llm_batch_py_model": result_row.get("model"),
            "llm_batch_py_input_tokens": None,
            "llm_batch_py_output_tokens": None,
            "llm_batch_py_output_raw_json": result_row.get("raw_json"),
            "llm_batch_py_output_raw_text": None,
            "llm_batch_py_error_code": result_row.get("error_code"),
            "llm_batch_py_error_raw_json": result_row.get("raw_json"),
            "llm_batch_py_result_cached": False,
            "llm_batch_py_cached": False,
            "llm_batch_py_updated_at": result_row.get("event_at", utc_now_iso()),
        }

    def _request_raw_metadata(
        self,
        job: Job,
        adapter: ProviderAdapter,
        request: RowRequest,
    ) -> dict[str, str]:
        transport_record = dict(adapter.render_transport_record(job, request))
        transport_record.pop("custom_id", None)
        return {
            "llm_batch_py_input_raw_json": request.payload_json,
            "llm_batch_py_request_raw_json": canonical_json(transport_record),
        }

    def _empty_output(self, job: Job) -> dict[str, Any]:
        if isinstance(job, EmbeddingJob):
            return {"embedding": None, "embedding_dim": None}
        return {
            result_field_name: None
            for result_field_name in structured_output_result_column_map(job).values()
        }

    def _structured_output_for_result(
        self,
        job: Job,
        output: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(job, EmbeddingJob):
            return output
        column_map = structured_output_result_column_map(job)
        return {
            column_map[field_name]: field_value for field_name, field_value in output.items()
        }

    def _choose_batch_size(
        self,
        job: Job,
        adapter: ProviderAdapter,
    ) -> int:
        if job.batch.batch_size is not None:
            return min(job.batch.batch_size, adapter.request_cap)
        return max(1, adapter.request_cap)

    def _chunk_prepared_requests(
        self,
        prepared_requests: list[PreparedRequest],
        *,
        max_requests: int,
        byte_cap: int,
    ) -> tuple[list[list[PreparedRequest]], list[PreparedRequest]]:
        chunks: list[list[PreparedRequest]] = []
        oversized: list[PreparedRequest] = []
        current_chunk: list[PreparedRequest] = []
        current_bytes = 0

        for prepared_request in prepared_requests:
            record_bytes = self._transport_record_bytes(prepared_request.transport_record)
            if record_bytes > byte_cap:
                oversized.append(prepared_request)
                continue
            if current_chunk and (
                len(current_chunk) >= max_requests or current_bytes + record_bytes > byte_cap
            ):
                chunks.append(current_chunk)
                current_chunk = []
                current_bytes = 0
            current_chunk.append(prepared_request)
            current_bytes += record_bytes

        if current_chunk:
            chunks.append(current_chunk)
        return chunks, oversized

    def _transport_record_bytes(self, transport_record: dict[str, Any]) -> int:
        return len(orjson.dumps(transport_record)) + 1

    def _persist_local_failures(
        self,
        catalog: ParquetCatalog,
        job: Job,
        prepared_requests: list[PreparedRequest],
        *,
        error_code: str,
        raw_payload: dict[str, Any],
    ) -> None:
        if not prepared_requests:
            return
        event_at = utc_now_iso()
        catalog.append_manifest(
            MANIFEST_RESULTS,
            [
                {
                    "event_at": event_at,
                    "job_name": job.name,
                    "batch_id": None,
                    "request_id": prepared_request.request.request_id,
                    "custom_id": prepared_request.request.custom_id,
                    "cache_key": prepared_request.request.cache_key,
                    "provider": job.provider.provider_name,
                    "model": job.provider.model,
                    "endpoint_kind": job.endpoint_kind,
                    "status": STATUS_FAILED,
                    "error_code": error_code,
                    "row_key_json": canonical_json(prepared_request.request.row_key),
                    "parsed_json": None,
                    "raw_json": canonical_json(raw_payload),
                    "raw_output_text": None,
                    "input_tokens": None,
                    "output_tokens": None,
                }
                for prepared_request in prepared_requests
            ],
        )

    def _estimate_cost(
        self,
        job: Job,
        input_tokens: int,
        output_tokens: int,
        *,
        input_token_estimation: ResolvedInputTokenEstimation,
    ) -> float | None:
        if input_token_estimation == "skip" and input_tokens == 0 and output_tokens > 0:
            return None
        pricing = job.pricing
        if pricing is None:
            return 0.0
        return (input_tokens / 1_000_000 * pricing.input_per_million) + (
            output_tokens / 1_000_000 * pricing.output_per_million
        )

    def _resolve_input_token_estimation(
        self,
        job: Job,
        *,
        dry_run: bool,
    ) -> ResolvedInputTokenEstimation:
        if isinstance(job, EmbeddingJob):
            return "exact"
        if not isinstance(job.provider, AnthropicConfig):
            return "exact"
        if job.provider.input_token_estimation == "exact":
            return "exact"
        if job.provider.input_token_estimation == "skip":
            return "skip"
        return "exact" if dry_run else "skip"

    def _is_terminal_success(self, result_row: dict[str, Any]) -> bool:
        return (
            result_row["status"] == STATUS_COMPLETED and result_row.get("parsed_json") is not None
        )

    def _should_hold_failure(
        self, result_row: dict[str, Any], attempts: int, max_retries: int
    ) -> bool:
        if result_row["status"] != STATUS_FAILED:
            return False
        if attempts > max_retries:
            return True
        return result_row.get("error_code") not in RETRYABLE_ERROR_CODES

    def _provider_request_attempts(self, catalog: ParquetCatalog, job_name: str) -> list[str]:
        requests_df = catalog.read_manifest(MANIFEST_REQUESTS)
        if not len(requests_df):
            return []
        return requests_df.filter(
            (pl.col("job_name") == job_name) & (pl.col("status") == STATUS_SUBMITTED)
        )["cache_key"].to_list()

    def _submit_to_provider(
        self,
        adapter: ProviderAdapter,
        job: Job,
        artifact_uri: str,
        request_payload: bytes,
        transport_records: list[dict[str, Any]],
    ) -> Any:
        last_failure: SubmitFailure | None = None
        for attempt in range(INLINE_SUBMIT_RETRY_ATTEMPTS + 1):
            try:
                return adapter.submit_batch(job, artifact_uri, request_payload, transport_records)
            except Exception as exc:
                failure = self._classify_submit_exception(exc)
                last_failure = failure
                if not failure.retryable or attempt >= INLINE_SUBMIT_RETRY_ATTEMPTS:
                    return failure
        return last_failure

    def _classify_submit_exception(self, exc: Exception) -> SubmitFailure:
        status_code = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        class_name = exc.__class__.__name__.lower()
        message = str(exc)
        message_lower = message.lower()

        if isinstance(exc, TimeoutError) or "timeout" in class_name or "timeout" in message_lower:
            return SubmitFailure(
                batch_id=None,
                retryable=True,
                error_code="submit_timeout",
                raw_payload=self._submit_error_payload(exc, status_code),
            )
        if (
            isinstance(exc, ConnectionError)
            or "connection" in class_name
            or "connect" in message_lower
        ):
            return SubmitFailure(
                batch_id=None,
                retryable=True,
                error_code="submit_network_error",
                raw_payload=self._submit_error_payload(exc, status_code),
            )
        if status_code == 429 or "ratelimit" in class_name or "rate limit" in message_lower:
            return SubmitFailure(
                batch_id=None,
                retryable=True,
                error_code="submit_rate_limit_error",
                raw_payload=self._submit_error_payload(exc, status_code),
            )
        if isinstance(status_code, int) and status_code >= 500:
            return SubmitFailure(
                batch_id=None,
                retryable=True,
                error_code="submit_server_error",
                raw_payload=self._submit_error_payload(exc, status_code),
            )
        if (
            "authentication" in class_name
            or "permission" in class_name
            or status_code in {400, 401, 403, 404, 422}
        ):
            return SubmitFailure(
                batch_id=None,
                retryable=False,
                error_code="submit_invalid_request",
                raw_payload=self._submit_error_payload(exc, status_code),
            )
        return SubmitFailure(
            batch_id=None,
            retryable=False,
            error_code="submit_error",
            raw_payload=self._submit_error_payload(exc, status_code),
        )

    def _submit_error_payload(self, exc: Exception, status_code: int | None) -> dict[str, Any]:
        return {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "status_code": status_code,
        }

    def _is_complete_batch_result_set(
        self,
        request_rows: list[dict[str, Any]],
        fetched: list[ProviderResult],
    ) -> bool:
        expected_custom_ids = {row["custom_id"] for row in request_rows}
        fetched_custom_ids = [item.custom_id for item in fetched]
        return (
            len(fetched_custom_ids) == len(request_rows)
            and len(set(fetched_custom_ids)) == len(fetched_custom_ids)
            and set(fetched_custom_ids) == expected_custom_ids
        )

    def _result_rows_for_batch(
        self,
        job: Job,
        *,
        batch_id: str,
        request_lookup: dict[tuple[str, str], dict[str, Any]],
        fetched: list[ProviderResult],
    ) -> list[dict[str, Any]]:
        result_rows: list[dict[str, Any]] = []
        for item in fetched:
            request_row = request_lookup.get((batch_id, item.custom_id))
            if request_row is None:
                continue
            result_rows.append(
                {
                    "event_at": utc_now_iso(),
                    "job_name": job.name,
                    "batch_id": batch_id,
                    "request_id": request_row["request_id"],
                    "custom_id": item.custom_id,
                    "cache_key": request_row["cache_key"],
                    "provider": job.provider.provider_name,
                    "model": job.provider.model,
                    "endpoint_kind": job.endpoint_kind,
                    "status": item.status,
                    "error_code": item.error_code,
                    "row_key_json": request_row["row_key_json"],
                    "parsed_json": canonical_json(item.parsed_output)
                    if item.parsed_output is not None
                    else None,
                    "raw_json": canonical_json(item.raw_payload),
                    "raw_output_text": item.raw_output_text,
                    "input_tokens": item.input_tokens,
                    "output_tokens": item.output_tokens,
                }
            )
        return result_rows

    def _retryable_submit_batches(
        self, catalog: ParquetCatalog, job_name: str
    ) -> list[dict[str, Any]]:
        batches_df = catalog.latest_batches(job_name)
        requests_df = catalog.latest_requests(job_name)
        if not len(batches_df) or not len(requests_df):
            return []
        retryable_batches = batches_df.filter(pl.col("status") == STATUS_SUBMIT_FAILED_RETRYABLE)
        if not len(retryable_batches):
            return []
        request_rows = requests_df.filter(pl.col("status") == STATUS_SUBMIT_FAILED_RETRYABLE)
        if not len(request_rows):
            return []
        rows_by_batch: dict[str, list[dict[str, Any]]] = {}
        for row in request_rows.iter_rows(named=True):
            rows_by_batch.setdefault(row["batch_id"], []).append(row)
        ordered = sorted(
            retryable_batches.iter_rows(named=True),
            key=lambda row: (row.get("created_at") or row["event_at"], row["event_at"]),
        )
        return [
            {
                "batch": batch,
                "request_rows": rows_by_batch.get(batch["batch_id"], []),
            }
            for batch in ordered
            if rows_by_batch.get(batch["batch_id"])
        ]

    def _recover_retryable_submit_batches(
        self,
        catalog: ParquetCatalog,
        job: Job,
        adapter: ProviderAdapter,
        requests: list[RowRequest],
        batches: list[dict[str, Any]],
        *,
        dry_run: bool,
    ) -> dict[str, Any]:
        request_by_cache = {request.cache_key: request for request in requests}
        row_states: dict[str, dict[str, Any]] = {}
        inflight_rows = 0
        failed_rows = 0
        submitted_batches = 0
        submitted_rows = 0

        for item in batches:
            batch = item["batch"]
            request_rows = item["request_rows"]
            current_requests = [
                request_by_cache[row["cache_key"]]
                for row in request_rows
                if row["cache_key"] in request_by_cache
            ]
            if len(current_requests) != len(request_rows):
                continue
            if dry_run:
                pending = PendingState(
                    batch_id=batch["batch_id"],
                    provider_batch_id=None,
                    status=batch["status"],
                )
                for request in current_requests:
                    row_states[request.cache_key] = self._pending_row_state(
                        job,
                        adapter,
                        request,
                        pending,
                    )
                    inflight_rows += 1
                continue

            attempts = int(batch.get("submit_attempts") or 1)
            if attempts > job.batch.max_retries:
                self._mark_submit_batch_failed(
                    catalog,
                    job,
                    batch,
                    request_rows,
                    error_code="submit_retries_exhausted",
                )
                for request in current_requests:
                    row_states[request.cache_key] = self._failed_row_state(
                        job,
                        adapter,
                        request,
                        {
                            "batch_id": batch["batch_id"],
                            "provider": job.provider.provider_name,
                            "model": job.provider.model,
                            "error_code": "submit_retries_exhausted",
                            "raw_json": canonical_json(
                                {
                                    "type": "SubmitRetriesExhausted",
                                    "message": "submit retries exhausted",
                                }
                            ),
                            "event_at": utc_now_iso(),
                        },
                    )
                    failed_rows += 1
                continue

            transport_records = self._load_transport_records(catalog, batch["artifact_uri"])
            prepared_requests = [
                PreparedRequest(
                    request=current_request,
                    transport_record=transport_record,
                )
                for current_request, transport_record in zip(
                    current_requests, transport_records, strict=False
                )
            ]
            submission = self._submit_batch(
                catalog,
                job,
                adapter,
                prepared_requests,
                batch_id=batch["batch_id"],
                artifact_uri=batch["artifact_uri"],
                request_rows=request_rows,
                submit_attempts=attempts + 1,
            )
            if isinstance(submission, SubmitFailure):
                if submission.retryable:
                    pending = PendingState(
                        batch_id=submission.batch_id or batch["batch_id"],
                        provider_batch_id=None,
                        status=STATUS_SUBMIT_FAILED_RETRYABLE,
                    )
                    for request in current_requests:
                        row_states[request.cache_key] = self._pending_row_state(
                            job,
                            adapter,
                            request,
                            pending,
                        )
                        inflight_rows += 1
                    return {
                        "row_states": row_states,
                        "inflight_rows": inflight_rows,
                        "failed_rows": failed_rows,
                        "submitted_batches": submitted_batches,
                        "submitted_rows": submitted_rows,
                        "blocked": True,
                    }
                for request in current_requests:
                    row_states[request.cache_key] = self._failed_row_state(
                        job,
                        adapter,
                        request,
                        {
                            "batch_id": submission.batch_id or batch["batch_id"],
                            "provider": job.provider.provider_name,
                            "model": job.provider.model,
                            "error_code": submission.error_code,
                            "raw_json": canonical_json(submission.raw_payload),
                            "event_at": utc_now_iso(),
                        },
                    )
                    failed_rows += 1
                continue

            submitted_batches += 1
            submitted_rows += len(current_requests)
            for request in current_requests:
                row_states[request.cache_key] = self._pending_row_state(
                    job,
                    adapter,
                    request,
                    submission,
                )

        return {
            "row_states": row_states,
            "inflight_rows": inflight_rows,
            "failed_rows": failed_rows,
            "submitted_batches": submitted_batches,
            "submitted_rows": submitted_rows,
            "blocked": False,
        }

    def _submit_new_batch(
        self,
        catalog: ParquetCatalog,
        job: Job,
        adapter: ProviderAdapter,
        request_group: list[PreparedRequest],
    ) -> dict[str, Any]:
        submission = self._submit_batch(catalog, job, adapter, request_group)
        if not isinstance(submission, SubmitFailure):
            return {
                "pending_state": submission,
                "row_states": {},
                "inflight_rows": 0,
                "failed_rows": 0,
                "blocked": False,
            }
        row_states: dict[str, dict[str, Any]] = {}
        if submission.retryable:
            pending = PendingState(
                batch_id=submission.batch_id or "",
                provider_batch_id=None,
                status=STATUS_SUBMIT_FAILED_RETRYABLE,
            )
            for prepared_request in request_group:
                row_states[prepared_request.request.cache_key] = self._pending_row_state(
                    job,
                    adapter,
                    prepared_request.request,
                    pending,
                )
            return {
                "pending_state": None,
                "row_states": row_states,
                "inflight_rows": len(request_group),
                "failed_rows": 0,
                "blocked": True,
            }

        failed_event_at = utc_now_iso()
        for prepared_request in request_group:
            row_states[prepared_request.request.cache_key] = self._failed_row_state(
                job,
                adapter,
                prepared_request.request,
                {
                    "batch_id": submission.batch_id or "",
                    "provider": job.provider.provider_name,
                    "model": job.provider.model,
                    "error_code": submission.error_code,
                    "raw_json": canonical_json(submission.raw_payload),
                    "event_at": failed_event_at,
                },
            )
        return {
            "pending_state": None,
            "row_states": row_states,
            "inflight_rows": 0,
            "failed_rows": len(request_group),
            "blocked": True,
        }

    def _load_transport_records(
        self, catalog: ParquetCatalog, artifact_uri: str
    ) -> list[dict[str, Any]]:
        payload = catalog.read_text(artifact_uri)
        return [orjson.loads(line) for line in payload.splitlines() if line.strip()]

    def _mark_submit_batch_failed(
        self,
        catalog: ParquetCatalog,
        job: Job,
        batch: dict[str, Any],
        request_rows: list[dict[str, Any]],
        *,
        error_code: str,
        raw_payload: dict[str, Any] | None = None,
    ) -> None:
        event_at = utc_now_iso()
        payload = raw_payload or {"type": "SubmitFailure", "message": error_code}
        catalog.append_manifest(
            MANIFEST_BATCHES,
            [
                {
                    **batch,
                    "event_at": event_at,
                    "status": STATUS_SUBMIT_FAILED,
                    "raw_json": canonical_json(payload),
                }
            ],
        )
        catalog.append_manifest(
            MANIFEST_RESULTS,
            self._submit_failure_result_rows(
                job,
                batch["batch_id"],
                request_rows,
                error_code=error_code,
                raw_payload=payload,
                event_at=event_at,
            ),
        )

    def _append_submit_failure_results(
        self,
        catalog: ParquetCatalog,
        job: Job,
        batch_id: str,
        request_rows: list[dict[str, Any]],
        *,
        error_code: str,
        raw_payload: dict[str, Any],
        event_at: str,
    ) -> None:
        catalog.append_manifest(
            MANIFEST_RESULTS,
            self._submit_failure_result_rows(
                job,
                batch_id,
                request_rows,
                error_code=error_code,
                raw_payload=raw_payload,
                event_at=event_at,
            ),
        )

    def _submit_failure_result_rows(
        self,
        job: Job,
        batch_id: str,
        request_rows: list[dict[str, Any]],
        *,
        error_code: str,
        raw_payload: dict[str, Any],
        event_at: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "event_at": event_at,
                "job_name": job.name,
                "batch_id": batch_id,
                "request_id": row["request_id"],
                "custom_id": row["custom_id"],
                "cache_key": row["cache_key"],
                "provider": job.provider.provider_name,
                "model": job.provider.model,
                "endpoint_kind": job.endpoint_kind,
                "status": STATUS_FAILED,
                "error_code": error_code,
                "row_key_json": row["row_key_json"],
                "parsed_json": None,
                "raw_json": canonical_json(raw_payload),
                "raw_output_text": None,
                "input_tokens": None,
                "output_tokens": None,
            }
            for row in request_rows
        ]

    def _iter_stream_chunks(
        self,
        job: Job,
        *,
        input_batch_rows: int | None,
        order_by: Sequence[str] | None,
        input_batches: Iterable[pl.DataFrame] | None,
    ) -> Iterator[pl.DataFrame]:
        if input_batches is not None:
            yield from self._iter_input_batches(input_batches)
            return
        if input_batch_rows is None:
            raise ValueError("input_batch_rows is required when input_batches is not provided")
        yield from self._iter_input_chunks(
            job,
            input_batch_rows=input_batch_rows,
            order_by=order_by,
        )

    def _validated_stream_chunks(
        self,
        job: Job,
        *,
        input_batch_rows: int | None,
        order_by: Sequence[str] | None,
        input_batches: Iterable[pl.DataFrame] | None,
    ) -> Iterator[pl.DataFrame]:
        if input_batches is not None:
            if isinstance(input_batches, Sequence):
                for _ in self._yield_validated_chunks(job, self._iter_input_batches(input_batches)):
                    pass
            return self._yield_validated_chunks(job, self._iter_input_batches(input_batches))
        if isinstance(job.input_df, pl.DataFrame):
            validate_job_input_columns(job, job.input_df.columns)
            return self._yield_validated_chunks(
                job,
                self._iter_input_chunks(
                    job,
                    input_batch_rows=input_batch_rows,
                    order_by=order_by,
                ),
            )
        if input_batch_rows is None:
            raise ValueError("input_batch_rows is required when input_batches is not provided")
        validate_job_input_columns(job, job.input_df.collect_schema().names())
        return self._yield_validated_chunks(
            job,
            self._iter_input_chunks(
                job,
                input_batch_rows=input_batch_rows,
                order_by=order_by,
            ),
        )

    def _iter_input_batches(self, input_batches: Iterable[pl.DataFrame]) -> Iterator[pl.DataFrame]:
        for chunk_df in input_batches:
            if not isinstance(chunk_df, pl.DataFrame):
                raise TypeError("input_batches must yield polars DataFrame values")
            yield chunk_df

    def _yield_validated_chunks(
        self,
        job: Job,
        chunk_iter: Iterable[pl.DataFrame],
    ) -> Iterator[pl.DataFrame]:
        seen_row_keys: set[str] = set()
        for chunk_df in chunk_iter:
            validate_job_input_columns(job, chunk_df.columns)
            self._validate_stream_chunk_keys(job, chunk_df, seen_row_keys)
            yield chunk_df

    def _iter_input_chunks(
        self,
        job: Job,
        *,
        input_batch_rows: int,
        order_by: Sequence[str] | None,
    ) -> Iterator[pl.DataFrame]:
        if isinstance(job.input_df, pl.DataFrame):
            sort_columns = list(order_by) if order_by is not None else list(job.key_cols)
            missing_cols = [column for column in sort_columns if column not in job.input_df.columns]
            if missing_cols:
                raise ValueError(f"Missing order_by columns: {missing_cols}")
            frame = job.input_df.sort(sort_columns) if sort_columns else job.input_df
            for offset in range(0, frame.height, input_batch_rows):
                yield frame.slice(offset, input_batch_rows)
            return

        sort_columns = list(order_by) if order_by is not None else list(job.key_cols)
        lazy_columns = job.input_df.collect_schema().names()
        missing_cols = [column for column in sort_columns if column not in lazy_columns]
        if missing_cols:
            raise ValueError(f"Missing order_by columns: {missing_cols}")

        sorted_input = job.input_df.sort(sort_columns)
        offset = 0
        while True:
            chunk_df = sorted_input.slice(offset, input_batch_rows).collect()
            if not len(chunk_df):
                break
            yield chunk_df
            offset += input_batch_rows

    def _validate_stream_chunk_keys(
        self,
        job: Job,
        chunk_df: pl.DataFrame,
        seen_row_keys: set[str],
    ) -> None:
        for row in chunk_df.select(list(job.key_cols)).iter_rows(named=True):
            row_key_hash = stable_hash(row)
            if row_key_hash in seen_row_keys:
                raise ValueError(
                    "key_cols must uniquely identify each input row across streamed chunks"
                )
            seen_row_keys.add(row_key_hash)


def _chunked(values: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        return [values]
    return [values[index : index + size] for index in range(0, len(values), size)]
