from __future__ import annotations

from pathlib import Path

import orjson
import polars as pl
import pytest
from pydantic import BaseModel

from llm_batch_py import (
    BatchConfig,
    LockConfig,
    OpenAIConfig,
    PromptCacheConfig,
    ResultCacheStoreConfig,
    StructuredOutputJob,
    prompt_udf,
)
from llm_batch_py.catalog import MANIFEST_BATCHES, MANIFEST_REQUESTS, MANIFEST_RUNS, ParquetCatalog
from llm_batch_py.jobs import (
    LLM_BATCH_PY_RESULT_METADATA_COLUMN_ORDER,
    output_model_json_schema,
    validate_job_input,
)
from llm_batch_py.runner import Runner
from tests.fakes import FakeAdapter, FakeOutcome


def test_runner_submits_missing_rows_and_materializes_on_next_run(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()

    first = runner.run(structured_job)

    assert first["llm_batch_py_status"].to_list() == ["submitted", "submitted"]
    assert adapter.submissions == ["provider_1"]

    adapter.complete_all()
    second = runner.run(structured_job)

    assert second["label"].to_list() == ["ALPHA", "BETA"]
    assert second["llm_batch_py_status"].to_list() == ["cached", "cached"]
    assert second["llm_batch_py_input_tokens"].to_list() == [8, 8]
    assert second["llm_batch_py_output_tokens"].to_list() == [4, 4]
    assert second["llm_batch_py_input_raw_json"].null_count() == 0
    assert second["llm_batch_py_request_raw_json"].null_count() == 0
    assert second["llm_batch_py_output_raw_json"].null_count() == 0
    assert second["llm_batch_py_output_raw_text"].to_list() == [
        '{"label":"ALPHA"}',
        '{"label":"BETA"}',
    ]
    assert second["llm_batch_py_result_cached"].to_list() == [True, True]
    assert second["llm_batch_py_cached"].to_list() == [True, True]
    assert runner.last_summary is not None
    assert runner.last_summary.submitted_rows == 0
    assert runner.last_summary.result_cache_hits == 2
    assert runner.last_summary.cache_hits == 2


def test_runner_populates_raw_request_metadata_for_submitted_rows(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    result = Runner().run(structured_job)

    assert result["llm_batch_py_status"].to_list() == ["submitted", "submitted"]
    assert result["llm_batch_py_output_raw_json"].to_list() == [None, None]
    assert result["llm_batch_py_output_raw_text"].to_list() == [None, None]

    first_input = orjson.loads(result["llm_batch_py_input_raw_json"].to_list()[0])
    first_request = orjson.loads(result["llm_batch_py_request_raw_json"].to_list()[0])
    request_manifest = ParquetCatalog(
        structured_job.result_cache,
        structured_job.lock,
    ).read_manifest(MANIFEST_REQUESTS)

    assert first_input["messages"][0]["content"] == "Label alpha"
    assert first_request["payload"]["messages"][0]["content"] == "Label alpha"
    assert request_manifest["transport_record_json"].null_count() == 0


def test_runner_metadata_columns_can_limit_joined_metadata(monkeypatch, structured_job) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    result = Runner().run(
        structured_job,
        metadata_columns=["llm_batch_py_status"],
    )

    assert result.columns == ["id", "text", "label", "llm_batch_py_status"]
    assert result["llm_batch_py_status"].to_list() == ["submitted", "submitted"]


def test_runner_metadata_columns_can_exclude_all_metadata(monkeypatch, structured_job) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    result = Runner().run(structured_job, metadata_columns=[])

    assert result.columns == ["id", "text", "label"]


def test_runner_metadata_columns_fail_fast_on_unknown_names(structured_job) -> None:
    with pytest.raises(
        ValueError,
        match=r"Unknown llm_batch_py metadata columns: \['llm_batch_py_not_real'\]",
    ):
        Runner().run(structured_job, metadata_columns=["llm_batch_py_not_real"])


def test_runner_metadata_columns_deduplicate_and_preserve_requested_order(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    result = Runner().run(
        structured_job,
        metadata_columns=[
            "llm_batch_py_output_raw_text",
            "llm_batch_py_status",
            "llm_batch_py_output_raw_text",
        ],
    )

    assert result.columns == [
        "id",
        "text",
        "label",
        "llm_batch_py_output_raw_text",
        "llm_batch_py_status",
    ]


def test_runner_default_metadata_columns_remain_full(monkeypatch, structured_job) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    result = Runner().run(structured_job)

    assert [
        column
        for column in result.columns
        if column.startswith("llm_batch_py_")
    ] == list(LLM_BATCH_PY_RESULT_METADATA_COLUMN_ORDER)


def test_runner_run_stream_lazyframe_yields_chunked_results_and_aggregate_summary(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()
    job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "input_df": structured_job.input_df.lazy(),
        }
    )

    first_chunks = list(runner.run_stream(job, input_batch_rows=1))

    assert [chunk["llm_batch_py_status"].to_list() for chunk in first_chunks] == [
        ["submitted"],
        ["submitted"],
    ]
    assert adapter.submissions == ["provider_1", "provider_2"]
    assert runner.last_stream_summary is not None
    assert runner.last_stream_summary.chunk_count == 2
    assert runner.last_stream_summary.total_rows == 2
    assert runner.last_stream_summary.submitted_rows == 2
    assert runner.last_stream_summary.result_cache_hits == 0

    adapter.complete_all()
    second_chunks = list(runner.run_stream(job, input_batch_rows=1))

    assert [chunk["label"].to_list() for chunk in second_chunks] == [["ALPHA"], ["BETA"]]
    assert [chunk["llm_batch_py_status"].to_list() for chunk in second_chunks] == [
        ["cached"],
        ["cached"],
    ]
    assert runner.last_stream_summary is not None
    assert runner.last_stream_summary.chunk_count == 2
    assert runner.last_stream_summary.total_rows == 2
    assert runner.last_stream_summary.result_cache_hits == 2
    assert runner.last_stream_summary.submitted_rows == 0


def test_runner_run_stream_lazyframe_validates_while_streaming(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "input_df": structured_job.input_df.lazy(),
        }
    )
    seen_chunk_ids: list[list[int]] = []
    original_yield_validated_chunks = Runner._yield_validated_chunks

    def recording_yield_validated_chunks(self, job, chunk_iter):
        for chunk_df in original_yield_validated_chunks(self, job, chunk_iter):
            seen_chunk_ids.append(chunk_df["id"].to_list())
            yield chunk_df

    monkeypatch.setattr(Runner, "_yield_validated_chunks", recording_yield_validated_chunks)

    list(Runner().run_stream(job, input_batch_rows=1))

    assert seen_chunk_ids == [[1], [2]]


def test_runner_run_stream_accepts_iterable_batches_and_preserves_cache_behavior(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()
    input_batches = [
        pl.DataFrame({"id": [1], "text": ["alpha"]}),
        pl.DataFrame({"id": [2], "text": ["beta"]}),
    ]

    first_chunks = list(runner.run_stream(structured_job, input_batches=input_batches))

    assert [chunk["llm_batch_py_status"].to_list() for chunk in first_chunks] == [
        ["submitted"],
        ["submitted"],
    ]
    assert adapter.submissions == ["provider_1", "provider_2"]
    assert runner.last_stream_summary is not None
    assert runner.last_stream_summary.chunk_count == 2
    assert runner.last_stream_summary.submitted_rows == 2

    adapter.complete_all()
    second_chunks = list(runner.run_stream(structured_job, input_batches=input_batches))

    assert [chunk["label"].to_list() for chunk in second_chunks] == [["ALPHA"], ["BETA"]]
    assert [chunk["llm_batch_py_status"].to_list() for chunk in second_chunks] == [
        ["cached"],
        ["cached"],
    ]
    assert runner.last_stream_summary is not None
    assert runner.last_stream_summary.result_cache_hits == 2
    assert runner.last_stream_summary.submitted_rows == 0


def test_runner_run_stream_metadata_columns_can_limit_joined_metadata(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    chunks = list(
        Runner().run_stream(
            structured_job,
            input_batch_rows=1,
            metadata_columns=["llm_batch_py_status"],
        )
    )

    assert [chunk.columns for chunk in chunks] == [
        ["id", "text", "label", "llm_batch_py_status"],
        ["id", "text", "label", "llm_batch_py_status"],
    ]


def test_runner_run_stream_iterable_batches_remains_lazy(monkeypatch, structured_job) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()
    consumed: list[int] = []

    def batches():
        for row_id in [1, 2, 3]:
            consumed.append(row_id)
            yield pl.DataFrame({"id": [row_id], "text": [f"value-{row_id}"]})

    stream = runner.run_stream(structured_job, input_batches=batches())
    first_chunk = next(stream)

    assert consumed == [1]
    assert first_chunk["id"].to_list() == [1]


def test_runner_run_stream_releases_lock_and_records_runs_per_chunk(
    monkeypatch, structured_job, tmp_path
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "result_cache": ResultCacheStoreConfig(root_uri=str(tmp_path / "stream-catalog")),
        }
    )
    runner = Runner()

    stream = runner.run_stream(job, input_batch_rows=1)
    first_chunk = next(stream)

    assert first_chunk["llm_batch_py_status"].to_list() == ["submitted"]
    assert not Path(job.result_cache.root_uri, "locks", f"{job.name}.json").exists()

    remaining_chunks = list(stream)

    assert [chunk["llm_batch_py_status"].to_list() for chunk in remaining_chunks] == [["submitted"]]
    assert not Path(job.result_cache.root_uri, "locks", f"{job.name}.json").exists()

    runs = ParquetCatalog(job.result_cache, job.lock).read_manifest(MANIFEST_RUNS)

    assert runs.height == 2
    assert runs["run_id"].n_unique() == 2


def test_runner_run_stream_matches_run_output_for_cached_rows(
    monkeypatch, structured_job, tmp_path
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    stream_job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "input_df": pl.DataFrame({"id": [1, 2, 3], "text": ["alpha", "beta", "gamma"]}).lazy(),
            "result_cache": ResultCacheStoreConfig(root_uri=str(tmp_path / "stream-catalog")),
        }
    )
    stream_runner = Runner()
    list(stream_runner.run_stream(stream_job, input_batch_rows=2))
    adapter.complete_all()
    streamed = pl.concat(list(stream_runner.run_stream(stream_job, input_batch_rows=2))).sort("id")

    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    run_job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "input_df": pl.DataFrame({"id": [1, 2, 3], "text": ["alpha", "beta", "gamma"]}),
            "result_cache": ResultCacheStoreConfig(root_uri=str(tmp_path / "run-catalog")),
        }
    )
    run_runner = Runner()
    run_runner.run(run_job)
    adapter.complete_all()
    materialized = run_runner.run(run_job).sort("id")

    stable_columns = [
        "id",
        "text",
        "label",
        "llm_batch_py_status",
        "llm_batch_py_provider",
        "llm_batch_py_model",
        "llm_batch_py_input_tokens",
        "llm_batch_py_output_tokens",
        "llm_batch_py_input_raw_json",
        "llm_batch_py_request_raw_json",
        "llm_batch_py_output_raw_text",
        "llm_batch_py_error_code",
        "llm_batch_py_result_cached",
        "llm_batch_py_cached",
    ]

    assert streamed.select(stable_columns).equals(materialized.select(stable_columns))


def test_runner_run_stream_matches_run_output_for_selected_metadata(
    monkeypatch, structured_job, tmp_path
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    stream_job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "input_df": pl.DataFrame({"id": [1, 2], "text": ["alpha", "beta"]}).lazy(),
            "result_cache": ResultCacheStoreConfig(root_uri=str(tmp_path / "stream-selected")),
        }
    )
    stream_runner = Runner()
    list(stream_runner.run_stream(stream_job, input_batch_rows=1))
    adapter.complete_all()
    streamed = pl.concat(
        list(
            stream_runner.run_stream(
                stream_job,
                input_batch_rows=1,
                metadata_columns=["llm_batch_py_status", "llm_batch_py_output_raw_text"],
            )
        )
    ).sort("id")

    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    run_job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "input_df": pl.DataFrame({"id": [1, 2], "text": ["alpha", "beta"]}),
            "result_cache": ResultCacheStoreConfig(root_uri=str(tmp_path / "run-selected")),
        }
    )
    run_runner = Runner()
    run_runner.run(run_job)
    adapter.complete_all()
    materialized = run_runner.run(
        run_job,
        metadata_columns=["llm_batch_py_status", "llm_batch_py_output_raw_text"],
    ).sort("id")

    assert streamed.equals(materialized)


def test_runner_run_stream_dataframe_defaults_to_key_order(structured_job) -> None:
    job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "input_df": pl.DataFrame({"id": [2, 1], "text": ["beta", "alpha"]}),
        }
    )

    chunks = list(Runner()._iter_input_chunks(job, input_batch_rows=1, order_by=None))

    assert [chunk["id"].item() for chunk in chunks] == [1, 2]


def test_runner_splits_batches_to_respect_provider_byte_cap(monkeypatch, structured_job) -> None:
    adapter = FakeAdapter()
    runner = Runner()
    requests = runner._build_requests(structured_job, structured_job.input_df)
    prepared_requests = adapter.prepare_requests(structured_job, requests)
    record_sizes = [
        runner._transport_record_bytes(prepared_request.transport_record)
        for prepared_request in prepared_requests
    ]
    adapter.byte_cap = sum(record_sizes) - 1
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    runner.run(structured_job)

    assert adapter.submissions == ["provider_1", "provider_2"]


def test_runner_marks_oversized_requests_failed_without_submission(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    runner = Runner()
    requests = runner._build_requests(structured_job, structured_job.input_df)
    prepared_requests = adapter.prepare_requests(structured_job, requests)
    adapter.byte_cap = min(
        runner._transport_record_bytes(prepared_request.transport_record)
        for prepared_request in prepared_requests
    ) - 1
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    result = runner.run(structured_job)

    assert adapter.submissions == []
    assert result["llm_batch_py_status"].to_list() == ["failed", "failed"]
    assert result["llm_batch_py_error_code"].to_list() == ["request_too_large", "request_too_large"]


def test_runner_only_submits_incremental_rows(monkeypatch, structured_job) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()

    runner.run(structured_job)
    adapter.complete_all()
    runner.run(structured_job)
    assert adapter.submissions == ["provider_1"]

    changed_job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "input_df": pl.DataFrame({"id": [1, 2], "text": ["alpha", "gamma"]}),
        }
    )

    incremental = runner.run(changed_job)

    assert adapter.submissions == ["provider_1", "provider_2"]
    assert incremental["llm_batch_py_status"].to_list() == ["cached", "submitted"]
    assert incremental["llm_batch_py_result_cached"].to_list() == [True, False]
    assert incremental["label"].to_list() == ["ALPHA", None]

    missed = runner.run(
        structured_job.__class__(
            **{
                **structured_job.__dict__,
                "input_df": pl.DataFrame({"id": [1, 2], "text": ["delta", "epsilon"]}),
            }
        )
    )


    assert missed["llm_batch_py_status"].to_list() == ["submitted", "submitted"]
    assert missed["llm_batch_py_result_cached"].to_list() == [False, False]
    assert runner.last_summary is not None
    assert runner.last_summary.result_cache_hits == 0
    assert runner.last_summary.submitted_rows == 2


def test_runner_result_cache_partial_hit_and_miss(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()

    runner.run(structured_job)
    adapter.complete_all()
    runner.run(structured_job)

    changed_job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "input_df": pl.DataFrame({"id": [1, 2], "text": ["alpha", "gamma"]}),
        }
    )

    mixed = runner.run(changed_job)

    assert adapter.submissions == ["provider_1", "provider_2"]
    assert mixed["llm_batch_py_status"].to_list() == ["cached", "submitted"]
    assert mixed["llm_batch_py_result_cached"].to_list() == [True, False]
    assert mixed["label"].to_list() == ["ALPHA", None]
    assert runner.last_summary is not None
    assert runner.last_summary.result_cache_hits == 1
    assert runner.last_summary.submitted_rows == 1


def test_runner_submit_retry_budget_is_per_small_batch(monkeypatch, structured_job) -> None:
    adapter = FakeAdapter(submit_failures=[ConnectionError("network down")] * 6)
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "batch": BatchConfig(batch_size=1, max_retries=1),
        }
    )

    Runner().run(job)
    Runner().run(job)
    result = Runner().run(job)

    assert result["llm_batch_py_status"].to_list() == ["failed", "submitted"]
    assert result["llm_batch_py_error_code"].to_list() == ["submit_retries_exhausted", None]
    assert adapter.submissions == ["provider_1"]


def test_runner_reuses_terminal_batches_with_uningested_results(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    first_runner = Runner()
    first_runner.run(structured_job)
    adapter.complete_all()

    catalog = ParquetCatalog(structured_job.result_cache, structured_job.lock)
    batches = catalog.latest_batches(structured_job.name).to_dicts()
    assert len(batches) == 1
    assert batches[0]["status"] == "submitted"

    catalog.append_manifest(
        MANIFEST_BATCHES,
        [
            {
                **batches[0],
                "status": "completed",
            }
        ],
    )

    second_runner = Runner()
    result = second_runner.run(structured_job)

    assert adapter.submissions == ["provider_1"]
    assert result["label"].to_list() == ["ALPHA", "BETA"]
    assert result["llm_batch_py_status"].to_list() == ["cached", "cached"]
    assert second_runner.last_summary is not None
    assert second_runner.last_summary.submitted_rows == 0
    assert second_runner.last_summary.result_cache_hits == 2


def test_runner_retries_terminal_batch_ingestion_after_fetch_failure(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    initial_runner = Runner()
    initial_runner.run(structured_job)
    adapter.complete_all()

    original_fetch_results = adapter.fetch_results
    fetch_attempts = {"count": 0}

    def flaky_fetch_results(job, batch_snapshot, provider_batch_id):
        fetch_attempts["count"] += 1
        if fetch_attempts["count"] == 1:
            raise RuntimeError("transient fetch failure")
        return original_fetch_results(job, batch_snapshot, provider_batch_id)

    adapter.fetch_results = flaky_fetch_results

    failed_runner = Runner()
    try:
        failed_runner.run(structured_job)
    except RuntimeError as exc:
        assert str(exc) == "transient fetch failure"
    else:
        raise AssertionError("Expected transient fetch failure")

    recovered_runner = Runner()
    result = recovered_runner.run(structured_job)

    assert adapter.submissions == ["provider_1"]
    assert fetch_attempts["count"] == 2
    assert result["label"].to_list() == ["ALPHA", "BETA"]
    assert result["llm_batch_py_status"].to_list() == ["cached", "cached"]
    assert recovered_runner.last_summary is not None
    assert recovered_runner.last_summary.submitted_rows == 0
    assert recovered_runner.last_summary.result_cache_hits == 2


def test_runner_keeps_terminal_batch_recoverable_when_result_fetch_is_incomplete(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)

    initial_runner = Runner()
    initial_runner.run(structured_job)
    adapter.complete_all()

    original_fetch_results = adapter.fetch_results

    def partial_fetch_results(job, batch_snapshot, provider_batch_id):
        return original_fetch_results(job, batch_snapshot, provider_batch_id)[:1]

    adapter.fetch_results = partial_fetch_results

    partial_runner = Runner()
    partial = partial_runner.run(structured_job)

    assert adapter.submissions == ["provider_1"]
    assert partial["label"].to_list() == ["ALPHA", None]
    assert partial["llm_batch_py_result_cached"].to_list() == [True, False]

    latest_batch = ParquetCatalog(structured_job.result_cache, structured_job.lock).latest_batches(
        structured_job.name
    ).to_dicts()[0]
    assert latest_batch["status"] == "completed"
    assert latest_batch.get("results_ingested_at") is None

    adapter.fetch_results = original_fetch_results
    recovered = Runner().run(structured_job)

    assert adapter.submissions == ["provider_1"]
    assert recovered["label"].to_list() == ["ALPHA", "BETA"]
    assert recovered["llm_batch_py_status"].to_list() == ["cached", "cached"]


def test_runner_materializes_embeddings(monkeypatch, embedding_job) -> None:
    adapter = FakeAdapter(mode="embeddings")
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()

    runner.run(embedding_job)
    adapter.complete_all()
    result = runner.run(embedding_job)

    assert result["embedding_dim"].to_list() == [2, 2]
    assert result["llm_batch_py_status"].to_list() == ["cached", "cached"]
    assert result["llm_batch_py_input_raw_json"].null_count() == 0
    assert result["llm_batch_py_request_raw_json"].null_count() == 0
    assert result["llm_batch_py_output_raw_json"].null_count() == 0
    assert result["llm_batch_py_output_raw_text"].to_list() == [None, None]
    assert result["llm_batch_py_result_cached"].to_list() == [True, True]


def test_runner_surfaces_terminal_failures(monkeypatch, structured_job) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()

    runner.run(structured_job)
    custom_id = next(iter(adapter.batches.values())).requests[0]["custom_id"]
    adapter.complete_all(fail_custom_ids={custom_id})

    result = runner.run(structured_job)

    assert result["llm_batch_py_status"].to_list()[0] == "failed"
    assert result["llm_batch_py_error_code"].to_list()[0] == "invalid_request"


def test_runner_holds_non_retryable_failed_rows_instead_of_resubmitting(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()

    runner.run(structured_job)
    custom_ids = [request["custom_id"] for request in next(iter(adapter.batches.values())).requests]
    adapter.complete_all(
        outcomes={
            custom_ids[0]: FakeOutcome(kind="failed", error_code="invalid_request"),
            custom_ids[1]: FakeOutcome(kind="completed"),
        }
    )

    result = runner.run(structured_job)

    assert adapter.submissions == ["provider_1"]
    assert result["llm_batch_py_status"].to_list() == ["failed", "cached"]
    assert result["llm_batch_py_input_tokens"].to_list() == [None, 8]
    assert result["llm_batch_py_output_tokens"].to_list() == [None, 4]
    assert result["llm_batch_py_input_raw_json"].null_count() == 0
    assert result["llm_batch_py_request_raw_json"].null_count() == 0
    assert result["llm_batch_py_output_raw_json"].null_count() == 0
    assert result["llm_batch_py_output_raw_text"].to_list() == [None, '{"label":"BETA"}']
    assert result["llm_batch_py_error_code"].to_list() == ["invalid_request", None]
    assert runner.last_summary is not None
    assert runner.last_summary.failed_rows == 1
    assert runner.last_summary.result_cache_hits == 1
    assert runner.last_summary.submitted_rows == 0


def test_runner_metadata_columns_can_select_failure_metadata_only(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()

    runner.run(structured_job)
    custom_id = next(iter(adapter.batches.values())).requests[0]["custom_id"]
    adapter.complete_all(outcomes={custom_id: FakeOutcome(kind="malformed")})

    result = runner.run(
        structured_job,
        metadata_columns=[
            "llm_batch_py_status",
            "llm_batch_py_error_code",
            "llm_batch_py_error_raw_json",
        ],
    )

    assert result.columns == [
        "id",
        "text",
        "label",
        "llm_batch_py_status",
        "llm_batch_py_error_code",
        "llm_batch_py_error_raw_json",
    ]
    assert result["llm_batch_py_status"].to_list()[0] == "failed"
    assert result["llm_batch_py_error_code"].to_list()[0] == "schema_validation_error"
    assert result["llm_batch_py_error_raw_json"].to_list()[0] is not None


def test_runner_preserves_non_retryable_submit_failures_on_rerun(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter(submit_failures=[ValueError("bad request")])
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()

    first = runner.run(structured_job)
    second = runner.run(structured_job)

    assert first["llm_batch_py_status"].to_list() == ["failed", "failed"]
    assert second["llm_batch_py_status"].to_list() == ["failed", "failed"]
    assert second["llm_batch_py_error_code"].to_list() == ["submit_error", "submit_error"]
    assert second["llm_batch_py_input_raw_json"].null_count() == 0
    assert second["llm_batch_py_request_raw_json"].null_count() == 0
    assert second["llm_batch_py_output_raw_json"].null_count() == 0
    assert second["llm_batch_py_output_raw_text"].to_list() == [None, None]
    assert runner.last_summary is not None
    assert runner.last_summary.failed_rows == 2
    assert runner.last_summary.submitted_rows == 0


def test_runner_resubmits_retryable_failures_while_retry_budget_remains(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()

    runner.run(structured_job)
    custom_ids = [request["custom_id"] for request in next(iter(adapter.batches.values())).requests]
    adapter.complete_all(
        outcomes={custom_ids[0]: FakeOutcome(kind="failed", error_code="rate_limit_error")}
    )

    retried = runner.run(structured_job)

    assert adapter.submissions == ["provider_1", "provider_2"]
    assert retried["llm_batch_py_status"].to_list() == ["submitted", "cached"]
    assert retried["llm_batch_py_result_cached"].to_list() == [False, True]
    assert runner.last_summary is not None
    assert runner.last_summary.result_cache_hits == 1
    assert runner.last_summary.submitted_rows == 1


def test_runner_run_stream_rejects_duplicate_keys_across_chunks(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "input_df": pl.DataFrame({"id": [1, 2, 1], "text": ["alpha", "beta", "again"]}),
        }
    )

    with pytest.raises(
        ValueError, match="key_cols must uniquely identify each input row across streamed chunks"
    ):
        list(Runner().run_stream(job, input_batch_rows=2))
    assert adapter.submissions == []


def test_runner_run_stream_rejects_duplicate_keys_across_iterable_batches(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    input_batches = [
        pl.DataFrame({"id": [1], "text": ["alpha"]}),
        pl.DataFrame({"id": [1], "text": ["alpha-again"]}),
    ]

    with pytest.raises(
        ValueError, match="key_cols must uniquely identify each input row across streamed chunks"
    ):
        list(Runner().run_stream(structured_job, input_batches=input_batches))
    assert adapter.submissions == []


def test_runner_run_stream_rejects_invalid_input_source_configuration(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()

    with pytest.raises(ValueError, match="input_batch_rows is required"):
        list(runner.run_stream(structured_job))

    with pytest.raises(ValueError, match="Pass only one of input_batch_rows or input_batches"):
        list(
            runner.run_stream(
                structured_job,
                input_batch_rows=1,
                input_batches=[pl.DataFrame({"id": [1], "text": ["alpha"]})],
            )
        )


def test_runner_run_stream_empty_input_sets_zero_summary(monkeypatch, structured_job) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()
    job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "input_df": pl.DataFrame(
                {"id": [], "text": []},
                schema={"id": pl.Int64, "text": pl.String},
            ),
        }
    )

    chunks = list(runner.run_stream(job, input_batch_rows=2))

    assert chunks == []
    assert runner.last_stream_summary is not None
    assert runner.last_stream_summary.chunk_count == 0
    assert runner.last_stream_summary.total_rows == 0
    assert runner.last_stream_summary.submitted_rows == 0
    assert runner.last_stream_summary.result_cache_hits == 0


def test_runner_prefixes_structured_output_columns_in_pending_and_completed_results(
    tmp_path, monkeypatch
) -> None:
    class LocalLabelOutput(BaseModel):
        label: str

    @prompt_udf(version="v1")
    def prompt_with_label(row):
        return {
            "messages": [{"role": "user", "content": f"Label {row.text}"}],
            "expected": {"label": row.text.upper()},
        }

    job = StructuredOutputJob(
        name="prefixed_labels",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "text": ["alpha"], "label": ["existing"]}),
        prompt_builder=prompt_with_label,
        output_model=LocalLabelOutput,
        provider=OpenAIConfig(model="gpt-4o-mini", api_key="test"),
        result_cache=ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")),
        lock=LockConfig(),
        batch=BatchConfig(),
        output_column_prefix="output_",
    )
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, current_job: adapter)
    runner = Runner()

    first = runner.run(job)

    assert first["label"].to_list() == ["existing"]
    assert first["output_label"].to_list() == [None]
    assert first["llm_batch_py_status"].to_list() == ["submitted"]

    adapter.complete_all()
    second = runner.run(job)

    assert second["label"].to_list() == ["existing"]
    assert second["output_label"].to_list() == ["ALPHA"]
    assert second["llm_batch_py_status"].to_list() == ["cached"]


def test_runner_fails_fast_when_prefixed_output_columns_still_collide(tmp_path) -> None:
    class LocalLabelOutput(BaseModel):
        label: str

    @prompt_udf(version="v1")
    def prompt_with_label(row):
        return {
            "messages": [{"role": "user", "content": f"Label {row.text}"}],
            "expected": {"label": row.text.upper()},
        }

    collision_job = StructuredOutputJob(
        name="labels",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "text": ["alpha"], "output_label": ["existing"]}),
        prompt_builder=prompt_with_label,
        output_model=LocalLabelOutput,
        provider=OpenAIConfig(model="gpt-4o-mini", api_key="test"),
        result_cache=ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")),
        lock=LockConfig(),
        batch=BatchConfig(),
        output_column_prefix="output_",
    )

    with pytest.raises(
        ValueError,
        match=r"Prefixed structured output fields collide with input columns: \['output_label'\]",
    ):
        Runner().run(collision_job)


def test_runner_fails_fast_when_prefixed_output_columns_collide_with_llm_batch_py_metadata(
    tmp_path,
) -> None:
    class LocalStatusOutput(BaseModel):
        status: str

    @prompt_udf(version="v1")
    def prompt_with_status(row):
        return {
            "messages": [{"role": "user", "content": f"Status {row.text}"}],
            "expected": {"status": row.text.upper()},
        }

    collision_job = StructuredOutputJob(
        name="status_labels",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "text": ["alpha"]}),
        prompt_builder=prompt_with_status,
        output_model=LocalStatusOutput,
        provider=OpenAIConfig(model="gpt-4o-mini", api_key="test"),
        result_cache=ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")),
        lock=LockConfig(),
        batch=BatchConfig(),
        output_column_prefix="llm_batch_py_",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Structured output fields collide with reserved llm_batch_py columns: "
            r"\['llm_batch_py_status'\]"
        ),
    ):
        Runner().run(collision_job)


def test_runner_rejects_output_fields_that_collide_with_token_metadata(tmp_path) -> None:
    class LocalTokenOutput(BaseModel):
        input_tokens: int

    @prompt_udf(version="v1")
    def prompt_with_input_tokens(row):
        return {
            "messages": [{"role": "user", "content": f"Tokens {row.text}"}],
            "expected": {"input_tokens": len(row.text)},
        }

    collision_job = StructuredOutputJob(
        name="token_labels",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "text": ["alpha"]}),
        prompt_builder=prompt_with_input_tokens,
        output_model=LocalTokenOutput,
        provider=OpenAIConfig(model="gpt-4o-mini", api_key="test"),
        result_cache=ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")),
        lock=LockConfig(),
        batch=BatchConfig(),
        output_column_prefix="llm_batch_py_",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Structured output fields collide with reserved llm_batch_py columns: "
            r"\['llm_batch_py_input_tokens'\]"
        ),
    ):
        Runner().run(collision_job)


def test_runner_rejects_output_fields_that_collide_with_raw_output_metadata(tmp_path) -> None:
    class LocalRawOutputText(BaseModel):
        output_raw_text: str

    @prompt_udf(version="v1")
    def prompt_with_raw_output_text(row):
        return {
            "messages": [{"role": "user", "content": f"Raw {row.text}"}],
            "expected": {"output_raw_text": row.text.upper()},
        }

    collision_job = StructuredOutputJob(
        name="raw_output_labels",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "text": ["alpha"]}),
        prompt_builder=prompt_with_raw_output_text,
        output_model=LocalRawOutputText,
        provider=OpenAIConfig(model="gpt-4o-mini", api_key="test"),
        result_cache=ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")),
        lock=LockConfig(),
        batch=BatchConfig(),
        output_column_prefix="llm_batch_py_",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Structured output fields collide with reserved llm_batch_py columns: "
            r"\['llm_batch_py_output_raw_text'\]"
        ),
    ):
        Runner().run(collision_job)


def test_cache_key_includes_openai_backend_identity(structured_job) -> None:
    runner = Runner()
    first_request = runner._build_requests(structured_job, structured_job.materialized_input())[0]

    alternate_job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "provider": structured_job.provider.__class__(
                **{
                    **structured_job.provider.__dict__,
                    "base_url": "https://example.invalid/v1",
                }
            ),
        }
    )
    second_request = runner._build_requests(alternate_job, alternate_job.materialized_input())[0]

    assert first_request.cache_key != second_request.cache_key


def test_cache_key_includes_provider_prompt_cache_identity(structured_job) -> None:
    runner = Runner()
    first_request = runner._build_requests(structured_job, structured_job.materialized_input())[0]

    cached_job = structured_job.__class__(
        **{
            **structured_job.__dict__,
            "prompt_cache": PromptCacheConfig(mode="auto", verbose=True),
        }
    )
    second_request = runner._build_requests(cached_job, cached_job.materialized_input())[0]

    assert first_request.cache_key != second_request.cache_key


def test_runner_reuses_output_schema_across_request_build_and_prepare(
    monkeypatch, structured_job
) -> None:
    output_model_json_schema.cache_clear()
    calls = 0
    original = structured_job.output_model.model_json_schema

    def counted_schema():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(structured_job.output_model, "model_json_schema", counted_schema)
    adapter = Runner()._adapter(structured_job)
    requests = Runner()._build_requests(structured_job, structured_job.materialized_input())

    adapter.prepare_requests(structured_job, requests)

    assert calls == 1


def test_runner_marks_schema_mismatch_results_failed(monkeypatch, structured_job) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()

    runner.run(structured_job)
    custom_id = next(iter(adapter.batches.values())).requests[0]["custom_id"]
    adapter.complete_all(outcomes={custom_id: FakeOutcome(kind="malformed")})

    result = runner.run(structured_job)

    assert result["llm_batch_py_status"].to_list()[0] == "failed"
    assert result["llm_batch_py_error_code"].to_list()[0] == "schema_validation_error"


def test_runner_failed_rows_preserve_provider_model_and_raw_payload(
    monkeypatch, structured_job
) -> None:
    adapter = FakeAdapter()
    monkeypatch.setattr(Runner, "_adapter", lambda self, job: adapter)
    runner = Runner()

    runner.run(structured_job)
    custom_id = next(iter(adapter.batches.values())).requests[0]["custom_id"]
    adapter.complete_all(outcomes={custom_id: FakeOutcome(kind="malformed")})

    result = runner.run(structured_job)

    assert result["llm_batch_py_status"].to_list()[0] == "failed"
    assert result["llm_batch_py_provider"].to_list()[0] == structured_job.provider.provider_name
    assert result["llm_batch_py_model"].to_list()[0] == structured_job.provider.model
    assert result["llm_batch_py_input_tokens"].to_list()[0] is None
    assert result["llm_batch_py_output_tokens"].to_list()[0] is None
    assert result["llm_batch_py_input_raw_json"].to_list()[0] is not None
    assert result["llm_batch_py_request_raw_json"].to_list()[0] is not None
    assert result["llm_batch_py_output_raw_json"].to_list()[0] is not None
    assert result["llm_batch_py_output_raw_text"].to_list()[0] is None
    assert result["llm_batch_py_error_raw_json"].to_list()[0] is not None


def test_runner_fails_fast_on_structured_output_input_column_collision(tmp_path) -> None:
    class LocalLabelOutput(BaseModel):
        label: str

    @prompt_udf(version="v1")
    def prompt_with_label(row):
        return {
            "messages": [{"role": "user", "content": f"Label {row.text}"}],
            "expected": {"label": row.text.upper()},
        }

    collision_job = StructuredOutputJob(
        name="labels",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "text": ["alpha"], "label": ["existing"]}),
        prompt_builder=prompt_with_label,
        output_model=LocalLabelOutput,
        provider=OpenAIConfig(model="gpt-4o-mini", api_key="test"),
        result_cache=ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")),
        lock=LockConfig(),
        batch=BatchConfig(),
    )

    with pytest.raises(
        ValueError, match=r"Structured output fields collide with input columns: \['label'\]"
    ):
        Runner().run(collision_job)


def test_runner_allows_key_column_overlap_with_structured_output(tmp_path) -> None:
    class OutputWithKey(BaseModel):
        id: int
        label: str

    @prompt_udf(version="v1")
    def prompt_with_key(row):
        return {
            "messages": [{"role": "user", "content": f"Label {row.text}"}],
            "expected": {"id": row.id, "label": row.text.upper()},
        }

    job = StructuredOutputJob(
        name="labels_with_id",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "text": ["alpha"]}),
        prompt_builder=prompt_with_key,
        output_model=OutputWithKey,
        provider=OpenAIConfig(model="gpt-4o-mini", api_key="test"),
        result_cache=ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")),
        lock=LockConfig(),
        batch=BatchConfig(),
    )
    validated = validate_job_input(job)

    assert validated.to_dicts() == [{"id": 1, "text": "alpha"}]
