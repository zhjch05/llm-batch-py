# Streaming Guide

`llm-batch-py` supports chunked input execution through `Runner.run_stream(...)`.

Use this when your source data is too large to materialize into one in-memory `DataFrame` for a single `Runner.run()` call.

## API

```python
from collections.abc import Iterator, Sequence

import polars as pl

from llm_batch_py import Runner, StreamRunSummary, StructuredOutputJob

runner = Runner()
chunks: Iterator[pl.DataFrame] = runner.run_stream(
    job,
    input_batch_rows=10_000,
    order_by=None,
    input_batches=None,
    dry_run=False,
    avg_output_tokens=0,
    metadata_columns=None,
)
```

Arguments:

- `input_batch_rows`: required chunk size for the materialized input slices
- `order_by`: optional deterministic ordering for chunking; defaults to `job.key_cols`
- `input_batches`: optional host-supplied `Iterable[pl.DataFrame]`; use this instead of `input_batch_rows` when the caller already pages the source
- `dry_run`: same meaning as `Runner.run()`
- `avg_output_tokens`: same meaning as `Runner.run()`
- `metadata_columns`: optional subset of `llm_batch_py_*` columns to join back; defaults to all metadata columns

Pass exactly one of `input_batch_rows` or `input_batches`.

Return value:

- an iterator of result `DataFrame` chunks

Summary:

- `runner.last_stream_summary` contains aggregate totals across all yielded chunks
- `runner.last_summary` contains the summary for the most recently processed chunk

## How Chunking Works

### `DataFrame` input

- `run_stream()` slices the existing in-memory frame into batches of `input_batch_rows`
- if `order_by` is provided, the frame is sorted before slicing

### `LazyFrame` input

- `run_stream()` sorts the lazy query by `order_by` or `job.key_cols`
- it repeatedly collects bounded slices until no rows remain

This is chunked materialization, not a database cursor API. The lazy query still needs to support deterministic sorted slicing.

### Host-supplied iterable batches

- `run_stream(input_batches=...)` accepts any iterable that yields Polars `DataFrame` batches
- `llm-batch-py` validates each batch independently and executes it as one short-lived run
- duplicate `key_cols` across provided batches raise `ValueError`

Use this when your application already pages from Postgres, Snowflake, or another source and wants `llm-batch-py` to preserve the normal cache and polling behavior for each pushed batch.

## Result Shape

Each yielded chunk is the same shape as a normal `Runner.run()` result for those rows:

- original input columns
- structured output columns or embedding columns
- `llm_batch_py_*` metadata columns, including raw request/response fields such as `llm_batch_py_input_raw_json`, `llm_batch_py_request_raw_json`, `llm_batch_py_output_raw_json`, and `llm_batch_py_output_raw_text`

`run_stream()` does not assemble one final combined result frame in memory.

If you only need a narrow metadata surface on the returned chunks, pass something like `metadata_columns=["llm_batch_py_status"]`. Pass `metadata_columns=[]` to return no `llm_batch_py_*` columns at all.

## Locking, Cache, and Retry Behavior

Each streamed chunk is processed as its own short-lived runner invocation.

That means every chunk:

- acquires and releases the normal job lock
- polls prior provider batches before deciding new work
- reuses result-cache hits from the shared catalog
- preserves retry and recovery behavior for pending rows and retryable submit failures

This keeps streamed execution aligned with the existing cron-friendly rerun model, including when the host supplies batches directly.

## Ordering and Key Requirements

`run_stream()` assumes `key_cols` uniquely identify rows across the full streamed source, not just within one chunk.

Rules:

- each chunk must contain the `key_cols`
- `order_by` columns must exist on the source
- duplicate `key_cols` across different chunks raise `ValueError`

If your source does not have globally stable keys, normalize that before using `run_stream()`.

## Postgres-style Host Paging Example

```python
import polars as pl

from llm_batch_py import Runner


def fetch_pg_batches() -> list[pl.DataFrame]:
    return [
        pl.DataFrame({"id": [1], "company_name": ["OpenAI"]}),
        pl.DataFrame({"id": [2], "company_name": ["Anthropic"]}),
    ]


runner = Runner()
for result_chunk in runner.run_stream(job, input_batches=fetch_pg_batches()):
    print(result_chunk.select(["id", "llm_batch_py_status"]))
```

This path preserves the same polling, result-cache reuse, and retry behavior as `run_stream(..., input_batch_rows=...)`. The difference is only who owns pagination.

## Example

```python
import polars as pl
from pydantic import BaseModel

from llm_batch_py import OpenAIConfig, ResultCacheStoreConfig, Runner, StructuredOutputJob, prompt_udf


class CompanyLabel(BaseModel):
    label: str


@prompt_udf(version="v1")
def build_prompt(row):
    return {
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": f"Label this company: {row.company_name}"},
        ]
    }


job = StructuredOutputJob(
    name="company_labels",
    key_cols=["id"],
    input_df=pl.scan_parquet("./companies.parquet"),
    prompt_builder=build_prompt,
    output_model=CompanyLabel,
    provider=OpenAIConfig(model="gpt-4o-mini"),
    result_cache=ResultCacheStoreConfig(root_uri="./.llm_batch_py"),
)

runner = Runner()
for result_chunk in runner.run_stream(job, input_batch_rows=10_000):
    print(result_chunk.select(["id", "label", "llm_batch_py_status"]))

print(runner.last_stream_summary)
```

## When Not To Use It

Prefer plain `Runner.run()` when:

- the full input already fits comfortably in memory
- you want one combined result `DataFrame` directly
- you do not need deterministic chunk ordering

If your main need is direct paging from Postgres or another external source, a future source-iterator API would be a better fit than `LazyFrame` chunking.
