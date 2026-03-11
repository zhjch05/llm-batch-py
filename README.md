# llm-batch-py

`llm-batch-py` is a cron-friendly batch LLM runner for Polars.

Install with `pip install llm-batch-py`.

It lets you:

- define structured-output or embedding jobs against a Polars `DataFrame` or `LazyFrame`
- build prompts with Rust-backed row templates, with Python UDFs available as fallback
- auto-poll prior batches, auto-submit only missing rows, and materialize a fresh Polars result table
- process large inputs incrementally with `Runner.run_stream(...)` from a `DataFrame`, `LazyFrame`, or host-supplied `Iterable[DataFrame]`
- persist manifests and raw artifacts in local or S3-backed parquet storage

Reruns are grouped implicitly by `job.name` inside one shared `result_cache.root_uri`. If you run the same job again against the same cache store, `llm-batch-py` reuses completed results, skips duplicate submission for still-active prior work, and only submits rows whose effective request identity changed or whose retries are still allowed.

`llm-batch-py` is cron-friendly, but it does not schedule itself. Your scheduler invokes `Runner.run()`, and each rerun acquires a short-lived job lock, polls prior batches, reuses completed results, keeps matching in-flight rows pending, and submits only new or changed rows. Changing `BatchConfig.batch_size` affects only future submissions; it does not invalidate result-cache hits for completed rows.

Provider batch submission is sequential. When one run needs multiple small provider batches, `llm-batch-py` submits them one by one rather than firing all submits concurrently.

If a small batch submit fails transiently before the provider accepts it, `llm-batch-py` does a short inline retry loop in the same `run()`. If the submit still fails, the batch stays recoverable in the result cache and later reruns retry that same small batch before submitting any later chunks. `BatchConfig.max_retries` applies per retryable small-batch submit and per retryable row-level provider failure.

`LazyFrame` support does not make `Runner.run()` fully streaming by itself. `Runner.run()` still materializes the full job input before validation and request building. Use `Runner.run_stream(...)` when you want chunked execution over a `DataFrame`, `LazyFrame`, or host-supplied `Iterable[DataFrame]`.

## Quickstart

```python
import polars as pl
from pydantic import BaseModel

from llm_batch_py import (
    BatchConfig,
    LockConfig,
    OpenAIConfig,
    PromptCacheConfig,
    ResultCacheStoreConfig,
    Runner,
    StructuredOutputJob,
    structured_template,
)


class CompanyLabel(BaseModel):
    label: str


build_prompt = structured_template(
    system="Return JSON only.",
    messages="Label this company: {{ row.company_name }}",
    name="build_prompt",
    version="v1",
)


job = StructuredOutputJob(
    name="company_labels",
    key_cols=["id"],
    input_df=pl.DataFrame({"id": [1], "company_name": ["OpenAI"]}),
    prompt_builder=build_prompt,
    output_model=CompanyLabel,
    provider=OpenAIConfig(model="gpt-4o-mini"),
    result_cache=ResultCacheStoreConfig(root_uri="./.llm_batch_py"),
    prompt_cache=PromptCacheConfig(mode="auto"),
    lock=LockConfig(ttl_seconds=3600),
    batch=BatchConfig(batch_size=500),
)

runner = Runner()
result_df = runner.run(job)
print(result_df)
print(runner.last_summary)

slim_df = runner.run(job, metadata_columns=["llm_batch_py_status"])
print(slim_df)
```

## Streaming Large Inputs

```python
import polars as pl

lazy_input = (
    pl.scan_parquet("./companies.parquet")
    .select(["id", "company_name"])
)

runner = Runner()
for chunk_df in runner.run_stream(
    job=job.__class__(**{**job.__dict__, "input_df": lazy_input}),
    input_batch_rows=10_000,
):
    print(chunk_df.select(["id", "llm_batch_py_status"]))

print(runner.last_stream_summary)
```

`run_stream()` yields one result `DataFrame` per input chunk. Each chunk uses the same result-cache, polling, retry, and lock semantics as a normal short-lived `run()`.

If your host already pages from Postgres or another source, you can also push batches directly:

```python
def pg_batches() -> list[pl.DataFrame]:
    return [
        pl.DataFrame({"id": [1], "company_name": ["OpenAI"]}),
        pl.DataFrame({"id": [2], "company_name": ["Anthropic"]}),
    ]


for chunk_df in runner.run_stream(job=job, input_batches=pg_batches()):
    print(chunk_df.select(["id", "llm_batch_py_status"]))
```

For private S3-backed result caches, pass filesystem options through `ResultCacheStoreConfig`:

```python
result_cache = ResultCacheStoreConfig(
    root_uri="s3://my-bucket/llm_batch_py-prod",
    storage_options={
        "profile": "my-profile",
        "client_kwargs": {"region_name": "us-west-2"},
    },
)
```

`structured_template(...)` and `embedding_template(...)` are the primary prompt-building APIs. They render `{{ row.field }}` placeholders through the Rust template engine for lower per-row overhead.

`prompt_udf(...)` remains supported as a compatibility fallback when templating is not expressive enough. A `prompt_udf` should return structured Python data such as dicts, lists, Pydantic models, and datetime-like values. `llm-batch-py` canonically serializes the rendered payload before computing result-cache keys, so dict insertion order does not affect cache hits.

Returned result frames include `llm_batch_py_*` metadata columns for provider status, token counts, and raw request/response inspection fields such as `llm_batch_py_input_raw_json`, `llm_batch_py_request_raw_json`, `llm_batch_py_output_raw_json`, and `llm_batch_py_output_raw_text`.

If you do not need the full metadata surface on the returned frame, pass `metadata_columns=[...]` to `Runner.run()` or `Runner.run_stream()` to join back only the requested `llm_batch_py_*` columns. Pass `metadata_columns=[]` to suppress metadata columns entirely.

## Docs

- [Prompt building guide](./docs/prompt-building.md)
- [Provider config reference](./docs/provider-configs.md)
- [Result cache guide](./docs/result-cache.md)
- [Prompt caching guide](./docs/prompt-caching.md)
- [Locking guide](./docs/locking.md)
- [Batching guide](./docs/batching.md)
- [Streaming guide](./docs/streaming.md)

## Customizing provider prompt cache

```python
from llm_batch_py import PromptCacheConfig

PromptCacheConfig(
    mode="auto",
    verbose=False,
)
```

- `mode: Literal["off", "auto"] = "auto"`
- `verbose: bool = False`

### `OpenAIConfig`

```python
from llm_batch_py import OpenAIConfig

OpenAIConfig(
    model="gpt-4o-mini",
    api_key=None,
    max_output_tokens=None,
    temperature=None,
    timeout=60.0,
    organization=None,
    base_url=None,
    dimensions=None,
    pricing=None,
)
```

- `model: str`
- `api_key: str | None = None`
- `max_output_tokens: int | None = None`
- `temperature: float | None = None`
- `timeout: float | None = 60.0`
- `organization: str | None = None`
- `base_url: str | None = None`
- `dimensions: int | None = None`
- `pricing: ModelPricing | None = None`
- `provider_name: Literal["openai"] = "openai"`

`OpenAIConfig` is used for:

- structured-output jobs
- embedding jobs

### `AnthropicConfig`

```python
from llm_batch_py import AnthropicConfig

AnthropicConfig(
    model="claude-3-5-haiku-latest",
    api_key=None,
    max_output_tokens=1024,
    temperature=None,
    timeout=60.0,
    pricing=None,
)
```

- `model: str`
- `api_key: str | None = None`
- `max_output_tokens: int = 1024`
- `temperature: float | None = None`
- `timeout: float | None = 60.0`
- `pricing: ModelPricing | None = None`
- `provider_name: Literal["anthropic"] = "anthropic"`

`AnthropicConfig` is used for structured-output jobs only.

## Customizing result cache store config

`result_cache=ResultCacheStoreConfig(...)` controls the persistent result cache catalog used to reuse prior results. It is separate from the `result_df` returned by `runner.run(job)`.

Completed results are stored in the result cache catalog and reused on later runs until you delete or replace that catalog. There is no result TTL or automatic result-cache eviction in `llm-batch-py` today.

```python
from llm_batch_py import ResultCacheStoreConfig

result_cache = ResultCacheStoreConfig(
    root_uri="s3://my-bucket/llm_batch_py-prod",
)
```

- `root_uri`: base location for manifests, raw artifacts, and reusable cached results. This can be a local path like `"./.llm_batch_py"` or any `fsspec` URI such as `s3://...`.

### Default result cache key

Each request gets a content-addressed cache key for result reuse. `llm-batch-py` hashes the request payload together with the job/provider context, using SHA-256 over a canonical JSON representation.

By default the cache key includes:

- the rendered request payload from your `prompt_builder` or `text_builder`
- `job.name`
- `prompt_builder.version` or `text_builder.version`
- provider name and model
- endpoint kind (`structured` or `embeddings`)
- structured-output schema, when applicable
- provider options that affect outputs: `temperature`, `max_output_tokens`, `dimensions`, `base_url`, and `organization`
- shared prompt-cache config, when set on a structured-output job

Notably, row IDs and `key_cols` are not part of the result cache key. If two rows in the same job render the same payload under the same config, they will hit the same cached result.

### Cache key customization

There is no explicit `cache_key_fn` or cache-key override API today.

The supported ways to change cache identity are:

- change `job.name` to create a separate cache namespace
- bump the prompt builder version when prompt semantics change
- change the rendered payload or provider/model settings
- point `root_uri` at a different result cache store if you want fully isolated cached results

## Customizing lock config

`lock=LockConfig(...)` controls how `llm-batch-py` recovers from abandoned job locks.

```python
from llm_batch_py import LockConfig

lock = LockConfig(ttl_seconds=6 * 60 * 60)
```

- `ttl_seconds`: if a previous run left a lock behind and it is older than this TTL, a new run can reclaim it. Default: `3600`.

If you mean provider prompt caching rather than the `llm-batch-py` result cache store, configure that on the job:

```python
from llm_batch_py import PromptCacheConfig

job = StructuredOutputJob(
    ...,
    prompt_cache=PromptCacheConfig(mode="auto"),
)
```

`PromptCacheConfig(...)` controls shared provider-side prompt caching behavior.

- `mode="off"` disables provider prompt caching.
- `mode="auto"` uses provider-managed prompt caching.
- `verbose=True` logs one INFO estimated-analysis diagnostic per distinct prompt shape showing the likely cacheable prefix candidate and likely trailing dynamic content candidate.
- `llm-batch-py` does not expose provider-specific prompt-cache options; it only enables shared automatic caching behavior.

## Customizing batch config

`batch=BatchConfig(...)` controls how pending rows are grouped into provider batch submissions.

```python
from llm_batch_py import BatchConfig

batch = BatchConfig(
    batch_size=500,
    max_retries=3,
)
```

- `batch_size`: hard cap on requests per submitted batch. If unset, `llm-batch-py` uses the provider's built-in batch request limit.
- `max_retries`: maximum retries for retryable failed rows before they are held as failed. Default: `2`.

Provider payload byte caps and batch completion windows are derived internally from the selected provider adapter rather than configured in `BatchConfig`.
