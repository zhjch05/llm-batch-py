# Result Cache Guide

`llm-batch-py` stores completed row results in a persistent result cache catalog so later runs can reuse them without re-submitting requests.

## Config

`ResultCacheStoreConfig` is the public config type for the result cache store:

```python
from typing import Any

from llm_batch_py import ResultCacheStoreConfig

ResultCacheStoreConfig(
    root_uri="./.llm_batch_py",
    storage_options=None,
)
```

Options:

- `root_uri: str`
- `storage_options: dict[str, Any] | None = None`

`root_uri` is the base location for manifests, raw artifacts, and reusable cached results. It can be a local path like `"./.llm_batch_py"` or an `fsspec` URI such as `"s3://my-bucket/llm_batch_py-prod"`.
`storage_options` is an optional passthrough of keyword arguments for the underlying `fsspec` filesystem. This is the supported way to configure private S3 buckets and other authenticated remote filesystems in code.

Rerun lineage is scoped by `job.name` within a single `root_uri`. Re-running the same logical pipeline with the same `job.name` and cache store lets `llm-batch-py` discover prior batches, reuse completed results, and skip duplicate submission while earlier matching work is still active or waiting for result ingestion.

On each rerun, `llm-batch-py` polls prior provider batches before deciding what new work to submit. Completed rows are served from the result cache, still-active matching rows stay pending/inflight without duplicate submission, and only new or changed rows are eligible for submission. New rows can be submitted in the same rerun that is still tracking older in-flight rows.

## API Aliases

The same type is exported under these public aliases:

- `ResultCacheConfig`
- `CacheStoreConfig`

Job constructors also accept either:

- `result_cache=...`
- `cache_store=...`

These arguments are equivalent. Pass only one. Passing both with different values raises `ValueError`.

## Basic Usage

```python
from llm_batch_py import OpenAIConfig, ResultCacheStoreConfig, StructuredOutputJob

job = StructuredOutputJob(
    ...,
    provider=OpenAIConfig(model="gpt-4o-mini"),
    result_cache=ResultCacheStoreConfig(root_uri="./.llm_batch_py"),
)
```

## Private S3 Buckets

For `s3://...` result cache stores, `llm-batch-py` does not implement a separate login flow. It passes `storage_options` through to `fsspec` / `s3fs`.

If you omit `storage_options`, normal AWS credential resolution still applies, including:

- environment variables
- `AWS_PROFILE` / shared credentials files
- IAM roles or container task roles

Use `storage_options` when you need explicit overrides in code.

Profile and region example:

```python
from llm_batch_py import ResultCacheStoreConfig

config = ResultCacheStoreConfig(
    root_uri="s3://my-bucket/llm_batch_py-prod",
    storage_options={
        "profile": "my-profile",
        "client_kwargs": {"region_name": "us-west-2"},
    },
)
```

Explicit credential example:

```python
from llm_batch_py import ResultCacheStoreConfig

config = ResultCacheStoreConfig(
    root_uri="s3://my-bucket/llm_batch_py-prod",
    storage_options={
        "key": "AWS_ACCESS_KEY_ID",
        "secret": "AWS_SECRET_ACCESS_KEY",
        "token": "AWS_SESSION_TOKEN",
        "client_kwargs": {"region_name": "us-west-2"},
    },
)
```

Custom endpoint example:

```python
from llm_batch_py import ResultCacheStoreConfig

config = ResultCacheStoreConfig(
    root_uri="s3://my-bucket/llm_batch_py-prod",
    storage_options={
        "key": "AWS_ACCESS_KEY_ID",
        "secret": "AWS_SECRET_ACCESS_KEY",
        "client_kwargs": {
            "region_name": "us-west-2",
            "endpoint_url": "https://s3.us-west-2.amazonaws.com",
        },
    },
)
```

Accepted `storage_options` keys are defined by the underlying filesystem backend. For S3, `s3fs` is the source of truth.

For a full private-S3 guide covering required credentials, required object-store permissions, and S3-compatible endpoint caveats, see [S3 cache storage guide](./s3-cache-storage.md).

If you configure a custom `endpoint_url`, `llm-batch-py` rejects job locking by default unless you opt into `LockConfig(allow_unsafe_s3_compatible_locks=True)`. That safeguard exists because the lock protocol depends on AWS S3 exclusive-create semantics.

## What Gets Stored

The result cache catalog persists:

- submitted batch manifests
- request manifests
- result manifests
- raw provider artifacts needed for polling and recovery
- completed reusable row results

Completed results remain reusable until you delete or replace the catalog. There is no built-in TTL or automatic eviction for result-cache entries.

If a prior batch has already finished at the provider but its results have not been ingested into the catalog yet, later runs keep that batch recoverable and retry ingestion instead of submitting the same requests again.

If a prior small-batch submit failed before the provider accepted it, later reruns retry recovery for that batch before submitting later chunks. If retryable submit failures persist past `BatchConfig.max_retries`, those rows are marked failed. There is no built-in timeout that automatically converts forever-active provider batches into failed rows.

## Result Cache vs Prompt Caching

`llm-batch-py` result caching is different from provider prompt caching:

- result caching reuses fully completed row outputs from the local or remote cache store
- prompt caching reduces repeated prompt processing at the provider side

If a row is served from the `llm-batch-py` result cache, no provider request is sent for that row.

For provider prompt caching, see [Prompt caching guide](./prompt-caching.md).

## Cache Identity

Each request gets a content-addressed cache key for result reuse. `llm-batch-py` hashes a canonical JSON representation of:

- the rendered request payload from your `prompt_builder` or `text_builder`
- `job.name`
- builder version: `prompt_builder.version` or `text_builder.version`
- provider name and model
- endpoint kind: `structured` or `embeddings`
- structured-output schema, when applicable
- `temperature`
- `max_output_tokens`
- `dimensions`
- `base_url`
- `organization`
- shared prompt-cache config, when present on a structured-output job

`key_cols` and row IDs are not part of the cache key. If two rows in the same job render the same payload under the same effective config, they will hit the same cached result.

Canonical JSON is the important mental model here: `llm-batch-py` normalizes the rendered payload into deterministic JSON text before hashing it. In practice, that means semantically equivalent payloads with different dict insertion order still produce the same cache key.

For example, these payloads are treated as identical for result-cache identity:

```python
payload_a = {
    "messages": [
        {"role": "user", "content": {"b": 2, "a": 1}},
    ]
}

payload_b = {
    "messages": [
        {"content": {"a": 1, "b": 2}, "role": "user"},
    ]
}
```

Both payloads canonicalize to the same JSON text and therefore hash to the same cache key.

If you want to inspect this behavior directly, the internal helper module exposes the same canonicalization utilities `llm-batch-py` uses when building requests:

```python
from llm_batch_py._core_wrapper import canonical_json, stable_hash

payload = {
    "messages": [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": {"b": 2, "a": 1}},
    ]
}

payload_json = canonical_json(payload)
cache_key = stable_hash(payload)
```

Treat this as an explanatory debugging aid rather than something you need to call inside `prompt_udf`. The intended pattern is still to return structured Python objects from your builder and let `llm-batch-py` own canonical serialization. Avoid returning ad hoc pre-serialized JSON strings when cache stability matters.

At a high level, the canonicalization step:

- orders dict keys deterministically
- normalizes Pydantic models via `model_dump(mode="json")`
- serializes datetime-like values via `isoformat()`

Unsupported custom object types may fail canonical serialization rather than being guessed from `str(obj)`.

## How To Change Cache Identity

There is no explicit cache-key override API today.

The supported ways to create a different cache identity are:

- change `job.name`
- bump the builder version when prompt semantics change
- change the rendered payload
- change provider/model settings that affect outputs
- change `root_uri` to point at a different result cache store
