# S3 Cache Storage Guide

`llm-batch-py` can store its result cache in an `s3://...` location through `fsspec` and `s3fs`. There is no separate login flow inside the library. Whatever credentials `s3fs` can use are the credentials the result cache can use.

## What You Need

For a private AWS S3 bucket, you need one of these credential paths:

- environment variables: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- temporary credentials: add `AWS_SESSION_TOKEN`
- a named profile via `AWS_PROFILE`
- an attached IAM role on the machine, container, or batch job

In code, `ResultCacheStoreConfig.storage_options` is passed straight through to `fsspec.core.url_to_fs(...)`, so explicit overrides use normal `s3fs` keys such as `key`, `secret`, `token`, `profile`, and `client_kwargs`.

## AWS S3 Examples

Profile-based config:

```python
from llm_batch_py import ResultCacheStoreConfig

result_cache = ResultCacheStoreConfig(
    root_uri="s3://my-bucket/llm_batch_py-prod",
    storage_options={
        "profile": "analytics-prod",
        "client_kwargs": {"region_name": "us-west-2"},
    },
)
```

Explicit credentials:

```python
from llm_batch_py import ResultCacheStoreConfig

result_cache = ResultCacheStoreConfig(
    root_uri="s3://my-bucket/llm_batch_py-prod",
    storage_options={
        "key": "AWS_ACCESS_KEY_ID",
        "secret": "AWS_SECRET_ACCESS_KEY",
        "token": "AWS_SESSION_TOKEN",
        "client_kwargs": {"region_name": "us-west-2"},
    },
)
```

If you omit `storage_options`, normal AWS credential resolution still applies.

## Required Bucket Permissions

The cache store needs to list, read, write, and delete objects under the configured prefix.

That requirement comes from the current catalog implementation, which:

- lists manifest chunks
- checks whether lock files and legacy manifests exist
- reads manifests, locks, and raw artifacts
- writes manifests, raw artifacts, and lock files
- deletes lock files during normal release and stale-lock recovery

In IAM terms, the role or user typically needs bucket-level list access plus object-level read, write, and delete access for the cache prefix.

## S3-Compatible Endpoints

For MinIO or other S3-compatible endpoints, you usually need:

- `key`
- `secret`
- optional `token`
- `client_kwargs.endpoint_url`
- usually `client_kwargs.region_name`

Example:

```python
from llm_batch_py import ResultCacheStoreConfig

result_cache = ResultCacheStoreConfig(
    root_uri="s3://my-bucket/llm_batch_py-prod",
    storage_options={
        "key": "ACCESS_KEY",
        "secret": "SECRET_KEY",
        "client_kwargs": {
            "endpoint_url": "https://minio.internal.example",
            "region_name": "us-east-1",
        },
    },
)
```

Accepted keys come from `s3fs`. If you need a provider-specific knob such as signature version, pass it through the matching `s3fs` option.

## Locking Safety

`llm-batch-py` uses lock files inside the result cache to prevent concurrent runs from corrupting shared state. The current implementation depends on exclusive-create semantics that `s3fs` documents as supported on AWS S3.

Because of that, `llm-batch-py` now rejects custom `endpoint_url` S3 lock configs by default:

```python
from llm_batch_py import LockConfig

lock = LockConfig()
```

If you really need to use an S3-compatible endpoint anyway, you must opt in explicitly:

```python
from llm_batch_py import LockConfig

lock = LockConfig(allow_unsafe_s3_compatible_locks=True)
```

That bypass only suppresses the safeguard. It does not make the lock protocol safe. Use it only if you serialize runners externally or otherwise accept the risk of concurrent cache corruption.

## Source of Truth

Credential and filesystem option behavior comes from `s3fs` / `botocore`, not from a custom auth layer in `llm-batch-py`.
