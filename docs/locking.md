# Locking Guide

`llm-batch-py` uses job locks to prevent concurrent runs from corrupting shared job state in the result cache catalog.

## Config

`LockConfig` controls stale-lock recovery:

```python
from llm_batch_py import LockConfig

LockConfig(
    ttl_seconds=3600,
)
```

Options:

- `ttl_seconds: int = 3600`

## Behavior

If a prior run exits unexpectedly and leaves a lock behind, a later run can reclaim that lock once it is older than `ttl_seconds`.

Use a larger TTL when:

- jobs run for a long time
- you want to reduce the chance of reclaiming a still-valid lock

Use a smaller TTL when:

- runs are short
- abandoned locks need to clear quickly

## Operational Guidance

`LockConfig.ttl_seconds` should cover the wall-clock time of one `Runner.run()` invocation, not the full provider-side batch lifetime.

- lock acquisition happens before polling, diffing, or submission
- if a valid lock is still present, that rerun exits with `Lock already held for job ...`
- in that blocked invocation, no extra rows are processed
- after the lock is released or reclaimed as stale, the next rerun sees the latest input and processes only missing or changed rows

This means a provider batch can remain `in_progress` for many hours while your cron job keeps re-running safely, as long as each individual `Runner.run()` call is short-lived and your TTL is long enough to avoid reclaiming an active runner.

### Choosing a TTL

Use a TTL that is longer than the worst-case runtime of one `Runner.run()` plus some safety buffer.

- size TTL to protect the active `llm-batch-py` process
- do not size TTL to the provider's batch completion window
- if cron runs more frequently than jobs usually finish, later invocations should treat `Lock already held` as an expected no-op

Example:

- cron every 10 minutes
- typical `Runner.run()` wall time is 1 to 3 minutes
- choose `ttl_seconds` around 15 to 30 minutes

If you set TTL too short, a still-running invocation can be treated as stale and another process can reclaim the lock early.

## Basic Usage

```python
from llm_batch_py import LockConfig, StructuredOutputJob

job = StructuredOutputJob(
    ...,
    lock=LockConfig(ttl_seconds=6 * 60 * 60),
)
```

## Scope

`LockConfig` only controls `llm-batch-py` job-lock behavior around the shared result cache catalog. It does not control:

- provider-side prompt caching
- provider-side batch retention windows
- result-cache entry lifetime

For those topics, see:

- [Prompt caching guide](./prompt-caching.md)
- [Result cache guide](./result-cache.md)
- [Batching guide](./batching.md)
