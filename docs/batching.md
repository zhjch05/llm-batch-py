# Batching Guide

`llm-batch-py` groups pending rows into provider batch submissions. `BatchConfig` controls the public batching knobs.

This is different from streamed input chunking. `Runner.run_stream(..., input_batch_rows=...)` controls how many input rows are materialized into each short-lived runner invocation. `BatchConfig.batch_size` controls how many prepared requests from that chunk are sent in each provider batch submission.

## Config

```python
from llm_batch_py import BatchConfig

BatchConfig(
    batch_size=None,
    max_retries=2,
)
```

Options:

- `batch_size: int | None = None`
- `max_retries: int = 2`

## Option Details

### `batch_size`

`batch_size` is the hard cap on requests per submitted batch.

- when set, `llm-batch-py` chunks prepared requests into batches of at most that size
- when `None`, `llm-batch-py` uses the selected provider adapter's built-in batch request limit
- chunked provider batches are submitted sequentially, one small batch at a time
- when using `run_stream()`, this limit applies inside each streamed input chunk rather than across the full source

Example:

```python
batch = BatchConfig(batch_size=500)
```

### `max_retries`

`max_retries` is the maximum retry budget for each retryable unit of work.

- for row-level provider failures, it controls how many retryable failed rows can be resubmitted before `llm-batch-py` holds the row as failed
- for small-batch submit failures before the provider accepts a batch, it controls how many reruns can retry that same small batch before `llm-batch-py` marks its rows failed
- `llm-batch-py` also performs a small internal inline retry loop during the same `run()` for transient submit failures; those inline retries do not consume the cross-run retry budget
- if an earlier small batch is still in retryable submit-failed state, later chunks are not submitted until that earlier chunk recovers or exhausts retries

Example:

```python
batch = BatchConfig(max_retries=3)
```

## Basic Usage

```python
from llm_batch_py import BatchConfig, StructuredOutputJob

job = StructuredOutputJob(
    ...,
    batch=BatchConfig(batch_size=500, max_retries=3),
)
```

## Input Chunking vs Provider Batching

These knobs operate at different layers:

- `Runner.run_stream(..., input_batch_rows=10_000)` controls input chunking and result-frame size.
- `BatchConfig(batch_size=500)` controls provider submission size inside each chunk.

For example, one streamed chunk of 10,000 input rows may still be submitted as 20 sequential provider batches of 500 requests each.

`run_stream()` processes each input chunk as a separate short-lived runner invocation. That means each chunk reacquires the normal job lock, polls prior batches, reuses cached results, and only submits missing work for that chunk.

## What Is Not Configurable Here

These are derived internally from the provider adapter rather than `BatchConfig`:

- provider payload byte caps
- provider batch completion windows
- provider-specific submission mechanics
