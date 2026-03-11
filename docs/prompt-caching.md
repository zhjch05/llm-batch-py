# Prompt Caching Guide

`llm-batch-py` exposes a single shared prompt-caching config:

- `PromptCacheConfig(mode="auto")` enables provider-managed prompt caching
- `PromptCacheConfig(mode="off")` disables it
- `verbose=True` logs one INFO estimated-analysis diagnostic per distinct prompt shape

There are no provider-specific prompt-cache config types.

## Config

```python
from llm_batch_py import PromptCacheConfig

PromptCacheConfig(
    mode="auto",
    verbose=False,
)
```

Options:

- `mode: Literal["off", "auto"] = "auto"`
- `verbose: bool = False`

### Option Details

- `mode="off"` disables provider prompt caching.
- `mode="auto"` enables the shared provider-managed prompt caching path used by `llm-batch-py`.
- `verbose=True` logs one INFO estimated-analysis diagnostic per distinct prompt shape.

## Prompt Caching vs Result Caching

Prompt caching and `llm-batch-py` result caching are different layers:

- Prompt caching reduces repeated prompt processing at the provider level
- `llm-batch-py` result caching reuses completed row results from the result cache catalog

If a row is served from the `llm-batch-py` result cache catalog, no provider request is sent, so provider prompt caching is not involved for that row.

## Basic Usage

Prompt caching is configured on `StructuredOutputJob`:

```python
import polars as pl
from pydantic import BaseModel

from llm_batch_py import (
    OpenAIConfig,
    PromptCacheConfig,
    ResultCacheStoreConfig,
    Runner,
    StructuredOutputJob,
    prompt_udf,
)


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
    input_df=pl.DataFrame({"id": [1], "company_name": ["OpenAI"]}),
    prompt_builder=build_prompt,
    output_model=CompanyLabel,
    provider=OpenAIConfig(model="gpt-4o-mini"),
    result_cache=ResultCacheStoreConfig(root_uri="./.llm_batch_py"),
    prompt_cache=PromptCacheConfig(mode="auto", verbose=True),
)

result_df = Runner().run(job)
```

The same `prompt_cache` field works with Anthropic:

```python
from llm_batch_py import AnthropicConfig, PromptCacheConfig

job = StructuredOutputJob(
    name="company_labels",
    key_cols=["id"],
    input_df=pl.DataFrame({"id": [1], "company_name": ["Anthropic"]}),
    prompt_builder=build_prompt,
    output_model=CompanyLabel,
    provider=AnthropicConfig(model="claude-3-5-haiku-latest"),
    result_cache=ResultCacheStoreConfig(root_uri="./.llm_batch_py"),
    prompt_cache=PromptCacheConfig(mode="auto"),
)
```

## OpenAI Behavior

With OpenAI providers:

- `llm-batch-py` keeps the request body shape the same
- prompt caching is provider-managed rather than configured through extra request fields
- `verbose=True` logs estimated cache-boundary analysis for the likely cached prefix and uncached suffix of the prompt

Use this when you want simple, shared automatic prompt caching without OpenAI-only config in your job interface.

## Anthropic Behavior

With Anthropic providers:

- `llm-batch-py` inserts one `cache_control={"type": "ephemeral"}` breakpoint on an Anthropic cacheable prompt block
- that breakpoint is applied to the estimated reusable prompt prefix rather than the final dynamic suffix
- string `system` or message content may be normalized into Anthropic text blocks so the breakpoint can be attached
- `verbose=True` logs estimated cache-boundary analysis for the likely cached prefix and uncached suffix of the prompt

This gives a simple Anthropic path without requiring explicit cache-breakpoint configuration in `llm-batch-py`.

## Diagnostics

Set `verbose=True` to emit one INFO log per distinct prompt shape.

These diagnostics show:

- the provider and model
- the estimated cached prompt prefix
- the estimated uncached dynamic suffix
- any request fields `llm-batch-py` added for prompt caching

These diagnostics are heuristic only. Actual caching is decided by the provider APIs.

The diagnostics are deduplicated by prompt shape and config, so repeated identical requests log once.

## Cache Identity

`llm-batch-py` result cache keys include prompt-cache config when prompt caching is configured on a structured-output job.

That means changing:

- `mode`
- `verbose`

changes `llm-batch-py` result cache identity for structured-output jobs.

This prevents cached results from being reused across materially different request configurations.

## Recommendations

- Start with `PromptCacheConfig(mode="auto")`
- Add `verbose=True` while tuning prompts, then turn it off for steady-state runs
- Keep using prompt-builder version bumps when prompt semantics change
- Treat provider prompt caching and `llm-batch-py` result caching as complementary layers
