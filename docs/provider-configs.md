# Provider Config Reference

`llm-batch-py` exposes two public provider config types:

- `OpenAIConfig`
- `AnthropicConfig`

`OpenAIConfig` works for structured-output jobs and embedding jobs. `AnthropicConfig` works for structured-output jobs only.

## `OpenAIConfig`

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

Options:

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

### Option Details

- `model`: provider model name.
- `api_key`: API key passed to the OpenAI client. If omitted, the underlying client environment fallback can be used.
- `max_output_tokens`: structured-output token cap. For embeddings, this field is not used.
- `temperature`: structured-output sampling temperature. For embeddings, this field is not used.
- `timeout`: client timeout in seconds.
- `organization`: optional OpenAI organization header.
- `base_url`: optional custom API base URL.
- `dimensions`: embedding dimension override for embedding jobs. This is also part of result-cache identity.
- `pricing`: optional pricing override used for cost estimation. If omitted, `llm-batch-py` uses built-in pricing for known models when available.
- `provider_name`: fixed internal discriminator used by `llm-batch-py`.

### Job Compatibility

Use `OpenAIConfig` with:

- `StructuredOutputJob`
- `EmbeddingJob`

### Basic Examples

Structured output:

```python
from llm_batch_py import OpenAIConfig, StructuredOutputJob

job = StructuredOutputJob(
    ...,
    provider=OpenAIConfig(
        model="gpt-4o-mini",
        temperature=0.1,
        max_output_tokens=512,
    ),
)
```

Embeddings:

```python
from llm_batch_py import EmbeddingJob, OpenAIConfig

job = EmbeddingJob(
    ...,
    provider=OpenAIConfig(
        model="text-embedding-3-small",
        dimensions=512,
    ),
)
```

## `AnthropicConfig`

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

Options:

- `model: str`
- `api_key: str | None = None`
- `max_output_tokens: int = 1024`
- `temperature: float | None = None`
- `timeout: float | None = 60.0`
- `pricing: ModelPricing | None = None`
- `provider_name: Literal["anthropic"] = "anthropic"`

### Option Details

- `model`: provider model name.
- `api_key`: API key passed to the Anthropic client. If omitted, the underlying client environment fallback can be used.
- `max_output_tokens`: default output token cap for structured-output requests.
- `temperature`: sampling temperature.
- `timeout`: client timeout in seconds.
- `pricing`: optional pricing override used for cost estimation. If omitted, `llm-batch-py` uses built-in pricing for known models when available.
- `provider_name`: fixed internal discriminator used by `llm-batch-py`.

### Job Compatibility

Use `AnthropicConfig` with:

- `StructuredOutputJob`

`EmbeddingJob` does not accept `AnthropicConfig`.

For `StructuredOutputJob`, `llm-batch-py` uses Anthropic Messages Batches with a generated tool schema derived from the job's Pydantic output model and forces `tool_choice` to that tool. If your prompt payload also supplies Anthropic tools, `llm-batch-py` keeps them but still appends the generated output tool and forces that generated tool choice. Responses are parsed from `tool_use.input` when present and then validated locally against the same model.

### Basic Example

```python
from llm_batch_py import AnthropicConfig, StructuredOutputJob

job = StructuredOutputJob(
    ...,
    provider=AnthropicConfig(
        model="claude-3-5-haiku-latest",
        temperature=0.1,
        max_output_tokens=512,
    ),
)
```

## Pricing Overrides

Both provider configs accept `pricing=...` for cost estimation overrides.

The `pricing` value uses `ModelPricing`:

```python
from llm_batch_py.pricing import ModelPricing

ModelPricing(
    input_per_million=0.15,
    output_per_million=0.60,
)
```

Options:

- `input_per_million: float`
- `output_per_million: float = 0.0`

If `pricing` is omitted, `llm-batch-py` falls back to built-in defaults for known models when available.
