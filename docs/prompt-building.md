# Prompt Building Guide

`llm-batch-py` supports two prompt-building styles:

- `structured_template(...)` / `embedding_template(...)` for declarative templating
- `prompt_udf(...)` for Python-defined prompt assembly

Use templates by default. Use a Python UDF only when the prompt shape cannot be expressed cleanly as a template.

## Template Builders

`structured_template(...)` is the primary API for structured-output jobs.

```python
from llm_batch_py import structured_template

build_prompt = structured_template(
    system="Return JSON only.",
    messages="Label this company: {{ row.company_name }}",
    name="build_prompt",
    version="v1",
)
```

`embedding_template(...)` is the equivalent API for embedding jobs.

```python
from llm_batch_py import embedding_template

build_text = embedding_template(
    "Embed {{ row.company_name }}",
    version="v1",
)
```

Template builders are the recommended default because they:

- run through the Rust-backed renderer
- make referenced input columns explicit
- keep prompt versions stable and predictable
- avoid Python work per row for common prompt shapes

## `row_snapshot(...)`

Templates include a `row_snapshot(...)` helper for rendering selected row fields as stable canonical JSON text.

```python
from llm_batch_py import structured_template

build_prompt = structured_template(
    messages=(
        "Product context:\n"
        '{{ row_snapshot(include=["current_name", "brand", "category"], priority=["current_name", "brand"]) }}'
    ),
    version="v1",
)
```

`row_snapshot(...)` is useful when you want:

- deterministic key ordering inside prompt text
- a compact JSON summary of selected row columns
- prompt text that is easy to diff and reason about

Supported options:

- `include=[...]`: only include these columns
- `exclude=[...]`: omit these columns
- `priority=[...]`: emit these keys first, then the rest in stable order

Important constraint: `row_snapshot(...)` works on row columns. It does not build arbitrary nested Python objects for you. If you already have a nested structure serialized into a row field like `vendor_data_json`, you can combine both patterns:

```python
build_prompt = structured_template(
    messages=(
        "Context:\n"
        '{{ row_snapshot(include=["current_name", "brand"]) }}\n\n'
        "Vendor data:\n"
        "{{ row.vendor_data_json }}"
    ),
    version="v1",
)
```

## Full Placeholder Rendering

When a field is exactly one placeholder, the rendered value keeps its structured type rather than being coerced to text.

```python
build_prompt = structured_template(
    messages=[{"role": "user", "content": "Summarize"}],
    expected={
        "id": "{{ row.id }}",
        "enabled": "{{ row.enabled }}",
        "tags": "{{ row.tags }}",
    },
    max_tokens="{{ row.max_tokens }}",
    version="v1",
)
```

This matters when you want integers, booleans, lists, or dicts to remain structured in the rendered payload.

## Python UDF Builders

Use `prompt_udf(...)` when prompt assembly genuinely needs Python logic.

```python
from llm_batch_py import prompt_udf


@prompt_udf(version="v1")
def build_prompt(row):
    tone = "short" if row.priority == "low" else "detailed"
    return {
        "system": "Return JSON only.",
        "messages": [
            {
                "role": "user",
                "content": f"Write a {tone} summary for {row.company_name}",
            }
        ],
    }
```

Typical reasons to use a UDF:

- row-dependent branching that would make a template unreadable
- nontrivial Python data shaping before prompt construction
- logic that depends on external helpers returning structured Python values

## Tradeoffs

Prefer templates when:

- the prompt can be expressed as text plus row placeholders
- you want the lowest per-row overhead
- you want stable, inspectable prompt definitions
- you want to use `row_snapshot(...)` for canonical JSON blocks

Prefer `prompt_udf(...)` when:

- prompt construction requires real Python branching or helper calls
- the payload is assembled from complex nested Python objects
- readability would suffer if you forced the logic into a template string

Costs of templates:

- less flexible for arbitrary Python transformations
- nested object synthesis must come from row fields or static template values

Costs of UDFs:

- more Python work per row
- easier to hide prompt-shape changes inside code
- easier to accidentally drift into ad hoc serialization patterns

## Lower-Level Canonicalization

If you need to inspect the same canonical JSON and hashing behavior used for cache identity, the lower-level helpers are also exposed:

```python
from llm_batch_py._core_wrapper import canonical_json, stable_hash
```

Use these for debugging or tooling. For normal prompt construction, prefer templates and `row_snapshot(...)`.
