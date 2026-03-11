from __future__ import annotations

from types import MappingProxyType

import polars as pl
import pytest
from pydantic import BaseModel

from llm_batch_py import (
    CacheStoreConfig,
    EmbeddingJob,
    OpenAIConfig,
    PromptCacheConfig,
    ResultCacheConfig,
    ResultCacheStoreConfig,
    RowContext,
    StructuredOutputJob,
    embedding_template,
    prompt_udf,
    structured_template,
)
from llm_batch_py.jobs import validate_job_input_frame


def test_structured_template_exposes_row_fields_and_version() -> None:
    build_prompt = structured_template(
        messages=[{"role": "user", "content": "{{ row.title }}:{{ row.body }}"}],
        name="build_prompt",
        version="custom-v1",
    )

    payload = build_prompt({"title": "Hello", "body": "World"})

    assert build_prompt.version == "custom-v1"
    assert payload["messages"][0]["content"] == "Hello:World"


def test_row_snapshot_supports_priority_and_filters() -> None:
    build_prompt = structured_template(
        messages=[
            {
                "role": "user",
                "content": '{{ row_snapshot(include=["body", "title"], priority=["title"]) }}',
            }
        ],
    )

    payload = build_prompt({"title": "Hello", "body": "World", "ignored": "x"})

    assert payload["messages"][0]["content"] == '{"title":"Hello","body":"World"}'


def test_embedding_template_renders_text() -> None:
    template = embedding_template("Embed {{ row.text | upper }}")

    assert template({"text": "hello"}) == "Embed HELLO"


def test_structured_template_accepts_mapping_rows() -> None:
    build_prompt = structured_template(
        messages=[{"role": "user", "content": "{{ row.title }}"}],
        expected={"title": "{{ row.title }}"},
    )

    payload = build_prompt(RowContext(MappingProxyType({"title": "Hello"})))

    assert payload["messages"][0]["content"] == "Hello"
    assert payload["expected"] == {"title": "Hello"}


def test_structured_template_preserves_full_placeholder_types() -> None:
    build_prompt = structured_template(
        messages=[{"role": "user", "content": "Summarize"}],
        expected={
            "id": "{{ row.id }}",
            "enabled": "{{ row.enabled }}",
            "tags": "{{ row.tags }}",
            "meta": "{{ row.meta }}",
        },
        max_tokens="{{ row.limit }}",
    )

    payload = build_prompt(
        {
            "id": 7,
            "enabled": True,
            "tags": ["a", "b"],
            "meta": {"score": 3},
            "limit": 128,
        }
    )

    assert payload["expected"] == {
        "id": 7,
        "enabled": True,
        "tags": ["a", "b"],
        "meta": {"score": 3},
    }
    assert payload["max_tokens"] == 128


class _LabelOutput(BaseModel):
    label: str


def test_row_snapshot_exclude_and_priority_do_not_require_missing_columns() -> None:
    build_prompt = structured_template(
        messages=[
            {
                "role": "user",
                "content": '{{ row_snapshot(exclude=["debug_only"], priority=["title"]) }}',
            }
        ],
        version="v1",
    )

    job = StructuredOutputJob(
        name="labels",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "title": ["Hello"]}),
        prompt_builder=build_prompt,
        output_model=_LabelOutput,
        provider=OpenAIConfig(model="gpt-4o-mini"),
        result_cache=ResultCacheStoreConfig(root_uri="./.llm_batch_py"),
    )

    validated = validate_job_input_frame(job, job.materialized_input())

    assert validated.shape == (1, 2)


def test_row_snapshot_excluded_include_columns_are_not_required() -> None:
    build_prompt = structured_template(
        messages=[
            {
                "role": "user",
                "content": (
                    '{{ row_snapshot(include=["title", "debug_only"], '
                    'exclude=["debug_only"]) }}'
                ),
            }
        ],
        version="v1",
    )

    job = StructuredOutputJob(
        name="labels",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "title": ["Hello"]}),
        prompt_builder=build_prompt,
        output_model=_LabelOutput,
        provider=OpenAIConfig(model="gpt-4o-mini"),
        result_cache=ResultCacheStoreConfig(root_uri="./.llm_batch_py"),
    )

    validated = validate_job_input_frame(job, job.materialized_input())

    assert validated.shape == (1, 2)


def test_prompt_udf_remains_supported_as_fallback() -> None:
    @prompt_udf(version="compat-v1")
    def build_prompt(row):
        return {"messages": [{"role": "user", "content": f"{row.title}:{row['body']}"}]}

    payload = build_prompt({"title": "Hello", "body": "World"})

    assert build_prompt.version == "compat-v1"
    assert payload["messages"][0]["content"] == "Hello:World"


def test_result_cache_store_config_is_importable_from_public_api() -> None:
    build_prompt = structured_template(
        messages=[{"role": "user", "content": "{{ row.text }}"}],
        expected={"label": "{{ row.text | upper }}"},
        version="v1",
    )

    job = StructuredOutputJob(
        name="labels",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "text": ["hello"]}),
        prompt_builder=build_prompt,
        output_model=_LabelOutput,
        provider=OpenAIConfig(model="gpt-4o-mini"),
        result_cache=ResultCacheStoreConfig(
            root_uri="./.llm_batch_py",
            storage_options={"profile": "test-profile"},
        ),
    )

    assert job.result_cache.root_uri == "./.llm_batch_py"
    assert job.result_cache.storage_options == {"profile": "test-profile"}
    assert job.cache_store.root_uri == "./.llm_batch_py"
    assert job.cache_store.storage_options == {"profile": "test-profile"}


def test_result_cache_config_alias_remains_supported() -> None:
    config = ResultCacheConfig(
        root_uri="./.llm_batch_py",
        storage_options={"client_kwargs": {"region_name": "us-west-2"}},
    )

    assert isinstance(config, ResultCacheStoreConfig)
    assert config.root_uri == "./.llm_batch_py"
    assert config.storage_options == {"client_kwargs": {"region_name": "us-west-2"}}


def test_cache_store_alias_remains_supported() -> None:
    config = CacheStoreConfig(
        root_uri="./.llm_batch_py",
        storage_options={"anon": False},
    )

    assert isinstance(config, ResultCacheStoreConfig)
    assert config.root_uri == "./.llm_batch_py"
    assert config.storage_options == {"anon": False}


def test_job_accepts_cache_store_alias() -> None:
    structured_job = StructuredOutputJob(
        name="labels",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "text": ["hello"]}),
        prompt_builder=structured_template(
            messages=[{"role": "user", "content": "{{ row.text }}"}],
            version="v1",
        ),
        output_model=_LabelOutput,
        provider=OpenAIConfig(model="gpt-4o-mini"),
        cache_store=ResultCacheStoreConfig(root_uri="./.llm_batch_py"),
    )
    embedding_job = EmbeddingJob(
        name="embeddings",
        key_cols=["id"],
        input_df=pl.DataFrame({"id": [1], "text": ["hello"]}),
        text_builder=embedding_template("{{ row.text }}", version="v1"),
        provider=OpenAIConfig(model="text-embedding-3-small"),
        cache_store=ResultCacheStoreConfig(root_uri="./.llm_batch_py-emb"),
    )

    assert structured_job.result_cache.root_uri == "./.llm_batch_py"
    assert structured_job.cache_store.root_uri == "./.llm_batch_py"
    assert embedding_job.result_cache.root_uri == "./.llm_batch_py-emb"
    assert embedding_job.cache_store.root_uri == "./.llm_batch_py-emb"


def test_job_rejects_conflicting_cache_arguments() -> None:
    with pytest.raises(ValueError, match="result_cache and cache_store must match"):
        StructuredOutputJob(
            name="labels",
            key_cols=["id"],
            input_df=pl.DataFrame({"id": [1], "text": ["hello"]}),
            prompt_builder=structured_template(
                messages=[{"role": "user", "content": "{{ row.text }}"}],
                version="v1",
            ),
            output_model=_LabelOutput,
            provider=OpenAIConfig(model="gpt-4o-mini"),
            result_cache=ResultCacheStoreConfig(root_uri="./one"),
            cache_store=ResultCacheStoreConfig(root_uri="./two"),
        )


def test_prompt_cache_config_defaults_to_auto() -> None:
    assert PromptCacheConfig().mode == "auto"
