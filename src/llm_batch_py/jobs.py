from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Generic, Literal, TypeVar

import polars as pl
from pydantic import BaseModel

from llm_batch_py.pricing import DEFAULT_PRICING, ModelPricing
from llm_batch_py.prompting import PromptBuilder, PromptTemplate

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)
InputTokenEstimation = Literal["auto", "exact", "skip"]
ResolvedInputTokenEstimation = Literal["exact", "skip"]


@dataclass(frozen=True)
class ResultCacheStoreConfig:
    root_uri: str
    storage_options: dict[str, Any] | None = None


CacheStoreConfig = ResultCacheStoreConfig
ResultCacheConfig = ResultCacheStoreConfig


@dataclass(frozen=True)
class LockConfig:
    ttl_seconds: int = 3600


@dataclass(frozen=True)
class BatchConfig:
    batch_size: int | None = None
    max_retries: int = 2


@dataclass(frozen=True)
class PromptCacheConfig:
    mode: Literal["auto", "off"] = "auto"
    verbose: bool = False


@dataclass(frozen=True)
class AnthropicConfig:
    model: str
    api_key: str | None = None
    max_output_tokens: int = 1024
    temperature: float | None = None
    timeout: float | None = 60.0
    pricing: ModelPricing | None = None
    input_token_estimation: InputTokenEstimation = "auto"
    provider_name: Literal["anthropic"] = "anthropic"


@dataclass(frozen=True)
class OpenAIConfig:
    model: str
    api_key: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    timeout: float | None = 60.0
    organization: str | None = None
    base_url: str | None = None
    dimensions: int | None = None
    pricing: ModelPricing | None = None
    provider_name: Literal["openai"] = "openai"


ProviderConfig = AnthropicConfig | OpenAIConfig


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    total_rows: int
    result_cache_hits: int
    inflight_rows: int
    submitted_rows: int
    completed_rows: int
    failed_rows: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float | None
    input_token_estimation: ResolvedInputTokenEstimation
    batches_submitted: int
    dry_run: bool

    @property
    def cache_hits(self) -> int:
        return self.result_cache_hits


@dataclass(frozen=True)
class StreamRunSummary:
    chunk_count: int
    total_rows: int
    result_cache_hits: int
    inflight_rows: int
    submitted_rows: int
    completed_rows: int
    failed_rows: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float | None
    batches_submitted: int
    dry_run: bool


def _resolve_cache_store(
    result_cache: ResultCacheStoreConfig | None,
    cache_store: ResultCacheStoreConfig | None,
) -> ResultCacheStoreConfig:
    if result_cache is None and cache_store is None:
        raise TypeError("Either result_cache or cache_store is required")
    if result_cache is not None and cache_store is not None and result_cache != cache_store:
        raise ValueError("result_cache and cache_store must match when both are provided")
    if result_cache is not None:
        return result_cache
    return cache_store


@dataclass(frozen=True, init=False)
class StructuredOutputJob(Generic[OutputModelT]):
    name: str
    key_cols: Sequence[str]
    input_df: pl.DataFrame | pl.LazyFrame
    prompt_builder: PromptBuilder[Any] | PromptTemplate[dict[str, Any]]
    output_model: type[OutputModelT]
    provider: ProviderConfig
    result_cache: ResultCacheStoreConfig
    lock: LockConfig = field(default_factory=LockConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    prompt_cache: PromptCacheConfig | None = None
    output_column_prefix: str | None = None

    def __init__(
        self,
        name: str,
        key_cols: Sequence[str],
        input_df: pl.DataFrame | pl.LazyFrame,
        prompt_builder: PromptBuilder[Any] | PromptTemplate[dict[str, Any]],
        output_model: type[OutputModelT],
        provider: ProviderConfig,
        result_cache: ResultCacheStoreConfig | None = None,
        *,
        cache_store: ResultCacheStoreConfig | None = None,
        lock: LockConfig | None = None,
        batch: BatchConfig | None = None,
        prompt_cache: PromptCacheConfig | None = None,
        output_column_prefix: str | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "key_cols", key_cols)
        object.__setattr__(self, "input_df", input_df)
        object.__setattr__(self, "prompt_builder", prompt_builder)
        object.__setattr__(self, "output_model", output_model)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(
            self,
            "result_cache",
            _resolve_cache_store(result_cache, cache_store),
        )
        object.__setattr__(self, "lock", lock or LockConfig())
        object.__setattr__(self, "batch", batch or BatchConfig())
        object.__setattr__(self, "prompt_cache", prompt_cache)
        object.__setattr__(self, "output_column_prefix", output_column_prefix)

    def materialized_input(self) -> pl.DataFrame:
        return self.input_df.collect() if isinstance(self.input_df, pl.LazyFrame) else self.input_df

    @property
    def endpoint_kind(self) -> str:
        return "structured"

    @property
    def pricing(self) -> ModelPricing | None:
        return self.provider.pricing or DEFAULT_PRICING.get(self.provider.model)

    @property
    def cache_store(self) -> ResultCacheStoreConfig:
        return self.result_cache


@dataclass(frozen=True, init=False)
class EmbeddingJob:
    name: str
    key_cols: Sequence[str]
    input_df: pl.DataFrame | pl.LazyFrame
    text_builder: PromptBuilder[str] | PromptTemplate[str]
    provider: OpenAIConfig
    result_cache: ResultCacheStoreConfig
    lock: LockConfig = field(default_factory=LockConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)

    def __init__(
        self,
        name: str,
        key_cols: Sequence[str],
        input_df: pl.DataFrame | pl.LazyFrame,
        text_builder: PromptBuilder[str] | PromptTemplate[str],
        provider: OpenAIConfig,
        result_cache: ResultCacheStoreConfig | None = None,
        *,
        cache_store: ResultCacheStoreConfig | None = None,
        lock: LockConfig | None = None,
        batch: BatchConfig | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "key_cols", key_cols)
        object.__setattr__(self, "input_df", input_df)
        object.__setattr__(self, "text_builder", text_builder)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(
            self,
            "result_cache",
            _resolve_cache_store(result_cache, cache_store),
        )
        object.__setattr__(self, "lock", lock or LockConfig())
        object.__setattr__(self, "batch", batch or BatchConfig())

    def materialized_input(self) -> pl.DataFrame:
        return self.input_df.collect() if isinstance(self.input_df, pl.LazyFrame) else self.input_df

    @property
    def endpoint_kind(self) -> str:
        return "embeddings"

    @property
    def pricing(self) -> ModelPricing | None:
        return self.provider.pricing or DEFAULT_PRICING.get(self.provider.model)

    @property
    def cache_store(self) -> ResultCacheStoreConfig:
        return self.result_cache


Job = StructuredOutputJob[Any] | EmbeddingJob

LLM_BATCH_PY_RESULT_METADATA_COLUMN_ORDER = (
    "llm_batch_py_status",
    "llm_batch_py_batch_id",
    "llm_batch_py_provider",
    "llm_batch_py_model",
    "llm_batch_py_input_tokens",
    "llm_batch_py_output_tokens",
    "llm_batch_py_input_raw_json",
    "llm_batch_py_request_raw_json",
    "llm_batch_py_output_raw_json",
    "llm_batch_py_output_raw_text",
    "llm_batch_py_error_code",
    "llm_batch_py_error_raw_json",
    "llm_batch_py_result_cached",
    "llm_batch_py_cached",
    "llm_batch_py_updated_at",
)
LLM_BATCH_PY_RESULT_METADATA_COLUMNS = frozenset(LLM_BATCH_PY_RESULT_METADATA_COLUMN_ORDER)


@cache
def output_model_json_schema(output_model: type[BaseModel]) -> dict[str, Any]:
    return output_model.model_json_schema()


def structured_output_result_column_map(job: StructuredOutputJob[Any]) -> dict[str, str]:
    prefix = job.output_column_prefix or ""
    return {
        field_name: f"{prefix}{field_name}"
        for field_name in job.output_model.model_fields
    }


def validate_job_input(job: Job) -> pl.DataFrame:
    return validate_job_input_frame(job, job.materialized_input())


def validate_job_input_frame(job: Job, df: pl.DataFrame) -> pl.DataFrame:
    validate_job_input_columns(job, df.columns)
    if df.select(pl.struct(list(job.key_cols)).is_duplicated().any()).item():
        raise ValueError("key_cols must uniquely identify each input row")
    return df


def validate_job_input_columns(job: Job, columns: Sequence[str]) -> None:
    column_names = list(columns)
    missing_key_cols = [column for column in job.key_cols if column not in column_names]
    if missing_key_cols:
        raise ValueError(f"Missing key columns: {missing_key_cols}")

    if isinstance(job, EmbeddingJob):
        return

    output_columns = list(structured_output_result_column_map(job).values())
    colliding_metadata = sorted(set(output_columns) & LLM_BATCH_PY_RESULT_METADATA_COLUMNS)
    if colliding_metadata:
        raise ValueError(
            "Structured output fields collide with reserved llm_batch_py columns: "
            f"{colliding_metadata}"
        )

    colliding_inputs = sorted((set(output_columns) & set(column_names)) - set(job.key_cols))
    if colliding_inputs:
        if job.output_column_prefix:
            raise ValueError(
                f"Prefixed structured output fields collide with input columns: {colliding_inputs}"
            )
        raise ValueError(f"Structured output fields collide with input columns: {colliding_inputs}")


def _required_input_columns(job: Job) -> list[str]:
    return sorted(set(job.key_cols))


def resolve_result_metadata_columns(metadata_columns: Sequence[str] | None) -> tuple[str, ...]:
    if metadata_columns is None:
        return LLM_BATCH_PY_RESULT_METADATA_COLUMN_ORDER

    resolved: list[str] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for column in metadata_columns:
        if column not in LLM_BATCH_PY_RESULT_METADATA_COLUMNS:
            unknown.append(column)
            continue
        if column in seen:
            continue
        seen.add(column)
        resolved.append(column)

    if unknown:
        raise ValueError(f"Unknown llm_batch_py metadata columns: {unknown}")
    return tuple(resolved)
