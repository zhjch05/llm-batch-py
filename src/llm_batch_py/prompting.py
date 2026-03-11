from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from llm_batch_py._core_wrapper import (
    evaluate_template_expr,
    extract_template_fields,
    render_template_string,
    stable_hash,
)

PromptReturn = TypeVar("PromptReturn")
FULL_PLACEHOLDER_RE = re.compile(r"^\s*{{\s*(.*?)\s*}}\s*$")


@dataclass(frozen=True)
class RowSnapshotConfig:
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    priority: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "include": list(self.include),
            "exclude": list(self.exclude),
            "priority": list(self.priority),
        }


@dataclass(frozen=True)
class RowContext(Mapping[str, Any]):
    data: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.data[name]
        except KeyError as exc:  # pragma: no cover - normal attribute semantics.
            raise AttributeError(name) from exc

    def as_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class PromptBuilder(Generic[PromptReturn]):
    func: Callable[[RowContext], PromptReturn]
    version: str
    name: str
    referenced_columns: tuple[str, ...] = ()

    def __call__(self, row: Mapping[str, Any]) -> PromptReturn:
        return self.func(RowContext(row))


@dataclass(frozen=True)
class PromptTemplate(Generic[PromptReturn]):
    kind: Literal["structured", "embedding"]
    template: Any
    version: str
    name: str
    referenced_columns: tuple[str, ...]

    def __call__(self, row: Mapping[str, Any]) -> PromptReturn:
        rendered = _render_template_value(self.template, row)
        return rendered


PromptSpec = PromptTemplate[PromptReturn] | PromptBuilder[PromptReturn]


def structured_template(
    *,
    messages: Sequence[Mapping[str, Any]] | str,
    expected: Mapping[str, Any] | None = None,
    system: str | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    max_tokens: int | None = None,
    name: str = "structured_template",
    version: str | None = None,
) -> PromptTemplate[dict[str, Any]]:
    normalized_messages: list[dict[str, Any]]
    if isinstance(messages, str):
        normalized_messages = [{"role": "user", "content": messages}]
    else:
        normalized_messages = [dict(message) for message in messages]
    template: dict[str, Any] = {"messages": normalized_messages}
    if expected is not None:
        template["expected"] = dict(expected)
    if system is not None:
        template["system"] = system
    if tools is not None:
        template["tools"] = [dict(tool) for tool in tools]
    if max_tokens is not None:
        template["max_tokens"] = max_tokens
    return PromptTemplate(
        kind="structured",
        template=template,
        version=version or _derive_template_version("structured", template),
        name=name,
        referenced_columns=_collect_referenced_columns(template),
    )


def embedding_template(
    template: str,
    *,
    name: str = "embedding_template",
    version: str | None = None,
) -> PromptTemplate[str]:
    return PromptTemplate(
        kind="embedding",
        template=template,
        version=version or _derive_template_version("embedding", template),
        name=name,
        referenced_columns=_collect_referenced_columns(template),
    )


def prompt_udf(
    version: str | None = None,
) -> Callable[[Callable[[RowContext], PromptReturn]], PromptBuilder[PromptReturn]]:
    def decorator(func: Callable[[RowContext], PromptReturn]) -> PromptBuilder[PromptReturn]:
        return PromptBuilder(
            func=func,
            version=version or _derive_udf_version(func),
            name=getattr(func, "__qualname__", func.__name__),
        )

    return decorator


def _render_template_value(value: Any, row: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        full_match = FULL_PLACEHOLDER_RE.match(value)
        if full_match and "{{" not in full_match.group(1) and "}}" not in full_match.group(1):
            return evaluate_template_expr(full_match.group(1), row)
        return render_template_string(value, row)
    if isinstance(value, list):
        return [_render_template_value(item, row) for item in value]
    if isinstance(value, tuple):
        return tuple(_render_template_value(item, row) for item in value)
    if isinstance(value, dict):
        return {str(key): _render_template_value(item, row) for key, item in value.items()}
    return value


def _collect_referenced_columns(value: Any) -> tuple[str, ...]:
    columns: set[str] = set()
    _collect_columns_into(value, columns)
    return tuple(sorted(columns))


def _collect_columns_into(value: Any, columns: set[str]) -> None:
    if isinstance(value, str):
        columns.update(extract_template_fields(value))
        return
    if isinstance(value, list | tuple):
        for item in value:
            _collect_columns_into(item, columns)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_columns_into(item, columns)


def _derive_template_version(kind: str, template: Any) -> str:
    return stable_hash({"kind": kind, "template": template})[:12]


def _derive_udf_version(func: Callable[..., Any]) -> str:
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = f"{func.__module__}:{getattr(func, '__qualname__', func.__name__)}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
