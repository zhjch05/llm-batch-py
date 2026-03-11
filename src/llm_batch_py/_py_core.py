from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any

PLACEHOLDER_RE = re.compile(r"{{\s*(.*?)\s*}}")
ROW_FIELD_RE = re.compile(r"^row\.([A-Za-z_][A-Za-z0-9_]*)$")
ROW_SNAPSHOT_RE = re.compile(r"^row_snapshot\((.*)\)$")
ROW_SNAPSHOT_ARG_RE = re.compile(r"(include|exclude|priority)\s*=\s*(\[[^\]]*\])")


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "model_dump"):
        return _normalize(value.model_dump(mode="json"))
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except TypeError:
            return str(value)
    return value


def canonical_json(value: Any) -> str:
    normalized = _normalize(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def jsonl_dump_bytes(items: list[Any]) -> bytes:
    return b"".join(f"{canonical_json(item)}\n".encode() for item in items)


def render_template_string(template: str, row: Any) -> str:
    def replace(match: re.Match[str]) -> str:
        return _render_expr(match.group(1), row)

    return PLACEHOLDER_RE.sub(replace, template)


def evaluate_template_expr(expr: str, row: Any) -> Any:
    base_expr, filters = _split_filters(expr)
    value = _evaluate_base_expr(base_expr, row)
    if not filters:
        return value
    rendered = "" if value is None else str(value)
    for filter_name in filters:
        if filter_name == "upper":
            rendered = rendered.upper()
        elif filter_name == "lower":
            rendered = rendered.lower()
        else:
            raise ValueError(f"Unsupported template filter: {filter_name}")
    return rendered


def extract_template_fields(template: str) -> list[str]:
    columns: set[str] = set()
    for expr in PLACEHOLDER_RE.findall(template):
        _collect_expr_fields(expr, columns)
    return sorted(columns)


def _render_expr(expr: str, row: Any) -> str:
    value = evaluate_template_expr(expr, row)
    return "" if value is None else str(value)


def _evaluate_base_expr(expr: str, row: Any) -> Any:
    stripped = expr.strip()
    field_match = ROW_FIELD_RE.match(stripped)
    if field_match:
        return _row_get(row, field_match.group(1))
    snapshot_match = ROW_SNAPSHOT_RE.match(stripped)
    if snapshot_match:
        config = _parse_snapshot_args(snapshot_match.group(1))
        return _row_snapshot_json(row, **config)
    raise ValueError(f"Unsupported template expression: {stripped}")


def _split_filters(expr: str) -> tuple[str, list[str]]:
    segments = [segment.strip() for segment in expr.split("|")]
    return segments[0], [segment for segment in segments[1:] if segment]


def _collect_expr_fields(expr: str, columns: set[str]) -> None:
    base_expr, _filters = _split_filters(expr)
    stripped = base_expr.strip()
    field_match = ROW_FIELD_RE.match(stripped)
    if field_match:
        columns.add(field_match.group(1))
        return
    snapshot_match = ROW_SNAPSHOT_RE.match(stripped)
    if snapshot_match:
        config = _parse_snapshot_args(snapshot_match.group(1))
        columns.update(config["include"])
        return
    raise ValueError(f"Unsupported template expression: {stripped}")


def _parse_snapshot_args(args: str) -> dict[str, list[str]]:
    config = {"include": [], "exclude": [], "priority": []}
    if not args.strip():
        return config
    consumed = []
    for match in ROW_SNAPSHOT_ARG_RE.finditer(args):
        key = match.group(1)
        value = match.group(2)
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError(f"row_snapshot {key} must be a list[str]")
        config[key] = parsed
        consumed.append(match.group(0))
    remainder = args
    for part in consumed:
        remainder = remainder.replace(part, "", 1)
    if remainder.replace(",", "").strip():
        raise ValueError(f"Unsupported row_snapshot arguments: {args}")
    return config


def _row_snapshot_json(
    row: Any,
    *,
    include: list[str],
    exclude: list[str],
    priority: list[str],
) -> str:
    row_dict = _row_to_dict(row)
    selected_keys = include or list(row_dict.keys())
    selected = {
        key: row_dict[key]
        for key in selected_keys
        if key in row_dict and key not in exclude
    }
    ordered_keys: list[str] = []
    for key in priority:
        if key in selected and key not in ordered_keys:
            ordered_keys.append(key)
    for key in sorted(selected):
        if key not in ordered_keys:
            ordered_keys.append(key)
    parts = [
        f"{json.dumps(key, ensure_ascii=False)}:{canonical_json(selected[key])}"
        for key in ordered_keys
    ]
    return "{" + ",".join(parts) + "}"


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        if key not in row:
            raise ValueError(f"Missing row field referenced in template: {key}")
        return row[key]
    return row[key]


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "items"):
        return dict(row.items())
    return dict(row)
