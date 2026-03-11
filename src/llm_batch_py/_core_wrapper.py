from __future__ import annotations

try:
    from llm_batch_py._core import (
        canonical_json,
        evaluate_template_expr,
        extract_template_fields,
        jsonl_dump_bytes,
        render_template_string,
        stable_hash,
    )
except ImportError:  # pragma: no cover - exercised only when the Rust module is unavailable.
    from llm_batch_py._py_core import (
        canonical_json,
        evaluate_template_expr,
        extract_template_fields,
        jsonl_dump_bytes,
        render_template_string,
        stable_hash,
    )

__all__ = [
    "canonical_json",
    "evaluate_template_expr",
    "extract_template_fields",
    "jsonl_dump_bytes",
    "render_template_string",
    "stable_hash",
]
