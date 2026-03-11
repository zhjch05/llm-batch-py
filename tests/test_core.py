from __future__ import annotations

from datetime import datetime, timezone

from llm_batch_py._core_wrapper import canonical_json, jsonl_dump_bytes, stable_hash
from llm_batch_py._py_core import (
    canonical_json as py_canonical_json,
)
from llm_batch_py._py_core import (
    jsonl_dump_bytes as py_jsonl_dump_bytes,
)
from llm_batch_py._py_core import (
    stable_hash as py_stable_hash,
)


def test_rust_core_matches_python_fallback() -> None:
    payload = {"b": [3, 2, {"x": True}], "a": "hello"}

    assert canonical_json(payload) == py_canonical_json(payload)
    assert stable_hash(payload) == py_stable_hash(payload)
    assert jsonl_dump_bytes([payload, payload]) == py_jsonl_dump_bytes([payload, payload])


def test_rust_core_matches_python_fallback_for_datetime_payloads() -> None:
    payload = {"created_at": datetime(2025, 1, 1, tzinfo=timezone.utc)}

    assert canonical_json(payload) == py_canonical_json(payload)
    assert stable_hash(payload) == py_stable_hash(payload)
