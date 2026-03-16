from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from unittest.mock import Mock, patch

import polars as pl
import pytest

from llm_batch_py.catalog import MANIFEST_BATCHES, MANIFEST_RESULTS, ParquetCatalog
from llm_batch_py.jobs import LockConfig, ResultCacheStoreConfig


def test_catalog_round_trips_local_manifest(tmp_path) -> None:
    catalog = ParquetCatalog(ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")))
    catalog.append_manifest(
        MANIFEST_BATCHES,
        [
            {
                "event_at": "2025-01-01T00:00:00+00:00",
                "job_name": "job",
                "batch_id": "b1",
                "status": "submitted",
            }
        ],
    )

    frame = catalog.read_manifest(MANIFEST_BATCHES)

    assert frame.shape == (1, 4)
    assert frame.to_dicts()[0]["batch_id"] == "b1"
    manifest_paths = catalog._manifest_paths(MANIFEST_BATCHES)
    assert len(manifest_paths) == 1
    assert PurePosixPath(manifest_paths[0]).suffix == ".parquet"
    assert PurePosixPath(manifest_paths[0]).parent.name == MANIFEST_BATCHES


def test_catalog_round_trips_memory_manifest() -> None:
    catalog = ParquetCatalog(ResultCacheStoreConfig(root_uri="memory://llm_batch_py-tests/catalog"))
    catalog.append_manifest(
        MANIFEST_BATCHES,
        [
            {
                "event_at": "2025-01-01T00:00:00+00:00",
                "job_name": "job",
                "batch_id": "b2",
                "status": "submitted",
            }
        ],
    )

    frame = catalog.read_manifest(MANIFEST_BATCHES)

    assert frame.shape == (1, 4)
    assert frame.to_dicts()[0]["batch_id"] == "b2"


def test_catalog_passes_storage_options_to_url_to_fs() -> None:
    fs = Mock()
    fs.protocol = "memory"

    with patch(
        "llm_batch_py.catalog.fsspec.core.url_to_fs",
        return_value=(fs, "bucket/prefix"),
    ) as mock_url_to_fs:
        catalog = ParquetCatalog(
            ResultCacheStoreConfig(
                root_uri="s3://bucket/prefix",
                storage_options={
                    "profile": "analytics",
                    "client_kwargs": {"region_name": "us-west-2"},
                },
            )
        )

    mock_url_to_fs.assert_called_once_with(
        "s3://bucket/prefix",
        profile="analytics",
        client_kwargs={"region_name": "us-west-2"},
    )
    assert catalog.fs is fs
    assert catalog.base_path == "bucket/prefix"


def test_catalog_reads_append_only_manifest_chunks(tmp_path) -> None:
    catalog = ParquetCatalog(ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")))
    catalog.append_manifest(
        MANIFEST_BATCHES,
        [
            {
                "event_at": "2025-01-01T00:00:00+00:00",
                "job_name": "job",
                "batch_id": "b1",
                "status": "submitted",
            }
        ],
    )
    catalog.append_manifest(
        MANIFEST_BATCHES,
        [
            {
                "event_at": "2025-01-01T00:01:00+00:00",
                "job_name": "job",
                "batch_id": "b2",
                "status": "completed",
            }
        ],
    )

    frame = catalog.read_manifest(MANIFEST_BATCHES).sort("event_at")

    assert frame["batch_id"].to_list() == ["b1", "b2"]
    assert len(catalog._manifest_paths(MANIFEST_BATCHES)) == 2


def test_catalog_appends_result_rows_when_nullable_fields_become_populated_late(tmp_path) -> None:
    catalog = ParquetCatalog(ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")))
    rows = [
        {
            "event_at": "2025-01-01T00:00:00+00:00",
            "job_name": "job",
            "batch_id": "batch-1",
            "request_id": f"request-{index}",
            "custom_id": f"custom-{index}",
            "cache_key": f"cache-{index}",
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "endpoint_kind": "structured_output",
            "status": "completed",
            "error_code": None,
            "row_key_json": '{"product_id":"p-1"}',
            "parsed_json": '{"best_name":"Widget"}',
            "raw_json": '{"result":{"type":"succeeded"}}',
            "raw_output_text": '{"best_name":"Widget"}',
            "input_tokens": 12,
            "output_tokens": 4,
        }
        for index in range(100)
    ]
    rows.append(
        {
            "event_at": "2025-01-01T00:01:00+00:00",
            "job_name": "job",
            "batch_id": "batch-1",
            "request_id": "request-100",
            "custom_id": "custom-100",
            "cache_key": "cache-100",
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "endpoint_kind": "structured_output",
            "status": "failed",
            "error_code": "error",
            "row_key_json": '{"product_id":"p-2"}',
            "parsed_json": None,
            "raw_json": '{"result":{"type":"error"}}',
            "raw_output_text": None,
            "input_tokens": None,
            "output_tokens": None,
        }
    )

    catalog.append_manifest(MANIFEST_RESULTS, rows)

    frame = catalog.read_manifest(MANIFEST_RESULTS).sort("request_id")
    failed_row = frame.filter(pl.col("request_id") == "request-100").to_dicts()[0]

    assert frame.height == 101
    assert frame["error_code"].null_count() == 100
    assert failed_row["error_code"] == "error"


def test_catalog_reads_legacy_single_manifest_file(tmp_path) -> None:
    catalog = ParquetCatalog(ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")))
    path = catalog._manifest_path(MANIFEST_BATCHES)
    catalog.fs.makedirs(str(PurePosixPath(path).parent), exist_ok=True)
    with catalog.fs.open(path, "wb") as handle:
        pl.DataFrame(
            [
                {
                    "event_at": "2025-01-01T00:00:00+00:00",
                    "job_name": "job",
                    "batch_id": "legacy",
                    "status": "submitted",
                }
            ]
        ).write_parquet(handle)
    catalog.append_manifest(
        MANIFEST_BATCHES,
        [
            {
                "event_at": "2025-01-01T00:01:00+00:00",
                "job_name": "job",
                "batch_id": "new",
                "status": "completed",
            }
        ],
    )

    frame = catalog.read_manifest(MANIFEST_BATCHES).sort("event_at")

    assert frame["batch_id"].to_list() == ["legacy", "new"]
    assert len(catalog._manifest_paths(MANIFEST_BATCHES)) == 2


def test_release_lock_preserves_newer_owner(tmp_path) -> None:
    catalog = ParquetCatalog(ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")))
    lock = catalog.acquire_lock("job", "run-1")
    lock_path = catalog._strip_protocol(lock.path)

    with catalog.fs.open(lock_path, "wb") as handle:
        handle.write(
            json.dumps(
                {
                    "run_id": "run-2",
                    "acquired_at": "2025-01-01T00:00:00+00:00",
                }
            ).encode("utf-8")
        )

    catalog.release_lock(lock)

    assert catalog.fs.exists(lock_path)
    with catalog.fs.open(lock_path, "rb") as handle:
        payload = json.loads(handle.read().decode("utf-8"))
    assert payload["run_id"] == "run-2"


def test_acquire_lock_reclaims_stale_entry(tmp_path) -> None:
    catalog = ParquetCatalog(
        ResultCacheStoreConfig(root_uri=str(tmp_path / "catalog")),
        LockConfig(ttl_seconds=30),
    )
    lock_path = catalog._path("locks", "job.json")
    catalog.fs.makedirs(str(PurePosixPath(lock_path).parent), exist_ok=True)
    stale_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    with catalog.fs.open(lock_path, "wb") as handle:
        handle.write(json.dumps({"run_id": "stale-run", "acquired_at": stale_at}).encode("utf-8"))

    lock = catalog.acquire_lock("job", "run-2")

    assert lock.run_id == "run-2"
    with catalog.fs.open(lock_path, "rb") as handle:
        payload = json.loads(handle.read().decode("utf-8"))
    assert payload["run_id"] == "run-2"


def test_catalog_rejects_unsafe_s3_compatible_locking() -> None:
    fs = Mock()
    fs.protocol = "s3"

    with patch(
        "llm_batch_py.catalog.fsspec.core.url_to_fs",
        return_value=(fs, "bucket/prefix"),
    ):
        catalog = ParquetCatalog(
            ResultCacheStoreConfig(
                root_uri="s3://bucket/prefix",
                storage_options={
                    "key": "access",
                    "secret": "secret",
                    "client_kwargs": {"endpoint_url": "https://minio.internal.example"},
                },
            )
        )

    with pytest.raises(RuntimeError, match="allow_unsafe_s3_compatible_locks=True"):
        catalog.acquire_lock("job", "run-1")


def test_catalog_allows_opt_in_for_s3_compatible_locking() -> None:
    fs = Mock()
    fs.protocol = "s3"
    fs.makedirs = Mock()
    fs.open.return_value = io.BytesIO()

    with patch(
        "llm_batch_py.catalog.fsspec.core.url_to_fs",
        return_value=(fs, "bucket/prefix"),
    ):
        catalog = ParquetCatalog(
            ResultCacheStoreConfig(
                root_uri="s3://bucket/prefix",
                storage_options={
                    "key": "access",
                    "secret": "secret",
                    "client_kwargs": {"endpoint_url": "https://minio.internal.example"},
                },
            ),
            LockConfig(allow_unsafe_s3_compatible_locks=True),
        )

    lock = catalog.acquire_lock("job", "run-1")

    assert lock.run_id == "run-1"
    fs.makedirs.assert_called_once()
    fs.open.assert_called_once()


def test_catalog_allows_known_aws_s3_endpoint_locking() -> None:
    fs = Mock()
    fs.protocol = "s3"
    fs.makedirs = Mock()
    fs.open.return_value = io.BytesIO()

    with patch(
        "llm_batch_py.catalog.fsspec.core.url_to_fs",
        return_value=(fs, "bucket/prefix"),
    ):
        catalog = ParquetCatalog(
            ResultCacheStoreConfig(
                root_uri="s3://bucket/prefix",
                storage_options={
                    "key": "access",
                    "secret": "secret",
                    "client_kwargs": {
                        "endpoint_url": "https://s3.us-west-2.amazonaws.com",
                        "region_name": "us-west-2",
                    },
                },
            )
        )

    lock = catalog.acquire_lock("job", "run-1")

    assert lock.run_id == "run-1"
    fs.makedirs.assert_called_once()
    fs.open.assert_called_once()
