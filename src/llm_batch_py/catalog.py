from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import fsspec
import polars as pl

from llm_batch_py.jobs import LockConfig, ResultCacheStoreConfig

logger = logging.getLogger(__name__)

MANIFEST_RUNS = "runs"
MANIFEST_BATCHES = "batches"
MANIFEST_REQUESTS = "requests"
MANIFEST_RESULTS = "results"

_MANIFEST_SCHEMA_OVERRIDES: dict[str, dict[str, pl.DataType]] = {
    MANIFEST_RUNS: {
        "event_at": pl.String,
        "job_name": pl.String,
        "run_id": pl.String,
        "dry_run": pl.Boolean,
    },
    MANIFEST_BATCHES: {
        "event_at": pl.String,
        "created_at": pl.String,
        "job_name": pl.String,
        "batch_id": pl.String,
        "provider_batch_id": pl.String,
        "provider": pl.String,
        "model": pl.String,
        "endpoint_kind": pl.String,
        "status": pl.String,
        "request_count": pl.Int64,
        "artifact_uri": pl.String,
        "raw_json": pl.String,
        "submit_attempts": pl.Int64,
        "results_ingested_at": pl.String,
        "output_artifact": pl.String,
        "error_artifact": pl.String,
    },
    MANIFEST_REQUESTS: {
        "event_at": pl.String,
        "created_at": pl.String,
        "job_name": pl.String,
        "request_id": pl.String,
        "batch_id": pl.String,
        "custom_id": pl.String,
        "cache_key": pl.String,
        "provider": pl.String,
        "model": pl.String,
        "endpoint_kind": pl.String,
        "row_key_json": pl.String,
        "payload_json": pl.String,
        "transport_record_json": pl.String,
        "prompt_version": pl.String,
        "status": pl.String,
    },
    MANIFEST_RESULTS: {
        "event_at": pl.String,
        "job_name": pl.String,
        "batch_id": pl.String,
        "request_id": pl.String,
        "custom_id": pl.String,
        "cache_key": pl.String,
        "provider": pl.String,
        "model": pl.String,
        "endpoint_kind": pl.String,
        "status": pl.String,
        "error_code": pl.String,
        "row_key_json": pl.String,
        "parsed_json": pl.String,
        "raw_json": pl.String,
        "raw_output_text": pl.String,
        "input_tokens": pl.Int64,
        "output_tokens": pl.Int64,
    },
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


@dataclass(frozen=True)
class LockHandle:
    path: str
    run_id: str
    acquired_at: str


class ParquetCatalog:
    def __init__(
        self, config: ResultCacheStoreConfig, lock_config: LockConfig | None = None
    ) -> None:
        self.config = config
        self.lock_config = lock_config or LockConfig()
        self.fs, base_path = fsspec.core.url_to_fs(
            config.root_uri, **(config.storage_options or {})
        )
        self.base_path = base_path.rstrip("/")

    def _path(self, *parts: str) -> str:
        values = [self.base_path, *parts]
        return str(PurePosixPath(*[value for value in values if value]))

    def _manifest_dir(self, name: str) -> str:
        return self._path("manifests", name)

    def _manifest_path(self, name: str) -> str:
        return self._path("manifests", f"{name}.parquet")

    def read_manifest(self, name: str) -> pl.DataFrame:
        paths = self._manifest_paths(name)
        if not paths:
            return pl.DataFrame()
        frames: list[pl.DataFrame] = []
        read_errors: list[tuple[str, Exception]] = []
        for path in paths:
            try:
                with self.fs.open(path, "rb") as handle:
                    frames.append(pl.read_parquet(handle))
            except Exception as exc:
                read_errors.append((path, exc))
                logger.warning(
                    "Skipping unreadable manifest chunk %s for %s: %s",
                    self._qualify(path),
                    name,
                    exc,
                )
        if not frames:
            if read_errors:
                raise read_errors[0][1]
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]

    def append_manifest(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = self._manifest_chunk_path(name)
        self.fs.makedirs(str(PurePosixPath(path).parent), exist_ok=True)
        with self.fs.open(path, "wb") as handle:
            pl.from_dicts(
                rows,
                schema_overrides=_MANIFEST_SCHEMA_OVERRIDES.get(name),
                infer_schema_length=None,
            ).write_parquet(handle)

    def write_artifact(self, relative_path: str, payload: bytes) -> str:
        path = self._path(relative_path)
        parent = str(PurePosixPath(path).parent)
        self.fs.makedirs(parent, exist_ok=True)
        with self.fs.open(path, "wb") as handle:
            handle.write(payload)
        return self._qualify(path)

    def read_text(self, qualified_path: str) -> str:
        path = self._strip_protocol(qualified_path)
        with self.fs.open(path, "rb") as handle:
            return handle.read().decode("utf-8")

    def acquire_lock(self, job_name: str, run_id: str) -> LockHandle:
        self._validate_lock_backend()
        path = self._path("locks", f"{job_name}.json")
        parent = str(PurePosixPath(path).parent)
        self.fs.makedirs(parent, exist_ok=True)
        for _ in range(3):
            payload = {"run_id": run_id, "acquired_at": utc_now_iso()}
            if self._try_create_lock(path, payload):
                return LockHandle(
                    path=self._qualify(path),
                    run_id=run_id,
                    acquired_at=payload["acquired_at"],
                )

            current = self._read_lock(path)
            if current is None:
                raise RuntimeError(f"Lock already held for job {job_name}")
            acquired_at = datetime.fromisoformat(current["acquired_at"])
            if utc_now() - acquired_at < timedelta(seconds=self.lock_config.ttl_seconds):
                raise RuntimeError(f"Lock already held for job {job_name}")
            if not self._remove_lock(path, current):
                continue

        raise RuntimeError(f"Lock already held for job {job_name}")

    def release_lock(self, lock: LockHandle) -> None:
        path = self._strip_protocol(lock.path)
        current = self._read_lock(path)
        if current is None:
            return
        if current["run_id"] != lock.run_id or current["acquired_at"] != lock.acquired_at:
            return
        try:
            self.fs.rm(path)
        except FileNotFoundError:
            return

    def latest_batches(self, job_name: str) -> pl.DataFrame:
        return self._latest_rows(self.read_manifest(MANIFEST_BATCHES), ["batch_id"], job_name)

    def latest_requests(self, job_name: str) -> pl.DataFrame:
        return self._latest_rows(self.read_manifest(MANIFEST_REQUESTS), ["request_id"], job_name)

    def latest_results(self, job_name: str) -> pl.DataFrame:
        return self._latest_rows(self.read_manifest(MANIFEST_RESULTS), ["cache_key"], job_name)

    def _latest_rows(self, frame: pl.DataFrame, subset: list[str], job_name: str) -> pl.DataFrame:
        if not len(frame):
            return frame
        filtered = frame.filter(pl.col("job_name") == job_name)
        if not len(filtered):
            return filtered
        return filtered.sort("event_at").unique(subset=subset, keep="last", maintain_order=True)

    def _qualify(self, path: str) -> str:
        protocol = self.fs.protocol
        if isinstance(protocol, tuple):
            protocol = protocol[0]
        if protocol == "file":
            return path
        return f"{protocol}://{path}"

    def _strip_protocol(self, path: str) -> str:
        if "://" in path:
            _, _, remainder = path.partition("://")
            return remainder
        return path

    def record_run(self, job_name: str, run_id: str, dry_run: bool) -> None:
        self.append_manifest(
            MANIFEST_RUNS,
            [
                {
                    "event_at": utc_now_iso(),
                    "job_name": job_name,
                    "run_id": run_id,
                    "dry_run": dry_run,
                }
            ],
        )

    def artifact_path(self, *parts: str) -> str:
        return self._path("artifacts", *parts)

    def _manifest_paths(self, name: str) -> list[str]:
        paths = sorted(self.fs.glob(self._path("manifests", name, "*.parquet")))
        legacy_path = self._manifest_path(name)
        if self.fs.exists(legacy_path):
            paths.insert(0, legacy_path)
        return paths

    def _manifest_chunk_path(self, name: str) -> str:
        filename = f"{utc_now().strftime('%Y%m%dT%H%M%S%f')}_{uuid4().hex}.parquet"
        return self._path("manifests", name, filename)

    def _try_create_lock(self, path: str, payload: dict[str, str]) -> bool:
        try:
            with self.fs.open(path, "xb") as handle:
                handle.write(json.dumps(payload).encode("utf-8"))
        except FileExistsError:
            return False
        return True

    def _read_lock(self, path: str) -> dict[str, str] | None:
        if not self.fs.exists(path):
            return None
        with self.fs.open(path, "rb") as handle:
            payload = handle.read()
        if not payload:
            return None
        try:
            data = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        run_id = data.get("run_id")
        acquired_at = data.get("acquired_at")
        if not isinstance(run_id, str) or not isinstance(acquired_at, str):
            return None
        return {"run_id": run_id, "acquired_at": acquired_at}

    def _remove_lock(self, path: str, expected: dict[str, str]) -> bool:
        current = self._read_lock(path)
        if current != expected:
            return False
        try:
            self.fs.rm(path)
        except FileNotFoundError:
            return False
        return True

    def _validate_lock_backend(self) -> None:
        protocol = self.fs.protocol
        if isinstance(protocol, tuple):
            protocol = protocol[0]
        if protocol != "s3":
            return

        endpoint_url = self._configured_s3_endpoint_url()
        if endpoint_url is None or self._is_known_aws_s3_endpoint(endpoint_url):
            return
        if self.lock_config.allow_unsafe_s3_compatible_locks:
            return

        raise RuntimeError(
            "Job locking for s3:// result caches requires AWS S3 semantics. "
            f"This config sets endpoint_url={endpoint_url!r}, which may not honor "
            "exclusive-create lock files safely. Use AWS S3, serialize runs "
            "outside llm-batch-py, or set "
            "LockConfig(allow_unsafe_s3_compatible_locks=True) to bypass this safeguard."
        )

    def _configured_s3_endpoint_url(self) -> str | None:
        options = self.config.storage_options or {}
        endpoint_url = options.get("endpoint_url")
        if isinstance(endpoint_url, str) and endpoint_url:
            return endpoint_url

        client_kwargs = options.get("client_kwargs")
        if isinstance(client_kwargs, dict):
            candidate = client_kwargs.get("endpoint_url")
            if isinstance(candidate, str) and candidate:
                return candidate
        return None

    def _is_known_aws_s3_endpoint(self, endpoint_url: str) -> bool:
        hostname = urlparse(endpoint_url).hostname
        if hostname is None:
            hostname = endpoint_url.split("/", 1)[0].split(":", 1)[0]
        return hostname.endswith("amazonaws.com") or hostname.endswith("amazonaws.com.cn")


def new_batch_id() -> str:
    return f"batch_{uuid4().hex}"


def new_run_id() -> str:
    return f"run_{uuid4().hex}"
