from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from llm_batch_py._core_wrapper import stable_hash
from llm_batch_py.jobs import PromptCacheConfig

_LOGGER = logging.getLogger("llm_batch_py.prompt_cache")
_EMITTED_DIAGNOSTICS: set[str] = set()
_PREVIEW_LIMIT = 160


def prompt_cache_enabled(config: PromptCacheConfig | None) -> bool:
    return config is not None and config.mode != "off"


def emit_prompt_cache_diagnostic_once(
    *,
    provider_name: str,
    model: str,
    config: PromptCacheConfig,
    payload: dict[str, Any],
    boundary: str,
    note: str,
    request_fields: dict[str, Any] | None = None,
) -> None:
    if not config.verbose:
        return
    effective_mode = "off" if config.mode == "off" else "auto"
    blocks = _flatten_prompt_blocks(payload)
    if boundary == "explicit":
        cached_blocks, uncached_blocks, breakpoints = _explicit_block_split(blocks)
    else:
        cached_blocks, uncached_blocks = _provider_managed_block_split(blocks)
        breakpoints = []

    diagnostic_key = stable_hash(
        {
            "provider": provider_name,
            "model": model,
            "config": asdict(config),
            "payload": payload,
            "boundary": boundary,
            "fields": request_fields or {},
        }
    )
    if diagnostic_key in _EMITTED_DIAGNOSTICS:
        return
    _EMITTED_DIAGNOSTICS.add(diagnostic_key)

    _LOGGER.info(
        "Prompt cache estimated analysis provider=%s model=%s mode=%s boundary=%s breakpoints=%s "
        "request_fields=%s estimated_cached_locations=%s estimated_uncached_locations=%s "
        "estimated_cached_preview=%r estimated_uncached_preview=%r note=%s",
        provider_name,
        model,
        effective_mode,
        boundary,
        breakpoints,
        request_fields or {},
        [block["location"] for block in cached_blocks],
        [block["location"] for block in uncached_blocks],
        _preview_text(cached_blocks),
        _preview_text(uncached_blocks),
        note,
    )


def _provider_managed_block_split(
    blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(blocks) <= 1:
        return [], list(blocks)
    return blocks[:-1], blocks[-1:]


def _explicit_block_split(
    blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    breakpoint_indexes = [index for index, block in enumerate(blocks) if block["cache_control"]]
    if not breakpoint_indexes:
        return [], list(blocks), []
    last_breakpoint = breakpoint_indexes[-1]
    return (
        blocks[: last_breakpoint + 1],
        blocks[last_breakpoint + 1 :],
        [blocks[index]["location"] for index in breakpoint_indexes],
    )


def _flatten_prompt_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for index, tool in enumerate(payload.get("tools") or []):
        blocks.append(
            {
                "location": f"tools[{index}]",
                "text": _stringify_block(tool),
                "cache_control": isinstance(tool, dict) and tool.get("cache_control") is not None,
            }
        )
    system = payload.get("system")
    if system is not None:
        blocks.extend(_flatten_content("system", system))
    for message_index, message in enumerate(payload.get("messages") or []):
        content = message.get("content")
        location = f"messages[{message_index}]"
        blocks.extend(_flatten_content(location, content))
    return blocks


def _flatten_content(base_location: str, content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        flattened: list[dict[str, Any]] = []
        for index, item in enumerate(content):
            location = f"{base_location}.content[{index}]"
            flattened.append(
                {
                    "location": location,
                    "text": _stringify_block(item),
                    "cache_control": isinstance(item, dict)
                    and item.get("cache_control") is not None,
                }
            )
        return flattened
    return [
        {
            "location": base_location,
            "text": _stringify_block(content),
            "cache_control": isinstance(content, dict) and content.get("cache_control") is not None,
        }
    ]


def _stringify_block(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), str):
            return value["content"]
    text = str(value)
    return text if len(text) <= _PREVIEW_LIMIT else f"{text[:_PREVIEW_LIMIT - 3]}..."


def _preview_text(blocks: list[dict[str, Any]]) -> str:
    combined = " | ".join(block["text"] for block in blocks if block["text"])
    if len(combined) <= _PREVIEW_LIMIT:
        return combined
    return f"{combined[:_PREVIEW_LIMIT - 3]}..."


def _reset_prompt_cache_diagnostics() -> None:
    _EMITTED_DIAGNOSTICS.clear()
