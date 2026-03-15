"""Token attribution logic - classifies token usage by source category using tiktoken."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import tiktoken

from contextflame.storage import (
    ContentPreviews,
    ContextSnapshot,
    TokenBreakdown,
    ToolInjection,
    ToolSchemaTokens,
)

# Lazy-loaded encoder (cl100k_base is closest to Claude's tokenizer)
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken BPE tokenizer."""
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def _content_hash(content: str) -> str:
    """Hash content for duplicate detection."""
    return hashlib.sha256(content.encode(errors="replace")).hexdigest()[:16]


def _truncate(text: str, limit: int = 300) -> str:
    """Truncate text to limit chars, appending a count of remaining chars."""
    if not text or len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text) - limit} more chars]"


def _extract_file_path(tool_input: dict) -> str | None:
    """Try to extract a file path from tool input parameters."""
    for key in ("file_path", "path", "filePath", "filename"):
        if key in tool_input:
            return tool_input[key]
    command = tool_input.get("command", "")
    if command:
        parts = command.split()
        for part in parts:
            if "/" in part and not part.startswith("-"):
                return part
    return None


def _content_to_text(content) -> str:
    """Convert message content to text for token counting."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    # Images are ~1600 tokens for a typical image
                    parts.append("x" * 6400)
                elif block.get("type") == "tool_use":
                    parts.append(json.dumps(block.get("input", {})))
                elif block.get("type") == "tool_result":
                    result_content = block.get("content", "")
                    parts.append(_content_to_text(result_content))
                elif block.get("type") == "thinking":
                    parts.append(block.get("thinking", ""))
                else:
                    parts.append(json.dumps(block))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _count_system_tokens(system) -> int:
    """Count tokens in the system prompt (string or list of blocks)."""
    if not system:
        return 0
    if isinstance(system, str):
        return _count_tokens(system)
    if isinstance(system, list):
        total = 0
        for block in system:
            if isinstance(block, dict):
                text = block.get("text", "")
                total += _count_tokens(text)
            elif isinstance(block, str):
                total += _count_tokens(block)
        return total
    return _count_tokens(str(system))


def _count_tool_schemas(tools: list[dict]) -> tuple[int, list[ToolSchemaTokens]]:
    """Count tokens for each tool schema and return total + per-tool breakdown."""
    if not tools:
        return 0, []
    schemas = []
    total = 0
    for tool in tools:
        text = json.dumps(tool)
        tokens = _count_tokens(text)
        name = tool.get("name", "unknown")
        schemas.append(ToolSchemaTokens(tool_name=name, tokens=tokens))
        total += tokens
    return total, schemas


def attribute_call(
    call_id: str,
    timestamp: datetime,
    request_body: dict,
    response_body: dict,
    content_hashes: dict[str, str],
    seen_tool_use_ids: set[str] | None = None,
    previous_input_tokens: int | None = None,
) -> ContextSnapshot:
    """Parse a request/response pair and produce a ContextSnapshot.

    Uses tiktoken for accurate per-component token estimation.
    Only counts tool results that are NEW in this call (not carried over
    from conversation history).
    """
    if seen_tool_use_ids is None:
        seen_tool_use_ids = set()

    model = response_body.get("model", request_body.get("model", "unknown"))
    usage = response_body.get("usage", {})

    # Ground truth from the API
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cache_read_tokens = usage.get("cache_read_input_tokens", 0)
    cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
    api_total = input_tokens + cache_read_tokens + cache_creation_tokens

    # --- Per-component token counting with tiktoken ---

    # 1. System prompt
    system = request_body.get("system", "")
    system_tokens = _count_system_tokens(system)
    system_preview_text = _content_to_text(system) if system else ""

    # 2. Tool schemas (tools[] array)
    tools = request_body.get("tools", [])
    tools_tokens, tool_schema_list = _count_tool_schemas(tools)

    # 3. Messages breakdown
    user_text_tokens = 0
    assistant_text_tokens = 0
    tool_result_tokens = 0
    tool_use_tokens = 0
    thinking_tokens = 0

    # Content preview accumulators
    user_text_parts: list[str] = []
    assistant_text_parts: list[str] = []
    tool_calls_parts: list[str] = []
    thinking_parts: list[str] = []

    # Map tool_use_id -> (tool_name, tool_input) from assistant messages
    tool_use_map: dict[str, tuple[str, dict]] = {}
    messages = request_body.get("messages", [])

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", [])

        if isinstance(content, str):
            tokens = _count_tokens(content)
            if role == "user":
                user_text_tokens += tokens
                user_text_parts.append(content)
            elif role == "assistant":
                assistant_text_tokens += tokens
                assistant_text_parts.append(content)
            continue

        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                if isinstance(block, str):
                    tokens = _count_tokens(block)
                    if role == "user":
                        user_text_tokens += tokens
                    elif role == "assistant":
                        assistant_text_tokens += tokens
                continue

            block_type = block.get("type", "")

            if block_type == "text":
                text = block.get("text", "")
                tokens = _count_tokens(text)
                if role == "user":
                    user_text_tokens += tokens
                    user_text_parts.append(text)
                elif role == "assistant":
                    assistant_text_tokens += tokens
                    assistant_text_parts.append(text)

            elif block_type == "tool_use":
                tool_name_str = block.get("name", "unknown")
                tool_use_map[block.get("id", "")] = (
                    tool_name_str,
                    block.get("input", {}),
                )
                # Count the tool_use block itself (name + input JSON)
                tool_use_text = json.dumps(block.get("input", {}))
                tool_use_tokens += _count_tokens(tool_name_str) + _count_tokens(tool_use_text)
                tool_calls_parts.append(f"{tool_name_str}({_truncate(tool_use_text, 100)})")

            elif block_type == "tool_result":
                result_text = _content_to_text(block.get("content", ""))
                tool_result_tokens += _count_tokens(result_text)

            elif block_type == "thinking":
                thinking_text = block.get("thinking", "")
                thinking_tokens += _count_tokens(thinking_text)
                thinking_parts.append(thinking_text)

            else:
                # Other block types - count as part of the role
                tokens = _count_tokens(json.dumps(block))
                if role == "user":
                    user_text_tokens += tokens
                elif role == "assistant":
                    assistant_text_tokens += tokens

    # Build content previews from accumulated text
    content_previews = ContentPreviews(
        system_preview=_truncate(system_preview_text, 500) if system_preview_text else None,
        user_text_preview=_truncate("\n".join(user_text_parts), 300) if user_text_parts else None,
        assistant_text_preview=_truncate("\n".join(assistant_text_parts), 300) if assistant_text_parts else None,
        tool_calls_preview=_truncate("\n".join(tool_calls_parts), 300) if tool_calls_parts else None,
        thinking_preview=_truncate("\n".join(thinking_parts), 300) if thinking_parts else None,
    )

    estimated_total = (
        system_tokens + tools_tokens
        + user_text_tokens + assistant_text_tokens
        + tool_result_tokens + tool_use_tokens + thinking_tokens
    )

    breakdown = TokenBreakdown(
        system_tokens=system_tokens,
        tools_tokens=tools_tokens,
        tool_schemas=tool_schema_list,
        user_text_tokens=user_text_tokens,
        assistant_text_tokens=assistant_text_tokens,
        tool_result_tokens=tool_result_tokens,
        tool_use_tokens=tool_use_tokens,
        thinking_tokens=thinking_tokens,
        estimated_total=estimated_total,
        api_total=api_total,
    )

    # --- Extract tool injections (new + carried-over) ---
    tool_injections: list[ToolInjection] = []
    total_new_tool_tokens = 0
    total_carried_over_tokens = 0

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue

            tool_use_id = block.get("tool_use_id", "")
            is_carried_over = tool_use_id in seen_tool_use_ids

            result_content = _content_to_text(block.get("content", ""))
            byte_size = len(result_content.encode(errors="replace"))
            estimated = _count_tokens(result_content)

            tool_name = "unknown"
            file_path = None
            if tool_use_id in tool_use_map:
                tool_name, tool_input = tool_use_map[tool_use_id]
                file_path = _extract_file_path(tool_input)

            # Only run duplicate detection for new injections
            is_duplicate = False
            if not is_carried_over:
                seen_tool_use_ids.add(tool_use_id)
                content_h = _content_hash(result_content)
                is_duplicate = content_h in content_hashes
                if not is_duplicate:
                    content_hashes[content_h] = call_id

            injection = ToolInjection(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                estimated_tokens=estimated,
                byte_size=byte_size,
                file_path=file_path,
                is_duplicate=is_duplicate,
                content_preview=_truncate(result_content, 300),
                is_carried_over=is_carried_over,
            )
            tool_injections.append(injection)
            if is_carried_over:
                total_carried_over_tokens += estimated
            else:
                total_new_tool_tokens += estimated

    # History tokens = everything in messages that isn't tool results
    history_tokens = max(0, api_total - system_tokens - total_new_tool_tokens - total_carried_over_tokens)

    # Context reset detection: significant drop in input tokens
    context_reset = False
    if previous_input_tokens is not None and previous_input_tokens > 0:
        if api_total < previous_input_tokens * 0.5:
            context_reset = True

    return ContextSnapshot(
        call_id=call_id,
        timestamp=timestamp,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        system_tokens=system_tokens,
        history_tokens=history_tokens,
        tool_injections=tool_injections,
        total_tool_tokens=total_new_tool_tokens,
        carried_over_tool_tokens=total_carried_over_tokens,
        context_reset=context_reset,
        token_breakdown=breakdown,
        content_previews=content_previews,
    )
