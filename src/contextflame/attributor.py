"""Token attribution logic - classifies token usage by source category."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from contextflame.storage import ContextSnapshot, ToolInjection


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _content_hash(content: str) -> str:
    """Hash content for duplicate detection."""
    return hashlib.sha256(content.encode(errors="replace")).hexdigest()[:16]


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
    """Convert message content to text for token estimation."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    parts.append("x" * 4000)
                elif block.get("type") == "tool_use":
                    parts.append(json.dumps(block.get("input", {})))
                elif block.get("type") == "tool_result":
                    result_content = block.get("content", "")
                    parts.append(_content_to_text(result_content))
                else:
                    parts.append(json.dumps(block))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


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

    Only counts tool results that are NEW in this call (not carried over
    from conversation history). Uses seen_tool_use_ids to track which
    tool_use_ids have already been counted in prior calls.
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

    # Estimate system prompt tokens
    system = request_body.get("system", "")
    system_text = _content_to_text(system) if isinstance(system, list) else system
    system_tokens = _estimate_tokens(system_text)

    # Map tool_use_id -> (tool_name, tool_input) from assistant messages
    tool_use_map: dict[str, tuple[str, dict]] = {}
    messages = request_body.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_use_map[block.get("id", "")] = (
                            block.get("name", "unknown"),
                            block.get("input", {}),
                        )

    # Extract tool results — only count NEW ones (not seen in prior calls)
    tool_injections: list[ToolInjection] = []
    total_tool_tokens = 0

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

            # Skip tool results we already counted in a previous call
            if tool_use_id in seen_tool_use_ids:
                continue

            # Mark as seen for future calls
            seen_tool_use_ids.add(tool_use_id)

            result_content = _content_to_text(block.get("content", ""))
            byte_size = len(result_content.encode(errors="replace"))
            estimated = _estimate_tokens(result_content)

            tool_name = "unknown"
            file_path = None
            if tool_use_id in tool_use_map:
                tool_name, tool_input = tool_use_map[tool_use_id]
                file_path = _extract_file_path(tool_input)

            # Duplicate = same content seen before (different tool call, same output)
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
            )
            tool_injections.append(injection)
            total_tool_tokens += estimated

    # Estimate how much of input_tokens is conversation history
    # (everything that isn't system prompt or new tool results)
    # This includes: prior messages, prior tool results still in context, user text
    history_tokens = max(0, input_tokens - system_tokens - total_tool_tokens)

    # Context reset detection: significant drop in input tokens
    context_reset = False
    if previous_input_tokens is not None and previous_input_tokens > 0:
        if input_tokens < previous_input_tokens * 0.5:
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
        total_tool_tokens=total_tool_tokens,
        context_reset=context_reset,
    )
