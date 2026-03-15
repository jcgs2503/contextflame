"""JSONL storage for context snapshots."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator


@dataclass
class ContentPreviews:
    """Truncated previews of each content category for click-to-inspect."""
    system_preview: str | None = None
    user_text_preview: str | None = None
    assistant_text_preview: str | None = None
    tool_calls_preview: str | None = None
    thinking_preview: str | None = None


@dataclass
class ToolInjection:
    tool_name: str
    tool_use_id: str
    estimated_tokens: int
    byte_size: int
    file_path: str | None = None
    is_duplicate: bool = False
    content_preview: str | None = None
    is_carried_over: bool = False


@dataclass
class ToolSchemaTokens:
    """Token count for a single tool schema in the tools[] array."""
    tool_name: str
    tokens: int


@dataclass
class TokenBreakdown:
    """Granular per-component token counts using tiktoken."""
    # System prompt components
    system_tokens: int = 0

    # Tool schemas (the tools[] array sent every request)
    tools_tokens: int = 0
    tool_schemas: list[ToolSchemaTokens] = field(default_factory=list)

    # Messages breakdown
    user_text_tokens: int = 0      # plain user text messages
    assistant_text_tokens: int = 0  # assistant text responses
    tool_result_tokens: int = 0    # tool_result content in user messages
    tool_use_tokens: int = 0       # tool_use blocks in assistant messages
    thinking_tokens: int = 0       # thinking blocks

    # Grand total from tiktoken
    estimated_total: int = 0

    # Tokens from serializing the entire request body (diagnostic)
    raw_body_tokens: int = 0

    # API ground truth
    api_total: int = 0  # input_tokens + cache_read + cache_creation

    @property
    def estimation_error(self) -> float:
        """How far off tiktoken is from API total. Negative = underestimate."""
        if self.api_total == 0:
            return 0.0
        return (self.estimated_total - self.api_total) / self.api_total


@dataclass
class ContextSnapshot:
    call_id: str
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    system_tokens: int
    history_tokens: int
    tool_injections: list[ToolInjection]
    total_tool_tokens: int
    carried_over_tool_tokens: int = 0
    context_reset: bool = False
    token_breakdown: TokenBreakdown | None = None
    content_previews: ContentPreviews | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ContextSnapshot:
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        d["tool_injections"] = [ToolInjection(**ti) for ti in d["tool_injections"]]
        tb = d.pop("token_breakdown", None)
        if tb is not None:
            schemas = [ToolSchemaTokens(**s) for s in tb.pop("tool_schemas", [])]
            tb = TokenBreakdown(**tb, tool_schemas=schemas)
        cp = d.pop("content_previews", None)
        if cp is not None:
            cp = ContentPreviews(**cp)
        snap = cls(**d)
        snap.token_breakdown = tb
        snap.content_previews = cp
        return snap


def append_snapshot(path: Path, snapshot: ContextSnapshot) -> None:
    """Append a single snapshot as a JSON line."""
    with open(path, "a") as f:
        f.write(json.dumps(snapshot.to_dict()) + "\n")


def read_snapshots(path: Path) -> Iterator[ContextSnapshot]:
    """Read all snapshots from a JSONL file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield ContextSnapshot.from_dict(json.loads(line))


def read_all_snapshots(path: Path) -> list[ContextSnapshot]:
    """Read all snapshots into a list."""
    return list(read_snapshots(path))
