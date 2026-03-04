"""JSONL storage for context snapshots."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator


@dataclass
class ToolInjection:
    tool_name: str
    tool_use_id: str
    estimated_tokens: int
    byte_size: int
    file_path: str | None = None
    is_duplicate: bool = False


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
    context_reset: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ContextSnapshot:
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        d["tool_injections"] = [ToolInjection(**ti) for ti in d["tool_injections"]]
        return cls(**d)


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
