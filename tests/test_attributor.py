"""Tests for the token attributor."""

from datetime import datetime, timezone

from contextflame.attributor import attribute_call
from contextflame.storage import ContextSnapshot


def _make_request_body(
    system: str = "You are a helpful assistant.",
    messages: list | None = None,
    model: str = "claude-sonnet-4-6",
) -> dict:
    return {
        "model": model,
        "system": system,
        "messages": messages or [],
    }


def _make_response_body(
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read: int = 0,
    cache_creation: int = 0,
    model: str = "claude-sonnet-4-6",
) -> dict:
    return {
        "model": model,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        },
    }


def test_basic_attribution():
    """Test basic call attribution with no tool injections."""
    req = _make_request_body()
    resp = _make_response_body(input_tokens=500, output_tokens=100)

    snap = attribute_call(
        call_id="test1",
        timestamp=datetime.now(timezone.utc),
        request_body=req,
        response_body=resp,
        content_hashes={},
    )

    assert snap.call_id == "test1"
    assert snap.input_tokens == 500
    assert snap.output_tokens == 100
    assert snap.tool_injections == []
    assert snap.total_tool_tokens == 0
    assert snap.context_reset is False


def test_tool_injection_attribution():
    """Test that tool results are correctly attributed."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_123",
                    "name": "Read",
                    "input": {"file_path": "/src/main.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_123",
                    "content": "x" * 4000,  # ~1000 tokens
                }
            ],
        },
    ]
    req = _make_request_body(messages=messages)
    resp = _make_response_body(input_tokens=2000, output_tokens=100)

    snap = attribute_call(
        call_id="test2",
        timestamp=datetime.now(timezone.utc),
        request_body=req,
        response_body=resp,
        content_hashes={},
    )

    assert len(snap.tool_injections) == 1
    assert snap.tool_injections[0].tool_name == "Read"
    assert snap.tool_injections[0].file_path == "/src/main.py"
    assert snap.tool_injections[0].estimated_tokens == 1000
    assert snap.tool_injections[0].is_duplicate is False
    assert snap.total_tool_tokens == 1000


def test_duplicate_detection():
    """Test that duplicate tool results are flagged."""
    content = "same content repeated"
    content_hashes: dict[str, str] = {}

    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "/a.py"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": content}],
        },
    ]

    req = _make_request_body(messages=messages)
    resp = _make_response_body()

    # First call — not duplicate
    snap1 = attribute_call(
        call_id="c1", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=resp,
        content_hashes=content_hashes,
    )
    assert snap1.tool_injections[0].is_duplicate is False

    # Second call with same content — duplicate
    snap2 = attribute_call(
        call_id="c2", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=resp,
        content_hashes=content_hashes,
    )
    assert snap2.tool_injections[0].is_duplicate is True


def test_context_reset_detection():
    """Test context reset is detected on significant token drop."""
    req = _make_request_body()

    # No reset on first call
    snap1 = attribute_call(
        call_id="c1", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=_make_response_body(input_tokens=100000),
        content_hashes={}, previous_input_tokens=None,
    )
    assert snap1.context_reset is False

    # No reset when tokens grow
    snap2 = attribute_call(
        call_id="c2", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=_make_response_body(input_tokens=120000),
        content_hashes={}, previous_input_tokens=100000,
    )
    assert snap2.context_reset is False

    # Reset when tokens drop >50%
    snap3 = attribute_call(
        call_id="c3", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=_make_response_body(input_tokens=30000),
        content_hashes={}, previous_input_tokens=100000,
    )
    assert snap3.context_reset is True


def test_snapshot_serialization():
    """Test round-trip serialization of ContextSnapshot."""
    req = _make_request_body(messages=[
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "Grep", "input": {"pattern": "TODO"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "found 3 matches"}]},
    ])
    resp = _make_response_body(input_tokens=5000, output_tokens=300)

    snap = attribute_call(
        call_id="ser1", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=resp,
        content_hashes={},
    )

    d = snap.to_dict()
    restored = ContextSnapshot.from_dict(d)

    assert restored.call_id == snap.call_id
    assert restored.input_tokens == snap.input_tokens
    assert len(restored.tool_injections) == len(snap.tool_injections)
    assert restored.tool_injections[0].tool_name == "Grep"
