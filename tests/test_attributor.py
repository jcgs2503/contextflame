"""Tests for the token attributor."""

from datetime import datetime, timezone

import tiktoken

from contextflame.attributor import _truncate, attribute_call
from contextflame.storage import ContextSnapshot


def _make_request_body(
    system: str = "You are a helpful assistant.",
    messages: list | None = None,
    model: str = "claude-sonnet-4-6",
    tools: list | None = None,
) -> dict:
    body = {
        "model": model,
        "system": system,
        "messages": messages or [],
    }
    if tools is not None:
        body["tools"] = tools
    return body


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


def _tiktoken_count(text: str) -> int:
    """Reference count using tiktoken directly."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


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

    # tiktoken should produce a reasonable system token count
    assert snap.system_tokens == _tiktoken_count("You are a helpful assistant.")
    assert snap.system_tokens > 0

    # Token breakdown should exist
    assert snap.token_breakdown is not None
    assert snap.token_breakdown.system_tokens == snap.system_tokens
    assert snap.token_breakdown.api_total == 500  # input_tokens + cache


def test_tool_injection_attribution():
    """Test that tool results are correctly attributed with tiktoken."""
    content = "x" * 4000
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
                    "content": content,
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
    assert snap.tool_injections[0].is_duplicate is False

    # tiktoken should give us the actual BPE token count
    expected_tokens = _tiktoken_count(content)
    assert snap.tool_injections[0].estimated_tokens == expected_tokens
    assert snap.total_tool_tokens == expected_tokens


def test_no_double_counting_across_calls():
    """Tool results from prior calls in the message history are not re-counted."""
    content_hashes: dict[str, str] = {}
    seen_ids: set[str] = set()

    # Call 1: one tool result
    messages_call1 = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "/a.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "x" * 400}]},
    ]
    snap1 = attribute_call(
        call_id="c1", timestamp=datetime.now(timezone.utc),
        request_body=_make_request_body(messages=messages_call1),
        response_body=_make_response_body(input_tokens=500),
        content_hashes=content_hashes, seen_tool_use_ids=seen_ids,
    )
    assert len(snap1.tool_injections) == 1
    expected_tokens = _tiktoken_count("x" * 400)
    assert snap1.total_tool_tokens == expected_tokens

    # Call 2: same history + one new tool result
    messages_call2 = messages_call1 + [
        {"role": "assistant", "content": [{"type": "text", "text": "I read the file."}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_2", "name": "Grep", "input": {"pattern": "TODO"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_2", "content": "y" * 800}]},
    ]
    snap2 = attribute_call(
        call_id="c2", timestamp=datetime.now(timezone.utc),
        request_body=_make_request_body(messages=messages_call2),
        response_body=_make_response_body(input_tokens=1000),
        content_hashes=content_hashes, seen_tool_use_ids=seen_ids,
    )
    # Should have 2 injections: 1 carried-over (tu_1) + 1 new (tu_2)
    assert len(snap2.tool_injections) == 2

    carried = [inj for inj in snap2.tool_injections if inj.is_carried_over]
    new = [inj for inj in snap2.tool_injections if not inj.is_carried_over]
    assert len(carried) == 1
    assert carried[0].tool_name == "Read"
    assert carried[0].tool_use_id == "tu_1"
    assert len(new) == 1
    assert new[0].tool_name == "Grep"

    # total_tool_tokens only counts NEW injections
    expected_tokens_2 = _tiktoken_count("y" * 800)
    assert snap2.total_tool_tokens == expected_tokens_2

    # carried_over_tool_tokens counts carried-over
    assert snap2.carried_over_tool_tokens == expected_tokens


def test_duplicate_detection():
    """Duplicate = different tool call, same content output."""
    content = "same content repeated"
    content_hashes: dict[str, str] = {}
    seen_ids: set[str] = set()

    # Call 1: Read /a.py → "same content repeated"
    messages1 = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "/a.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": content}]},
    ]
    snap1 = attribute_call(
        call_id="c1", timestamp=datetime.now(timezone.utc),
        request_body=_make_request_body(messages=messages1),
        response_body=_make_response_body(),
        content_hashes=content_hashes, seen_tool_use_ids=seen_ids,
    )
    assert snap1.tool_injections[0].is_duplicate is False

    # Call 2: Read /a.py again (different tool_use_id, same content)
    messages2 = messages1 + [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_2", "name": "Read", "input": {"file_path": "/a.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_2", "content": content}]},
    ]
    snap2 = attribute_call(
        call_id="c2", timestamp=datetime.now(timezone.utc),
        request_body=_make_request_body(messages=messages2),
        response_body=_make_response_body(),
        content_hashes=content_hashes, seen_tool_use_ids=seen_ids,
    )
    # tu_1 is carried-over, tu_2 is new and IS a duplicate (same content)
    assert len(snap2.tool_injections) == 2
    carried = [inj for inj in snap2.tool_injections if inj.is_carried_over]
    new = [inj for inj in snap2.tool_injections if not inj.is_carried_over]
    assert len(carried) == 1
    assert carried[0].is_duplicate is False  # carried-over skips duplicate detection
    assert len(new) == 1
    assert new[0].is_duplicate is True


def test_context_reset_detection():
    """Test context reset is detected on significant token drop."""
    req = _make_request_body()

    snap1 = attribute_call(
        call_id="c1", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=_make_response_body(input_tokens=100000),
        content_hashes={}, previous_input_tokens=None,
    )
    assert snap1.context_reset is False

    snap2 = attribute_call(
        call_id="c2", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=_make_response_body(input_tokens=120000),
        content_hashes={}, previous_input_tokens=100000,
    )
    assert snap2.context_reset is False

    snap3 = attribute_call(
        call_id="c3", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=_make_response_body(input_tokens=30000),
        content_hashes={}, previous_input_tokens=100000,
    )
    assert snap3.context_reset is True


def test_snapshot_serialization():
    """Test round-trip serialization of ContextSnapshot with token breakdown."""
    req = _make_request_body(
        messages=[
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "Grep", "input": {"pattern": "TODO"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "found 3 matches"}]},
        ],
        tools=[{"name": "Grep", "description": "Search files", "input_schema": {"type": "object"}}],
    )
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

    # Token breakdown should survive serialization
    assert restored.token_breakdown is not None
    assert restored.token_breakdown.system_tokens == snap.token_breakdown.system_tokens
    assert restored.token_breakdown.tools_tokens == snap.token_breakdown.tools_tokens
    assert restored.token_breakdown.estimated_total == snap.token_breakdown.estimated_total
    assert restored.token_breakdown.api_total == snap.token_breakdown.api_total
    assert len(restored.token_breakdown.tool_schemas) == 1
    assert restored.token_breakdown.tool_schemas[0].tool_name == "Grep"


def test_token_breakdown_components():
    """Test that token breakdown captures all component categories."""
    messages = [
        {"role": "user", "content": "Hello, can you help me?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Sure, let me look at your code."},
            {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "/main.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "def main():\n    print('hello')"},
        ]},
        {"role": "assistant", "content": "I see your code."},
    ]
    tools = [
        {"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}},
        {"name": "Bash", "description": "Run a command", "input_schema": {"type": "object"}},
    ]
    req = _make_request_body(messages=messages, tools=tools)
    resp = _make_response_body(input_tokens=1000, output_tokens=50)

    snap = attribute_call(
        call_id="bd1", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=resp,
        content_hashes={},
    )

    tb = snap.token_breakdown
    assert tb is not None
    assert tb.system_tokens > 0  # "You are a helpful assistant."
    assert tb.tools_tokens > 0   # 2 tool schemas
    assert tb.user_text_tokens > 0  # "Hello, can you help me?"
    assert tb.assistant_text_tokens > 0  # "Sure..." + "I see your code."
    assert tb.tool_result_tokens > 0  # The file content
    assert tb.tool_use_tokens > 0  # Read tool_use block
    assert tb.estimated_total > 0
    assert tb.api_total == 1000  # input_tokens only (no cache)

    # Should have 2 tool schemas
    assert len(tb.tool_schemas) == 2
    schema_names = {s.tool_name for s in tb.tool_schemas}
    assert schema_names == {"Read", "Bash"}

    # Estimation error should be a float
    assert isinstance(tb.estimation_error, float)


def test_tool_schemas_tokenization():
    """Test that tool schemas are individually tokenized."""
    tools = [
        {"name": "Read", "description": "Read file contents from disk", "input_schema": {
            "type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"],
        }},
        {"name": "Bash", "description": "Execute a bash command", "input_schema": {
            "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"],
        }},
    ]
    req = _make_request_body(tools=tools)
    resp = _make_response_body(input_tokens=500)

    snap = attribute_call(
        call_id="ts1", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=resp,
        content_hashes={},
    )

    tb = snap.token_breakdown
    assert len(tb.tool_schemas) == 2

    # Each schema should have non-zero tokens
    for schema in tb.tool_schemas:
        assert schema.tokens > 0

    # Sum of individual schemas should equal total tools_tokens
    assert sum(s.tokens for s in tb.tool_schemas) == tb.tools_tokens


def test_truncate_short_text():
    """_truncate returns text unchanged when within limit."""
    assert _truncate("hello", 300) == "hello"
    assert _truncate("", 300) == ""
    assert _truncate("abc", 3) == "abc"


def test_truncate_long_text():
    """_truncate cuts text and appends char count."""
    result = _truncate("a" * 500, 300)
    assert result.startswith("a" * 300)
    assert "200 more chars" in result
    assert len(result) < 500


def test_truncate_none():
    """_truncate handles None/empty input."""
    assert _truncate(None) is None
    assert _truncate("") == ""


def test_content_previews_populated():
    """Content previews are populated during attribution."""
    messages = [
        {"role": "user", "content": "Hello, can you help me?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Sure, let me look at your code."},
            {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "/main.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "def main():\n    print('hello')"},
        ]},
    ]
    req = _make_request_body(messages=messages)
    resp = _make_response_body(input_tokens=1000, output_tokens=50)

    snap = attribute_call(
        call_id="cp1", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=resp,
        content_hashes={},
    )

    assert snap.content_previews is not None
    assert snap.content_previews.system_preview is not None  # system prompt
    assert "helpful assistant" in snap.content_previews.system_preview
    assert snap.content_previews.user_text_preview is not None
    assert "Hello" in snap.content_previews.user_text_preview
    assert snap.content_previews.assistant_text_preview is not None
    assert "look at your code" in snap.content_previews.assistant_text_preview
    assert snap.content_previews.tool_calls_preview is not None
    assert "Read" in snap.content_previews.tool_calls_preview


def test_tool_injection_content_preview():
    """Tool injections get content_preview set."""
    content = "x" * 500
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "/a.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": content}]},
    ]
    req = _make_request_body(messages=messages)
    resp = _make_response_body(input_tokens=1000)

    snap = attribute_call(
        call_id="tip1", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=resp,
        content_hashes={},
    )

    assert len(snap.tool_injections) == 1
    preview = snap.tool_injections[0].content_preview
    assert preview is not None
    assert preview.startswith("x" * 300)
    assert "200 more chars" in preview


def test_content_previews_serialization():
    """ContentPreviews round-trip through to_dict/from_dict."""
    messages = [
        {"role": "user", "content": "test user message"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_1", "name": "Grep", "input": {"pattern": "TODO"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "found matches"},
        ]},
    ]
    req = _make_request_body(messages=messages)
    resp = _make_response_body(input_tokens=500, output_tokens=50)

    snap = attribute_call(
        call_id="cps1", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=resp,
        content_hashes={},
    )

    d = snap.to_dict()
    restored = ContextSnapshot.from_dict(d)

    assert restored.content_previews is not None
    assert restored.content_previews.system_preview == snap.content_previews.system_preview
    assert restored.content_previews.user_text_preview == snap.content_previews.user_text_preview
    assert restored.content_previews.tool_calls_preview == snap.content_previews.tool_calls_preview

    # Tool injection content_preview should also survive
    assert restored.tool_injections[0].content_preview == snap.tool_injections[0].content_preview


def test_backward_compat_no_content_previews():
    """Old snapshots without content_previews deserialize cleanly."""
    req = _make_request_body()
    resp = _make_response_body(input_tokens=500)

    snap = attribute_call(
        call_id="bc1", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=resp,
        content_hashes={},
    )

    d = snap.to_dict()
    # Simulate old data by removing new fields
    del d["content_previews"]
    for ti in d["tool_injections"]:
        ti.pop("content_preview", None)

    restored = ContextSnapshot.from_dict(d)
    assert restored.content_previews is None
    # Should not raise


def test_carried_over_content_preview():
    """Carried-over injections get content_preview set."""
    content_hashes: dict[str, str] = {}
    seen_ids: set[str] = set()

    # Call 1
    messages1 = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "/a.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "hello world content"}]},
    ]
    attribute_call(
        call_id="c1", timestamp=datetime.now(timezone.utc),
        request_body=_make_request_body(messages=messages1),
        response_body=_make_response_body(input_tokens=500),
        content_hashes=content_hashes, seen_tool_use_ids=seen_ids,
    )

    # Call 2: tu_1 is carried over
    messages2 = messages1 + [
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]
    snap2 = attribute_call(
        call_id="c2", timestamp=datetime.now(timezone.utc),
        request_body=_make_request_body(messages=messages2),
        response_body=_make_response_body(input_tokens=600),
        content_hashes=content_hashes, seen_tool_use_ids=seen_ids,
    )

    carried = [inj for inj in snap2.tool_injections if inj.is_carried_over]
    assert len(carried) == 1
    assert carried[0].content_preview is not None
    assert "hello world" in carried[0].content_preview


def test_carried_over_serialization():
    """Round-trip is_carried_over and carried_over_tool_tokens."""
    content_hashes: dict[str, str] = {}
    seen_ids: set[str] = set()

    messages1 = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "/a.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "file content"}]},
    ]
    attribute_call(
        call_id="c1", timestamp=datetime.now(timezone.utc),
        request_body=_make_request_body(messages=messages1),
        response_body=_make_response_body(input_tokens=500),
        content_hashes=content_hashes, seen_tool_use_ids=seen_ids,
    )

    messages2 = messages1 + [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_2", "name": "Grep", "input": {"pattern": "x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_2", "content": "grep result"}]},
    ]
    snap = attribute_call(
        call_id="c2", timestamp=datetime.now(timezone.utc),
        request_body=_make_request_body(messages=messages2),
        response_body=_make_response_body(input_tokens=800),
        content_hashes=content_hashes, seen_tool_use_ids=seen_ids,
    )

    d = snap.to_dict()
    restored = ContextSnapshot.from_dict(d)

    assert restored.carried_over_tool_tokens == snap.carried_over_tool_tokens
    assert restored.carried_over_tool_tokens > 0

    carried = [inj for inj in restored.tool_injections if inj.is_carried_over]
    assert len(carried) == 1
    assert carried[0].tool_name == "Read"
    assert carried[0].is_carried_over is True


def test_backward_compat_no_carried_over():
    """Old JSONL without carried-over fields deserializes with defaults."""
    req = _make_request_body(
        messages=[
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "Read", "input": {"file_path": "/a.py"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "data"}]},
        ]
    )
    resp = _make_response_body(input_tokens=500)

    snap = attribute_call(
        call_id="bc2", timestamp=datetime.now(timezone.utc),
        request_body=req, response_body=resp,
        content_hashes={},
    )

    d = snap.to_dict()
    # Simulate old data by removing new fields
    del d["carried_over_tool_tokens"]
    for ti in d["tool_injections"]:
        ti.pop("is_carried_over", None)

    restored = ContextSnapshot.from_dict(d)
    assert restored.carried_over_tool_tokens == 0
    assert restored.tool_injections[0].is_carried_over is False
