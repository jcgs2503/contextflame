"""Raw logging proxy — dumps full request/response payloads to JSONL."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route, Mount

ANTHROPIC_API_BASE = "https://api.anthropic.com"


def _append_log(path: Path, entry: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def create_raw_app(log_path: Path) -> Starlette:
    """Create a proxy that logs raw request/response payloads."""

    call_counter = {"n": 0}

    async def proxy_messages(request: Request) -> Response:
        call_counter["n"] += 1
        call_num = call_counter["n"]
        call_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).isoformat()

        body_bytes = await request.body()
        request_body = json.loads(body_bytes)

        forward_headers = {}
        for key in ("authorization", "content-type", "anthropic-version",
                     "anthropic-beta", "x-api-key"):
            val = request.headers.get(key)
            if val:
                forward_headers[key] = val

        is_streaming = request_body.get("stream", False)
        target_url = f"{ANTHROPIC_API_BASE}{request.url.path}"

        # Log the request
        _append_log(log_path, {
            "type": "request",
            "call_num": call_num,
            "call_id": call_id,
            "timestamp": timestamp,
            "path": request.url.path,
            "model": request_body.get("model"),
            "stream": is_streaming,
            "system": request_body.get("system"),
            "messages": request_body.get("messages"),
            "max_tokens": request_body.get("max_tokens"),
            "tools": request_body.get("tools"),
            "temperature": request_body.get("temperature"),
        })

        if is_streaming:
            return await _handle_streaming(
                target_url, forward_headers, body_bytes,
                request_body, call_num, call_id, timestamp, log_path,
            )
        else:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                resp = await client.post(
                    target_url, content=body_bytes, headers=forward_headers,
                )
                response_body = resp.json()

                _append_log(log_path, {
                    "type": "response",
                    "call_num": call_num,
                    "call_id": call_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": resp.status_code,
                    "body": response_body,
                })

                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers=dict(resp.headers),
                    media_type=resp.headers.get("content-type"),
                )

    async def _handle_streaming(
        target_url, forward_headers, body_bytes,
        request_body, call_num, call_id, timestamp, log_path,
    ) -> StreamingResponse:

        async def stream_generator():
            sse_events = []

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                async with client.stream(
                    "POST", target_url,
                    content=body_bytes, headers=forward_headers,
                ) as resp:
                    async for line in resp.aiter_lines():
                        yield line + "\n"

                        # Capture SSE data events
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                continue
                            try:
                                sse_events.append(json.loads(data_str))
                            except json.JSONDecodeError:
                                pass

            # Reconstruct the full response from SSE events
            usage = {}
            model = request_body.get("model", "unknown")
            stop_reason = None
            content_blocks = []

            for event in sse_events:
                event_type = event.get("type")

                if event_type == "message_start":
                    msg = event.get("message", {})
                    usage = msg.get("usage", {})
                    model = msg.get("model", model)

                elif event_type == "content_block_start":
                    block = event.get("content_block", {})
                    content_blocks.append(block)

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    idx = event.get("index", 0)
                    if idx < len(content_blocks):
                        block = content_blocks[idx]
                        if delta.get("type") == "text_delta":
                            block["text"] = block.get("text", "") + delta.get("text", "")
                        elif delta.get("type") == "input_json_delta":
                            block["_partial_json"] = block.get("_partial_json", "") + delta.get("partial_json", "")

                elif event_type == "message_delta":
                    delta_usage = event.get("usage", {})
                    usage.update(delta_usage)
                    stop_reason = event.get("delta", {}).get("stop_reason")

            # Parse accumulated JSON for tool_use blocks
            for block in content_blocks:
                if block.get("type") == "tool_use" and "_partial_json" in block:
                    try:
                        block["input"] = json.loads(block.pop("_partial_json"))
                    except json.JSONDecodeError:
                        block["input"] = block.pop("_partial_json")

            _append_log(log_path, {
                "type": "response",
                "call_num": call_num,
                "call_id": call_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": 200,
                "body": {
                    "model": model,
                    "usage": usage,
                    "stop_reason": stop_reason,
                    "content": content_blocks,
                },
            })

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def proxy_catchall(request: Request) -> Response:
        body_bytes = await request.body()
        forward_headers = {}
        for key, val in request.headers.items():
            if key.lower() not in ("host", "transfer-encoding"):
                forward_headers[key] = val

        target_url = f"{ANTHROPIC_API_BASE}{request.url.path}"

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.request(
                method=request.method, url=target_url,
                content=body_bytes, headers=forward_headers,
            )
            return Response(
                content=resp.content, status_code=resp.status_code,
                headers=dict(resp.headers),
            )

    routes = [
        Route("/v1/messages", proxy_messages, methods=["POST"]),
        Mount("/", app=Starlette(routes=[
            Route("/{path:path}", proxy_catchall,
                  methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
        ])),
    ]

    return Starlette(routes=routes)
