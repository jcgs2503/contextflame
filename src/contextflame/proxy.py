"""MITM proxy that sits between Claude Code and the Anthropic API."""

from __future__ import annotations

import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route, Mount

from contextflame.attributor import attribute_call
from contextflame.storage import append_snapshot

ANTHROPIC_API_BASE = "https://api.anthropic.com"
DEFAULT_LOG_PATH = Path("contextflame.jsonl")


def create_app(log_path: Path = DEFAULT_LOG_PATH) -> Starlette:
    """Create the proxy Starlette app."""
    # Track content hashes for duplicate detection across session
    content_hashes: dict[str, str] = {}  # hash -> first call_id
    previous_input_tokens: int | None = None

    async def proxy_messages(request: Request) -> Response:
        nonlocal previous_input_tokens

        body_bytes = await request.body()
        request_body = json.loads(body_bytes)

        # Forward headers (pass through auth, content-type, etc.)
        forward_headers = {}
        for key in ("authorization", "content-type", "anthropic-version",
                     "anthropic-beta", "x-api-key"):
            val = request.headers.get(key)
            if val:
                forward_headers[key] = val

        is_streaming = request_body.get("stream", False)
        target_url = f"{ANTHROPIC_API_BASE}{request.url.path}"

        call_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc)

        if is_streaming:
            # Client must live as long as the streaming response, so don't
            # use async-with here — the stream generator manages the lifecycle.
            return await _handle_streaming(
                target_url, forward_headers, body_bytes,
                request_body, call_id, timestamp,
                log_path, content_hashes,
            )
        else:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                resp = await client.post(
                    target_url,
                    content=body_bytes,
                    headers=forward_headers,
                )

                response_body = resp.json()

                snapshot = attribute_call(
                    call_id=call_id,
                    timestamp=timestamp,
                    request_body=request_body,
                    response_body=response_body,
                    content_hashes=content_hashes,
                    previous_input_tokens=previous_input_tokens,
                )
                append_snapshot(log_path, snapshot)
                previous_input_tokens = snapshot.input_tokens

                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers=dict(resp.headers),
                    media_type=resp.headers.get("content-type"),
                )

    async def _handle_streaming(
        target_url: str,
        forward_headers: dict,
        body_bytes: bytes,
        request_body: dict,
        call_id: str,
        timestamp: datetime,
        log_path: Path,
        content_hashes: dict,
    ) -> StreamingResponse:
        """Handle streaming SSE responses by buffering for logging while streaming to client."""
        nonlocal previous_input_tokens

        async def stream_generator():
            nonlocal previous_input_tokens

            # Client lives inside the generator so it stays open for the
            # entire duration of the streamed response.
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                async with client.stream(
                    "POST",
                    target_url,
                    content=body_bytes,
                    headers=forward_headers,
                ) as resp:
                    usage_data = {}
                    model = request_body.get("model", "unknown")

                    async for line in resp.aiter_lines():
                        yield line + "\n"

                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                continue
                            try:
                                event_data = json.loads(data_str)
                                if event_data.get("type") == "message_start":
                                    msg = event_data.get("message", {})
                                    usage_data = msg.get("usage", {})
                                    model = msg.get("model", model)
                                elif event_data.get("type") == "message_delta":
                                    delta_usage = event_data.get("usage", {})
                                    usage_data.update(delta_usage)
                            except (json.JSONDecodeError, KeyError):
                                pass

                # Log after stream completes
                response_body = {"model": model, "usage": usage_data}
                snapshot = attribute_call(
                    call_id=call_id,
                    timestamp=timestamp,
                    request_body=request_body,
                    response_body=response_body,
                    content_hashes=content_hashes,
                    previous_input_tokens=previous_input_tokens,
                )
                append_snapshot(log_path, snapshot)
                previous_input_tokens = snapshot.input_tokens

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    async def proxy_catchall(request: Request) -> Response:
        """Forward any non-messages API calls transparently."""
        body_bytes = await request.body()

        forward_headers = {}
        for key, val in request.headers.items():
            if key.lower() not in ("host", "transfer-encoding"):
                forward_headers[key] = val

        target_url = f"{ANTHROPIC_API_BASE}{request.url.path}"

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.request(
                method=request.method,
                url=target_url,
                content=body_bytes,
                headers=forward_headers,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )

    routes = [
        Route("/v1/messages", proxy_messages, methods=["POST"]),
        Mount("/", app=Starlette(routes=[
            Route("/{path:path}", proxy_catchall, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
        ])),
    ]

    return Starlette(routes=routes)
