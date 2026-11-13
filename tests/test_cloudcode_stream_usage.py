"""Cloud Code Assist streams book the provider's usage, not a local estimate.

``/v1internal:streamGenerateContent`` (Antigravity, Gemini Code Assist) wraps
every SSE chunk as ``{"response": {...}, "traceId": ...}``. The finish-reason
and prompt-block checks already unwrap that envelope; usage extraction did not,
so every Cloud Code turn fell back to the local token estimate and booked zero
cache reads.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

USAGE = {
    "promptTokenCount": 12000,
    "candidatesTokenCount": 40,
    "totalTokenCount": 12040,
    "cachedContentTokenCount": 9000,
}


def _cloudcode_upstream(request: httpx.Request) -> httpx.Response:
    chunk: dict[str, Any] = {
        "response": {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "ack"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": USAGE,
        },
        "traceId": "trace-1",
    }
    return httpx.Response(
        200,
        content=f"data: {json.dumps(chunk)}\n\n".encode(),
        headers={"content-type": "text/event-stream"},
    )


def test_cloudcode_stream_records_provider_usage_from_the_response_envelope() -> None:
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.http_client = httpx.AsyncClient(transport=httpx.MockTransport(_cloudcode_upstream))
        proxy._record_request_outcome = AsyncMock(wraps=proxy._record_request_outcome)

        resp = client.post(
            "/v1internal:streamGenerateContent?alt=sse",
            json={
                "model": "gemini-2.5-pro",
                "project": "p",
                "request": {"contents": [{"role": "user", "parts": [{"text": "hello there"}]}]},
            },
        )
        assert resp.status_code == 200, resp.text[:300]
        resp.read()

    outcome = proxy._record_request_outcome.await_args_list[-1].args[0]
    assert outcome.provider_input_tokens == 12000
    assert outcome.output_tokens == 40
    assert outcome.cache_read_tokens == 9000
