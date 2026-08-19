# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from relax.utils.http_utils import _post


class _StubClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def post(self, url, json=None, headers=None):
        self.calls += 1
        return self._responses.pop(0)


def _response(status_code: int, *, body: str | dict):
    request = httpx.Request("POST", "http://test/post")
    if isinstance(body, dict):
        return httpx.Response(status_code, request=request, json=body)
    return httpx.Response(status_code, request=request, text=body)


def test_post_does_not_retry_non_retryable_400():
    client = _StubClient(
        [
            _response(
                400,
                body={"error": {"message": "Requested token count exceeds the model's maximum context length."}},
            )
        ]
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_post(client, "http://test/post", {}, max_retries=5))

    assert client.calls == 1


def test_post_translates_only_marked_sglang_vision_feature_cache_miss_400():
    from relax.backends.sglang.precomputed_vision import SGLangVisionFeatureCacheMiss

    server_error = SGLangVisionFeatureCacheMiss(
        feature_id="feature-a",
        vision_revision="revision-a",
        feature_schema_version="schema-a",
    )
    client = _StubClient([_response(400, body={"error": {"message": str(server_error)}})])

    with pytest.raises(SGLangVisionFeatureCacheMiss) as raised:
        asyncio.run(_post(client, "http://test/post", {}, max_retries=5))

    assert raised.value.feature_id == "feature-a"
    assert raised.value.vision_revision == "revision-a"
    assert raised.value.feature_schema_version == "schema-a"
    assert client.calls == 1


def test_post_retries_retryable_503_then_succeeds():
    client = _StubClient(
        [
            _response(503, body={"error": {"message": "No available workers"}}),
            _response(200, body={"ok": True}),
        ]
    )

    result = asyncio.run(_post(client, "http://test/post", {}, max_retries=5))

    assert result == {"ok": True}
    assert client.calls == 2


def test_post_collects_request_build_and_response_metrics():
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["request_body"] = request.content
        return httpx.Response(200, request=request, json={"ok": True})

    async def run_request():
        metrics = {}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await _post(
                client,
                "http://test/post",
                {"values": [1.0, 2.0]},
                max_retries=1,
                request_metrics=metrics,
            )
        return result, metrics

    result, metrics = asyncio.run(run_request())

    assert result == {"ok": True}
    assert json.loads(observed["request_body"]) == {"values": [1.0, 2.0]}
    assert metrics["attempts"] == 1
    assert metrics["request_body_bytes"] == len(observed["request_body"])
    assert metrics["response_body_bytes"] > 0
    assert metrics["request_build_seconds"] >= 0.0
    assert metrics["response_wait_seconds"] >= 0.0
    assert metrics["response_read_seconds"] >= 0.0
    assert metrics["response_decode_seconds"] >= 0.0
