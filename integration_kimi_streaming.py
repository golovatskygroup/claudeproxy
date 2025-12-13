#!/usr/bin/env python3
"""Integration tests for streaming through claudeproxy using moonshotai/kimi-k2-thinking.

This script is intentionally framework-free (no pytest) to keep the repo lightweight.
It runs 10 sequential checks and exits non-zero on the first failure.

Env vars:
- PROXY_BASE_URL (default: http://127.0.0.1:8001)
- PROXY_API_KEY  (default: localtest-key)
- TEST_MODEL     (default: moonshotai/kimi-k2-thinking)
- FLAKE_RUNS     (default: 20)
- STREAM_TIMEOUT_S (default: 90)
- READ_IDLE_TIMEOUT_S (default: 30)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx


PROXY_BASE_URL = os.environ.get("PROXY_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "localtest-key")
TEST_MODEL = os.environ.get("TEST_MODEL", "moonshotai/kimi-k2-thinking")
FLAKE_RUNS = int(os.environ.get("FLAKE_RUNS", "20"))
STREAM_TIMEOUT_S = float(os.environ.get("STREAM_TIMEOUT_S", "90"))
READ_IDLE_TIMEOUT_S = float(os.environ.get("READ_IDLE_TIMEOUT_S", "30"))


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-API-Key": PROXY_API_KEY,
    }


def _url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"{PROXY_BASE_URL}{path}"


async def wait_for_health(timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[str] = None

    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                r = await client.get(_url("/health"))
                if r.status_code == 200:
                    data = r.json()
                    if data.get("status") == "ok":
                        return
                    last_err = f"health not ok: {data!r}"
                else:
                    last_err = f"health status_code={r.status_code} body={r.text[:200]!r}"
            except Exception as e:
                last_err = repr(e)
            await asyncio.sleep(0.3)

    raise AssertionError(f"Proxy not healthy at {PROXY_BASE_URL}. Last error: {last_err}")


@dataclass
class SSEEvent:
    event: str
    data: Any


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[SSEEvent]:
    """Parse text/event-stream (simple event/data lines) from proxy."""
    current_event: Optional[str] = None
    current_data_lines: list[str] = []

    async for line in response.aiter_lines():
        # SSE frames are separated by an empty line.
        if line == "":
            if current_event is not None:
                data_str = "\n".join(current_data_lines)
                data_obj: Any
                try:
                    data_obj = json.loads(data_str)
                except Exception:
                    data_obj = data_str
                yield SSEEvent(event=current_event, data=data_obj)
            current_event = None
            current_data_lines = []
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            current_data_lines.append(line[len("data:") :].lstrip())
            continue

    # flush trailing frame if any
    if current_event is not None:
        data_str = "\n".join(current_data_lines)
        try:
            data_obj = json.loads(data_str)
        except Exception:
            data_obj = data_str
        yield SSEEvent(event=current_event, data=data_obj)


async def post_json(path: str, payload: dict[str, Any], timeout_s: float = 60.0) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        return await client.post(_url(path), headers=_headers(), json=payload)


async def run_stream_request(
    *,
    prompt: str,
    max_tokens: int,
) -> tuple[list[SSEEvent], dict[str, Any]]:
    """Run a streaming request and return (events, response_headers)."""

    payload = {
        "model": TEST_MODEL,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }

    # httpx timeouts: overall + idle read watchdog
    timeout = httpx.Timeout(STREAM_TIMEOUT_S, connect=10.0, read=READ_IDLE_TIMEOUT_S, write=10.0, pool=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", _url("/v1/messages"), headers=_headers(), json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise AssertionError(
                    f"stream request failed: status={resp.status_code} body={body[:400]!r}"
                )

            events: list[SSEEvent] = []
            async for ev in iter_sse_events(resp):
                events.append(ev)
                if ev.event == "message_stop":
                    break

            # Drain remainder quickly (should end right after message_stop).
            # If the server keeps the connection open, this would hang; rely on read timeout.
            return events, dict(resp.headers)


def assert_has_event(events: list[SSEEvent], name: str) -> None:
    if not any(e.event == name for e in events):
        tail = [(e.event, e.data) for e in events[-10:]]
        raise AssertionError(f"missing event {name!r}. tail={tail!r}")


def assert_event_order(events: list[SSEEvent], first: str, later: str) -> None:
    pos_first = next((i for i, e in enumerate(events) if e.event == first), None)
    pos_later = next((i for i, e in enumerate(events) if e.event == later), None)
    if pos_first is None or pos_later is None or pos_first >= pos_later:
        raise AssertionError(
            f"bad order: expected {first} before {later}. "
            f"pos_first={pos_first} pos_later={pos_later}"
        )


def assert_has_text_delta(events: list[SSEEvent]) -> None:
    for e in events:
        if e.event != "content_block_delta":
            continue
        if not isinstance(e.data, dict):
            continue
        d = e.data.get("delta")
        if isinstance(d, dict) and d.get("type") == "text_delta" and (d.get("text") or ""):
            return
    tail = [(e.event, e.data) for e in events[-10:]]
    raise AssertionError(f"no non-empty text_delta observed. tail={tail!r}")


def assert_no_reasoning_leakage(events: list[SSEEvent]) -> None:
    """Assert that no SSE event contains reasoning/thinking fields."""
    for i, e in enumerate(events):
        if not isinstance(e.data, dict):
            continue

        # Check top-level data for reasoning fields
        data = e.data
        if "reasoning" in data:
            raise AssertionError(
                f"Event {i} ({e.event}) contains 'reasoning' field: {data.get('reasoning')!r}"
            )
        if "reasoning_content" in data:
            raise AssertionError(
                f"Event {i} ({e.event}) contains 'reasoning_content' field: {data.get('reasoning_content')!r}"
            )
        if "reasoning_details" in data:
            raise AssertionError(
                f"Event {i} ({e.event}) contains 'reasoning_details' field: {data.get('reasoning_details')!r}"
            )

        # Check nested structures (message, delta, content_block, etc.)
        for key in ["message", "delta", "content_block"]:
            if key in data and isinstance(data[key], dict):
                nested = data[key]
                if "reasoning" in nested:
                    raise AssertionError(
                        f"Event {i} ({e.event}).{key} contains 'reasoning': {nested.get('reasoning')!r}"
                    )
                if "reasoning_content" in nested:
                    raise AssertionError(
                        f"Event {i} ({e.event}).{key} contains 'reasoning_content': {nested.get('reasoning_content')!r}"
                    )
                if "reasoning_details" in nested:
                    raise AssertionError(
                        f"Event {i} ({e.event}).{key} contains 'reasoning_details': {nested.get('reasoning_details')!r}"
                    )


async def step(name: str, fn) -> None:
    print(f"[TEST] {name} ...", flush=True)
    await fn()
    print(f"[OK]   {name}", flush=True)


async def main() -> int:
    print(
        f"proxy={PROXY_BASE_URL} model={TEST_MODEL} flake_runs={FLAKE_RUNS} "
        f"stream_timeout_s={STREAM_TIMEOUT_S} idle_read_timeout_s={READ_IDLE_TIMEOUT_S}",
        flush=True,
    )

    # 1) health
    await step("1/10 health is ok", lambda: wait_for_health())

    # 2) count_tokens endpoint
    async def _count_tokens():
        r = await post_json(
            "/v1/messages/count_tokens",
            {
                "model": TEST_MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
            },
            timeout_s=20.0,
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert isinstance(data.get("input_tokens"), int) and data["input_tokens"] > 0, data

    await step("2/10 count_tokens works", _count_tokens)

    # 3) non-streaming baseline
    async def _non_streaming():
        r = await post_json(
            "/v1/messages",
            {
                "model": TEST_MODEL,
                "max_tokens": 64,
                "stream": False,
                "messages": [{"role": "user", "content": "Say: nonstream-ok"}],
            },
            timeout_s=60.0,
        )
        assert r.status_code == 200, r.text[:400]
        data = r.json()
        assert data.get("type") == "message", data
        assert data.get("role") == "assistant", data

    await step("3/10 non-streaming /v1/messages works", _non_streaming)

    # 4) streaming baseline with strict end + no reasoning leakage
    async def _streaming_minimal():
        events, _hdrs = await run_stream_request(prompt="Say: streaming-ok", max_tokens=128)
        assert_has_event(events, "message_start")
        assert_has_event(events, "content_block_start")
        assert_has_text_delta(events)
        assert_has_event(events, "message_stop")
        assert_event_order(events, "message_start", "message_stop")
        assert_no_reasoning_leakage(events)

    await step("4/10 streaming returns message_stop + no reasoning leak", _streaming_minimal)

    # 5) streaming provides X-Request-Id header
    async def _request_id_header():
        events, hdrs = await run_stream_request(prompt="Return any text", max_tokens=64)
        assert_has_event(events, "message_stop")
        rid = hdrs.get("x-request-id") or hdrs.get("X-Request-Id")
        assert rid and isinstance(rid, str), hdrs

    await step("5/10 streaming has X-Request-Id header", _request_id_header)

    # 6) 10 quick streaming runs
    async def _ten_runs():
        for i in range(1, 11):
            events, _ = await run_stream_request(prompt=f"run-{i}", max_tokens=64)
            assert_has_event(events, "message_stop")

    await step("6/10 10 quick streams finish", _ten_runs)

    # 7) flake detector: N runs (default 20) + reasoning leakage check
    async def _flake_runs():
        for i in range(1, FLAKE_RUNS + 1):
            events, _ = await run_stream_request(prompt=f"flake-{i}", max_tokens=96)
            assert_has_event(events, "message_stop")
            assert_no_reasoning_leakage(events)

    await step(f"7/10 flake detector ({FLAKE_RUNS} runs) + no reasoning leak", _flake_runs)

    # 8) long streaming response (more opportunities to stall) + reasoning check
    async def _long_stream():
        events, _ = await run_stream_request(
            prompt="Write 120 numbered items, one per line.",
            max_tokens=1200,
        )
        assert_has_event(events, "message_start")
        assert_has_text_delta(events)
        assert_has_event(events, "message_stop")
        assert_no_reasoning_leakage(events)

    await step("8/10 long stream finishes + no reasoning leak", _long_stream)

    # 9) ensure we emit at least one content_block_stop (proxy should close blocks)
    async def _has_block_stop():
        events, _ = await run_stream_request(prompt="Say: stop-blocks", max_tokens=128)
        assert_has_event(events, "content_block_stop")
        assert_has_event(events, "message_stop")

    await step("9/10 content_block_stop exists", _has_block_stop)

    # 10) sanity: model pass-through works (model contains '/')
    async def _model_passthrough():
        # This is more of a config regression check: proxy.py:1034-1036
        r = await post_json(
            "/v1/messages",
            {
                "model": TEST_MODEL,
                "max_tokens": 32,
                "stream": False,
                "messages": [{"role": "user", "content": "Say: model-passthrough"}],
            },
            timeout_s=60.0,
        )
        assert r.status_code == 200, r.text[:400]

    await step("10/10 model slug pass-through request succeeds", _model_passthrough)

    print("ALL TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
