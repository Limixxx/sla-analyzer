#!/usr/bin/env python3
"""本地模拟的 OpenAI 兼容推理服务（仅用于 benchmark 工具自测，不参与生产）。

模拟行为：
- 基础 TTFT / TPOT 时延（可配置）；
- 在途请求超过 soft_capacity 后，排队等待与 TPOT 随负载抬升（模拟 Router 缓冲）；
- 在途请求超过 hard_capacity 后直接返回 503（模拟 Router 拒绝）；
- 支持流式（SSE）与非流式两种响应，末 chunk 带回 usage。

用法：
    python mock_server.py --port 8123 --ttft-ms 200 --tpot-ms 15 \
        --soft-capacity 8 --hard-capacity 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Dict

_WORDS = (
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
)


class MockState:
    def __init__(self, args: argparse.Namespace):
        self.ttft_ms = args.ttft_ms
        # 每 1024 个 prompt token 增加的 TTFT（ms），模拟 prefill 随上下文增长的耗时。
        # 置 0 表示 TTFT 与长度无关（旧行为）。
        self.ttft_per_1k_ms = args.ttft_per_1k_ms
        self.tpot_ms = args.tpot_ms
        self.soft_capacity = max(1, args.soft_capacity)
        self.hard_capacity = max(self.soft_capacity, args.hard_capacity)
        self.error_rate = args.error_rate
        self.in_flight = 0
        self.lock = asyncio.Lock()

    def load_factor(self) -> float:
        """在途/软容量的负载系数，>1 表示进入缓冲区。"""
        return self.in_flight / float(self.soft_capacity)

    def base_ttft_s(self, prompt_tokens: int) -> float:
        """无排队时的 TTFT（秒）：固定分量 + 与 prompt 长度相关的分量。"""
        extra = (prompt_tokens / 1024.0) * (self.ttft_per_1k_ms / 1000.0)
        return self.ttft_ms / 1000.0 + extra


async def _read_request(reader: asyncio.StreamReader) -> Dict:
    line = await reader.readline()
    if not line:
        return {}
    method, path, _ = line.decode("latin-1").split()
    headers: Dict[str, str] = {}
    while True:
        raw = await reader.readline()
        if not raw or raw in (b"\r\n", b"\n"):
            break
        text = raw.decode("latin-1").rstrip("\r\n")
        if ":" in text:
            key, value = text.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    body = b""
    length = int(headers.get("content-length", "0") or "0")
    if length:
        body = await reader.readexactly(length)
    return {"method": method, "path": path, "headers": headers, "body": body}


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5))


def _build_prompt(payload: Dict) -> str:
    if "messages" in payload and payload["messages"]:
        return str(payload["messages"][-1].get("content", ""))
    return str(payload.get("prompt", ""))


async def _send_json(writer: asyncio.StreamWriter, status: int, obj: Dict) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    head = (
        f"HTTP/1.1 {status} {'OK' if status == 200 else 'ERROR'}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("latin-1")
    writer.write(head + body)
    await writer.drain()


async def _send_sse_error(writer: asyncio.StreamWriter, status: int, message: str) -> None:
    body = json.dumps({"error": {"message": message, "type": "overloaded", "code": status}},
                      ensure_ascii=False).encode("utf-8")
    head = (
        f"HTTP/1.1 {status} Service Unavailable\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("latin-1")
    writer.write(head + body)
    await writer.drain()


async def _handle_chat(state: MockState, payload: Dict, writer: asyncio.StreamWriter) -> None:
    import random

    rng = random.Random(uuid.uuid4().int)
    prompt = _build_prompt(payload)
    prompt_tokens = _estimate_tokens(prompt)
    max_tokens = int(payload.get("max_tokens") or 64)
    stream = bool(payload.get("stream"))

    async with state.lock:
        state.in_flight += 1
        in_flight = state.in_flight
    try:
        if in_flight > state.hard_capacity:
            await _send_sse_error(writer, 503, "router overloaded: worker queue is full")
            return

        load = in_flight / float(state.soft_capacity)
        queue_wait_s = 0.0
        tpot_scale = 1.0
        if load > 1.0:
            # 缓冲区：排队等待与逐 token 时延同时抬升
            queue_wait_s = (load - 1.0) * 0.25
            tpot_scale = 1.0 + (load - 1.0) * 2.0

        if state.error_rate > 0 and rng.random() < state.error_rate:
            await _send_json(writer, 500, {"error": {"message": "simulated backend failure",
                                                     "type": "internal"}})
            return

        effective_ttft_s = state.base_ttft_s(prompt_tokens) + queue_wait_s
        effective_tpot_s = (state.tpot_ms / 1000.0) * tpot_scale

        if not stream:
            await asyncio.sleep(effective_ttft_s + effective_tpot_s * max_tokens)
            content = " ".join(_WORDS[i % len(_WORDS)] for i in range(max_tokens))
            await _send_json(writer, 200, {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model", "mock"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens,
                          "total_tokens": prompt_tokens + max_tokens},
            })
            return

        head = (
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
        ).encode("latin-1")
        writer.write(head)
        await writer.drain()

        cmpl_id = f"cmpl-{uuid.uuid4().hex[:12]}"

        async def emit(obj: Dict) -> None:
            writer.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
            await writer.drain()

        await asyncio.sleep(effective_ttft_s)
        await emit({"id": cmpl_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": payload.get("model", "mock"),
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        for i in range(max_tokens):
            await asyncio.sleep(effective_tpot_s)
            await emit({"id": cmpl_id, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": payload.get("model", "mock"),
                        "choices": [{"index": 0,
                                     "delta": {"content": _WORDS[i % len(_WORDS)] + " "},
                                     "finish_reason": None}]})
        await emit({"id": cmpl_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": payload.get("model", "mock"),
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        await emit({"id": cmpl_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": payload.get("model", "mock"), "choices": [],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens,
                              "total_tokens": prompt_tokens + max_tokens}})
        writer.write(b"data: [DONE]\n\n")
        await writer.drain()
    finally:
        async with state.lock:
            state.in_flight -= 1


async def _handle(state: MockState, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        request = await _read_request(reader)
        if not request:
            return
        if request["method"] == "GET" and request["path"].startswith("/v1/models"):
            await _send_json(writer, 200, {
                "object": "list",
                "data": [{"id": "mock-model", "object": "model", "created": int(time.time()),
                          "owned_by": "mock", "max_model_len": 32768}],
            })
            return
        if request["method"] == "GET" and request["path"] in ("/health", "/v1/health"):
            await _send_json(writer, 200, {"status": "ok"})
            return
        if request["method"] == "POST" and "/completions" in request["path"]:
            try:
                payload = json.loads(request["body"] or b"{}")
            except ValueError:
                await _send_json(writer, 400, {"error": {"message": "invalid json"}})
                return
            await _handle_chat(state, payload, writer)
            return
        await _send_json(writer, 404, {"error": {"message": f"not found: {request.get('path')}"}})
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    except Exception as exc:  # 保证 mock 服务不因单连接异常退出
        try:
            await _send_json(writer, 500, {"error": {"message": f"mock error: {exc}"}})
        except Exception:
            pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _serve(args: argparse.Namespace) -> None:
    state = MockState(args)
    server = await asyncio.start_server(lambda r, w: _handle(state, r, w), args.host, args.port)
    print(f"[mock] listening on http://{args.host}:{args.port}  "
          f"ttft={args.ttft_ms}ms ttft_per_1k={args.ttft_per_1k_ms}ms tpot={args.tpot_ms}ms "
          f"soft_capacity={state.soft_capacity} hard_capacity={state.hard_capacity} "
          f"error_rate={args.error_rate}")
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 兼容 mock 推理服务（benchmark 自测用）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--ttft-ms", type=float, default=200.0)
    parser.add_argument("--ttft-per-1k-tokens", type=float, default=0.0, dest="ttft_per_1k_ms",
                        help="每 1024 个 prompt token 额外增加的 TTFT（ms），0 表示与长度无关")
    parser.add_argument("--tpot-ms", type=float, default=15.0)
    parser.add_argument("--soft-capacity", type=int, default=8, help="超过后开始排队/降速")
    parser.add_argument("--hard-capacity", type=int, default=12, help="超过后直接 503 拒绝")
    parser.add_argument("--error-rate", type=float, default=0.0)
    args = parser.parse_args()
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        print("\n[mock] stopped")


if __name__ == "__main__":
    main()
