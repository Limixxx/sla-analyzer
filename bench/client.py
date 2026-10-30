"""单请求执行器：构造 OpenAI 兼容请求、消费 SSE 流、产出 RequestResult。

时间口径（全部基于 time.monotonic()，与墙钟无关，避免系统时钟抖动）：
- connect_ms : 建连耗时
- ttft_ms    : 请求写出完成 → 首个含内容的 chunk（首 token 时间）
- tpot_ms    : 首 token → 末 token 之间，平均每个输出 token 的间隔
- e2e_ms     : 请求写出完成 → 流结束
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import RequestConfig, ServiceConfig
from .httpclient import open_stream
from .prompts import RequestSpec

# 失败类型（A1 的子类，落库 error_kind）
KIND_REJECTED = "rejected"            # Router 过载拒绝（429/503）
KIND_HTTP_4XX = "http_4xx"
KIND_HTTP_5XX = "http_5xx"
KIND_TIMEOUT = "timeout"              # 整体超时
KIND_CONNECT_TIMEOUT = "connect_timeout"
KIND_CONN = "conn_error"              # 连接失败/重置
KIND_STREAM_ERROR = "stream_error"    # 流内错误事件
KIND_INCOMPLETE = "incomplete"        # 流异常结束且无内容
KIND_EMPTY = "empty_completion"       # 成功但 0 输出 token
KIND_PROTOCOL = "protocol_error"      # 响应解析失败
KIND_ABORTED = "aborted"              # 本级结束时仍在途、被放弃的请求（不计入 A1/A2/A3）


@dataclass
class RequestResult:
    """一次请求的完整特征记录（不保存请求/响应原文）。"""

    stage_id: int = 0
    concurrency: int = 0
    worker_id: int = 0
    is_warmup: int = 0
    mode: str = "closed"

    sent_at: float = 0.0               # time.time()，请求写出的墙钟
    connect_ms: float = 0.0
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    e2e_ms: Optional[float] = None

    prompt_tokens_planned: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    max_tokens: int = 0
    token_source: str = "none"         # usage | estimated | none

    status: str = "failed"             # ok | failed
    error_kind: Optional[str] = None
    http_code: Optional[int] = None
    finish_reason: Optional[str] = None
    note: str = ""

    latency_class: Optional[str] = None  # A1 | A2 | A3，由 metrics 回填

    def as_row(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "concurrency": self.concurrency,
            "worker_id": self.worker_id,
            "is_warmup": self.is_warmup,
            "mode": self.mode,
            "sent_at": self.sent_at,
            "connect_ms": self.connect_ms,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "e2e_ms": self.e2e_ms,
            "prompt_tokens_planned": self.prompt_tokens_planned,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "max_tokens": self.max_tokens,
            "token_source": self.token_source,
            "status": self.status,
            "error_kind": self.error_kind,
            "http_code": self.http_code,
            "finish_reason": self.finish_reason,
            "latency_class": self.latency_class,
            "note": self.note,
        }


def build_payload(svc: ServiceConfig, req: RequestConfig, prompt: str, spec: RequestSpec) -> Dict[str, Any]:
    """按 chat/completions 两种端点形态构造请求体。"""
    payload: Dict[str, Any] = {"model": svc.model, "stream": req.stream}
    if svc.path.rstrip("/").endswith("/completions") and "chat" not in svc.path:
        payload["prompt"] = prompt
    else:
        payload["messages"] = [{"role": "user", "content": prompt}]
    payload["max_tokens"] = spec.max_tokens
    payload["temperature"] = req.temperature
    if req.stream:
        # include_usage 让末 chunk 带回真实 token 统计
        payload["stream_options"] = {"include_usage": True}
    return payload


def _auth_headers(svc: ServiceConfig) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if svc.api_key:
        headers["Authorization"] = f"Bearer {svc.api_key}"
    headers.update(svc.extra_headers)
    return headers


async def execute_request(
    svc: ServiceConfig,
    req: RequestConfig,
    prompt: str,
    spec: RequestSpec,
    stage_id: int,
    concurrency: int,
    worker_id: int,
    is_warmup: int = 0,
    mode: str = "closed",
) -> RequestResult:
    """执行单次请求，整体受 svc.timeout_s 保护（分钟级请求的关键保护）。"""
    try:
        return await asyncio.wait_for(
            _execute_request_inner(svc, req, prompt, spec, stage_id, concurrency, worker_id, is_warmup, mode),
            timeout=svc.timeout_s,
        )
    except asyncio.TimeoutError:
        result = RequestResult(
            stage_id=stage_id,
            concurrency=concurrency,
            worker_id=worker_id,
            is_warmup=is_warmup,
            mode=mode,
            sent_at=time.time(),
            prompt_tokens_planned=spec.prompt_tokens_target,
            max_tokens=spec.max_tokens,
            error_kind=KIND_TIMEOUT,
            note=f"total timeout after {svc.timeout_s}s",
        )
        return result


async def _execute_request_inner(
    svc: ServiceConfig,
    req: RequestConfig,
    prompt: str,
    spec: RequestSpec,
    stage_id: int,
    concurrency: int,
    worker_id: int,
    is_warmup: int,
    mode: str,
) -> RequestResult:
    result = RequestResult(
        stage_id=stage_id,
        concurrency=concurrency,
        worker_id=worker_id,
        is_warmup=is_warmup,
        mode=mode,
        sent_at=time.time(),
        prompt_tokens_planned=spec.prompt_tokens_target,
        max_tokens=spec.max_tokens,
    )

    url = svc.base_url.rstrip("/") + svc.path
    body = json.dumps(build_payload(svc, req, prompt, spec)).encode("utf-8")
    headers = _auth_headers(svc)

    writer = None
    t_start = time.monotonic()
    try:
        t_conn = time.monotonic()
        reader, writer, response = await open_stream(
            url, "POST", headers, body, svc.connect_timeout_s, svc.verify_tls
        )
        result.connect_ms = round((time.monotonic() - t_conn) * 1000.0, 3)
        result.http_code = response.status

        if response.status != 200:
            text = await response.read_text(limit=2048)
            lowered = text.lower()
            if response.status in (429, 503) or "overload" in lowered or "queue" in lowered and "full" in lowered:
                result.error_kind = KIND_REJECTED
            elif 400 <= response.status < 500:
                result.error_kind = KIND_HTTP_4XX
            else:
                result.error_kind = KIND_HTTP_5XX
            result.note = text[:300]
            return result

        if not req.stream:
            raw = await response.read_text(limit=1 << 22)
            obj = json.loads(raw)
            usage = obj.get("usage") or {}
            content = ""
            choices = obj.get("choices") or []
            if choices:
                content = (choices[0].get("message") or {}).get("content") or ""
                result.finish_reason = choices[0].get("finish_reason")
            result.ttft_ms = result.e2e_ms = round((time.monotonic() - t_start) * 1000.0, 3)
            result.tpot_ms = None
            _fill_tokens(result, usage, content, req)
            _finalize_success(result)
            return result

        # ---- 流式：逐行消费 SSE ----
        content_parts = []
        first_token_at: Optional[float] = None
        last_token_at: Optional[float] = None
        usage: Dict[str, Any] = {}
        stream_error: Optional[str] = None

        async for raw_line in response.lines():
            line = raw_line.strip()
            if not line or line.startswith(b":"):
                continue
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except (ValueError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            if "error" in obj and obj["error"]:
                stream_error = json.dumps(obj["error"], ensure_ascii=False)[:300]
                break
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                result.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            if content:
                now = time.monotonic()
                if first_token_at is None:
                    first_token_at = now
                last_token_at = now
                content_parts.append(content)

        content_text = "".join(content_parts)
        result.e2e_ms = round((time.monotonic() - t_start) * 1000.0, 3)

        if stream_error is not None:
            result.error_kind = KIND_STREAM_ERROR
            result.note = stream_error
            return result
        if first_token_at is None:
            if not content_text and not usage:
                result.error_kind = KIND_INCOMPLETE
                result.note = "stream ended without any content"
                return result
            # 只有 usage 没有内容（极少见）：视为空输出
            result.error_kind = KIND_EMPTY
            return result

        result.ttft_ms = round((first_token_at - t_start) * 1000.0, 3)
        if last_token_at is not None and last_token_at > first_token_at:
            span_ms = (last_token_at - first_token_at) * 1000.0
            n_out = _token_count(result, usage, content_text, req)
            denom = max(1, n_out - 1)
            result.tpot_ms = round(span_ms / denom, 3)
        _fill_tokens(result, usage, content_text, req)
        _finalize_success(result)
        return result

    except asyncio.TimeoutError:
        result.error_kind = KIND_TIMEOUT
        result.note = f"total timeout after {svc.timeout_s}s"
        return result
    except (ConnectionError, OSError) as exc:
        if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
            result.error_kind = KIND_CONNECT_TIMEOUT
        else:
            result.error_kind = KIND_CONN
        result.note = f"{type(exc).__name__}: {exc}"[:300]
        return result
    except asyncio.IncompleteReadError as exc:
        result.error_kind = KIND_INCOMPLETE
        result.note = f"incomplete read: {exc}"[:300]
        return result
    except Exception as exc:  # 兜底，保证压测不中断
        result.error_kind = KIND_PROTOCOL
        result.note = f"{type(exc).__name__}: {exc}"[:300]
        return result
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass


def _token_count(result: RequestResult, usage: Dict[str, Any], content: str, req: RequestConfig) -> int:
    if usage.get("completion_tokens"):
        return int(usage["completion_tokens"])
    return max(1, int(len(content) / max(1.0, req.chars_per_token)))


def _fill_tokens(result: RequestResult, usage: Dict[str, Any], content: str, req: RequestConfig) -> None:
    if usage.get("prompt_tokens"):
        result.prompt_tokens = int(usage["prompt_tokens"])
    if usage.get("completion_tokens"):
        result.completion_tokens = int(usage["completion_tokens"])
        result.token_source = "usage"
    else:
        result.completion_tokens = max(1, int(len(content) / max(1.0, req.chars_per_token)))
        result.token_source = "estimated"


def _finalize_success(result: RequestResult) -> None:
    if result.completion_tokens is not None and result.completion_tokens <= 0:
        result.status = "failed"
        result.error_kind = KIND_EMPTY
        return
    result.status = "ok"
