"""零依赖的异步 HTTP/1.1 客户端，支持 SSE 流式响应。

仅依赖标准库（asyncio/ssl），便于直接拷贝进集群 Pod 运行。
每次请求使用独立连接（Connection: close），在压测规模下（数百并发）开销可接受。
"""

from __future__ import annotations

import asyncio
import ssl
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlsplit


class HttpStatusError(Exception):
    def __init__(self, status: int, reason: str):
        super().__init__(f"HTTP {status} {reason}")
        self.status = status
        self.reason = reason


@dataclass
class StreamResponse:
    status: int
    reason: str
    headers: Dict[str, str] = field(default_factory=dict)

    async def lines(self):
        """异步迭代响应体中的行（含 SSE 的 data: 行）。"""
        raise NotImplementedError

    async def read_text(self, limit: int = 4096) -> str:
        """读取响应体文本（用于错误响应，限量读取）。"""
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


class _StreamResponseImpl(StreamResponse):
    def __init__(self, status: int, reason: str, headers: Dict[str, str], reader: asyncio.StreamReader,
                 framing: str, content_length: Optional[int]):
        super().__init__(status, reason, headers)
        self._reader = reader
        self._framing = framing
        self._content_length = content_length
        self._buf = bytearray()
        self._done = False

    async def lines(self):
        if self._framing == "chunked":
            async for chunk in self._iter_chunked():
                self._buf.extend(chunk)
                for line in self._drain_lines():
                    yield line
        elif self._framing == "length":
            assert self._content_length is not None
            remaining = self._content_length
            while remaining > 0:
                piece = await self._reader.read(min(65536, remaining))
                if not piece:
                    break
                remaining -= len(piece)
                self._buf.extend(piece)
                for line in self._drain_lines():
                    yield line
        else:  # close
            while True:
                piece = await self._reader.read(65536)
                if not piece:
                    break
                self._buf.extend(piece)
                for line in self._drain_lines():
                    yield line
        self._done = True
        for line in self._drain_lines(final=True):
            yield line

    async def read_text(self, limit: int = 4096) -> str:
        collected = bytearray()
        try:
            async for chunk in self._raw_body():
                collected.extend(chunk)
                if len(collected) >= limit:
                    break
        except Exception:
            pass
        return bytes(collected[:limit]).decode("utf-8", errors="replace")

    async def close(self) -> None:
        self._done = True

    # ------------------------------------------------------------------ #
    async def _raw_body(self):
        if self._framing == "chunked":
            async for chunk in self._iter_chunked():
                yield chunk
        elif self._framing == "length":
            assert self._content_length is not None
            remaining = self._content_length
            while remaining > 0:
                piece = await self._reader.read(min(65536, remaining))
                if not piece:
                    break
                remaining -= len(piece)
                yield piece
        else:
            while True:
                piece = await self._reader.read(65536)
                if not piece:
                    break
                yield piece

    async def _iter_chunked(self):
        while True:
            size_line = await self._reader.readline()
            if not size_line:
                return
            token = size_line.strip().split(b";", 1)[0]
            try:
                size = int(token, 16)
            except ValueError:
                return
            if size == 0:
                # 读取 trailer 直至空行
                while True:
                    trailer = await self._reader.readline()
                    if not trailer or trailer in (b"\r\n", b"\n"):
                        break
                return
            data = await self._reader.readexactly(size)
            await self._reader.readexactly(2)  # CRLF
            yield data

    def _drain_lines(self, final: bool = False):
        lines: List[bytes] = []
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(self._buf[:nl])
            del self._buf[: nl + 1]
            lines.append(line.rstrip(b"\r"))
        if final and self._buf:
            lines.append(bytes(self._buf).rstrip(b"\r"))
            self._buf.clear()
        return lines


async def open_stream(
    url: str,
    method: str,
    headers: Dict[str, str],
    body: Optional[bytes],
    connect_timeout_s: float,
    verify_tls: bool,
) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter, StreamResponse]:
    """建立连接并发送请求，返回 (reader, writer, response)。

    调用方负责在结束后关闭 writer。
    """
    split = urlsplit(url)
    scheme = split.scheme or "http"
    host = split.hostname or "127.0.0.1"
    port = split.port or (443 if scheme == "https" else 80)
    path = split.path or "/"
    if split.query:
        path = f"{path}?{split.query}"

    sslctx: Optional[ssl.SSLContext] = None
    if scheme == "https":
        if verify_tls:
            sslctx = ssl.create_default_context()
        else:
            sslctx = ssl._create_unverified_context()

    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, port, ssl=sslctx),
        timeout=connect_timeout_s,
    )

    req_headers = {
        "Host": f"{host}:{port}",
        "Connection": "close",
        "Accept": "text/event-stream, application/json",
        "User-Agent": "pd-qos-benchmark/1.0",
    }
    req_headers.update(headers)
    if body is not None:
        req_headers["Content-Type"] = "application/json"
        req_headers["Content-Length"] = str(len(body))

    request = f"{method} {path} HTTP/1.1\r\n".encode("ascii")
    for key, value in req_headers.items():
        request += f"{key}: {value}\r\n".encode("latin-1")
    request += b"\r\n"
    if body:
        request += body

    writer.write(request)
    await writer.drain()

    status_line = await reader.readline()
    if not status_line:
        raise ConnectionError("empty status line from server")
    parts = status_line.decode("latin-1").strip().split(" ", 2)
    if len(parts) < 2 or not parts[0].startswith("HTTP/"):
        raise ConnectionError(f"malformed status line: {status_line!r}")
    status = int(parts[1])
    reason = parts[2].strip() if len(parts) > 2 else ""

    resp_headers: Dict[str, str] = {}
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            break
        text = line.decode("latin-1").rstrip("\r\n")
        if ":" in text:
            key, value = text.split(":", 1)
            resp_headers[key.strip().lower()] = value.strip()

    transfer = resp_headers.get("transfer-encoding", "").lower()
    content_length_header = resp_headers.get("content-length")
    if "chunked" in transfer:
        framing, content_length = "chunked", None
    elif content_length_header is not None and content_length_header.isdigit():
        framing, content_length = "length", int(content_length_header)
    else:
        framing, content_length = "close", None

    response = _StreamResponseImpl(status, reason, resp_headers, reader, framing, content_length)
    return reader, writer, response
