"""纯标准库 .npz/.npy 读写：与 numpy 的 `np.savez_compressed` / `np.load` 互操作。

.npy 格式（version 1.0）：
    魔数 b"\\x93NUMPY" + 主版本 1 + 次版本 0 + 2 字节小端头长 + 头字典字面量
    头字典形如 {'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }
    以空格填充到 64 字节对齐并以换行结束；其后是 C 序原始数据。
.npz = ZIP 容器，每个成员为 <name>.npy（压缩方式 STORED 或 DEFLATED 均可读）。

本模块只依赖标准库，覆盖 dtype：<f8 <f4 <i8 <i4 u1；shape 支持 1-2 维。
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# dtype 描述符 -> (struct 格式字符, 每元素字节数)
_DTYPES: Dict[str, Tuple[str, int]] = {
    "<f8": ("d", 8),
    "<f4": ("f", 4),
    "<i8": ("q", 8),
    "<i4": ("i", 4),
    "u1": ("B", 1),
}
_ITEMSIZE = {"d": 8, "f": 4, "q": 8, "i": 4, "B": 1}


class NpzError(RuntimeError):
    """npz/npy 解析失败。"""


def _infer_descr(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return "u1"
    if isinstance(value, bool):
        return "<i8"
    if isinstance(value, int):
        return "<i8"
    return "<f8"


def _as_flat(value: Any, descr: str) -> List[Any]:
    """把嵌套列表/标量展平为 C 序一维；元素统一为 descr 对应类型。"""
    fmt = _DTYPES[descr][0]
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    if isinstance(value, (int, float, bool)) and not isinstance(value, (list, tuple)):
        value = [value]

    flat: List[Any] = []

    def _walk(v: Any) -> None:
        if isinstance(v, (list, tuple)):
            for item in v:
                _walk(item)
        elif isinstance(v, (bytes, bytearray)):
            flat.extend(v)
        elif isinstance(v, bool):
            flat.append(1 if v else 0)
        elif isinstance(v, int):
            flat.append(v)
        else:
            flat.append(float(v))

    _walk(value)
    if descr == "u1":
        return [int(x) & 0xFF for x in flat]
    if fmt in ("q", "i"):
        return [int(x) for x in flat]
    return [float(x) for x in flat]


def _shape_of(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (bytes, bytearray)):
        return (len(value),)
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        return (len(value),) + _shape_of(value[0])
    return ()  # 标量：零维


def _pack_npy(flat: Sequence[Any], shape: Tuple[int, ...], descr: str) -> bytes:
    fmt, size = _DTYPES[descr]
    shape_txt = ", ".join(str(s) for s in shape)
    if len(shape) == 1:
        shape_txt += ","  # 单元素元组必须带尾逗号，否则字面量解析成 int
    header = (
        "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }" % (descr, shape_txt)
    )
    # 64 字节对齐：magic(6) + version(2) + hlen(2) + header + '\n'
    prefix = 6 + 2 + 2
    pad = 64 - (prefix + len(header) + 1) % 64
    header = header + " " * pad + "\n"
    body = struct.pack("<%d%s" % (len(flat), fmt), *flat)
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin-1") + body


def _parse_npy(data: bytes) -> Tuple[List[Any], Tuple[int, ...], str]:
    if len(data) < 10 or data[:6] != b"\x93NUMPY":
        raise NpzError("不是合法的 .npy 数据（魔数不匹配）")
    major, minor = data[6], data[7]
    if major == 1:
        (hlen,) = struct.unpack_from("<H", data, 8)
        offset = 10
    elif major in (2, 3):
        (hlen,) = struct.unpack_from("<I", data, 8)
        offset = 12
    else:
        raise NpzError(f"不支持的 .npy 版本: {major}.{minor}")
    header = data[offset : offset + hlen].decode("latin-1")
    try:
        spec = eval(header, {"__builtins__": {}}, {})  # noqa: S307 - 头字典是字面量
    except SyntaxError as exc:
        raise NpzError(f".npy 头解析失败: {header!r}") from exc
    descr = str(spec["descr"])
    if descr not in _DTYPES:
        raise NpzError(f"不支持的 dtype: {descr}")
    shape = tuple(int(s) for s in spec["shape"])
    fmt, size = _DTYPES[descr]
    expected = size
    for s in shape:
        expected *= s
    body = data[offset + hlen :]
    if len(body) < expected:
        raise NpzError(f".npy 数据长度不足: 期望 {expected} 字节, 实际 {len(body)}")
    flat = list(struct.unpack_from("<%d%s" % (expected // size, fmt), body, 0))
    if descr == "u1":
        flat = [int(x) for x in flat]
    return flat, shape, descr


def save_npz(path: str, arrays: Dict[str, Any]) -> None:
    """把 {name: 嵌套列表/标量/bytes} 写成 .npz（DEFLATED，与 np.savez_compressed 同容器）。"""
    p = Path(path)
    if p.parent and str(p.parent):
        p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, value in arrays.items():
            descr = _infer_descr(value)
            flat = _as_flat(value, descr)
            shape = _shape_of(value)
            zf.writestr(f"{name}.npy", _pack_npy(flat, shape, descr))


def load_npz(path: str) -> Dict[str, Any]:
    """读取 .npz，返回 {name: 嵌套列表}；bytes 数组原样返回。"""
    p = Path(path)
    if not p.exists():
        raise NpzError(f"模型文件不存在: {path}")
    out: Dict[str, Any] = {}
    with zipfile.ZipFile(p, "r") as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            name = info.filename[: -len(".npy")]
            flat, shape, descr = _parse_npy(zf.read(info))
            out[name] = _reshape(flat, shape, descr)
    return out


def _reshape(flat: List[Any], shape: Tuple[int, ...], descr: str) -> Any:
    if descr == "u1":
        return bytes(int(x) & 0xFF for x in flat)

    def _build(dim: int, offset: int) -> Tuple[Any, int]:
        if dim == len(shape):
            return flat[offset], offset + 1
        rows = shape[dim]
        items = []
        for _ in range(rows):
            item, offset = _build(dim + 1, offset)
            items.append(item)
        return items, offset

    if not shape:
        return list(flat)
    value, _ = _build(0, 0)
    return value
