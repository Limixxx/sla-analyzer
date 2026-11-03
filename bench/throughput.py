"""容量面模型：``throughput_model.npz``（docs/07 §7.10.5）。

膝点只是单次运行的一个**点**；容量规划要反问题——"扛 X RPM @ 长度 L
需要多少并发？"。本模块把多次容量运行（每次一个固定输入长度）的膝点
汇成一张表，按长度线性插值回答反问题：

- 与在线模型的分工：``qos_model.npz`` 预测**没测过的负载点的质量**
  （MLP，docs/06）；本表面回答**容量规划反问题**（表格 + 插值，不训练）。
- 只存事实点（每次运行一个：``input_len`` / ``mu_rpm`` / ``mu_tps`` /
  ``input_tps`` / ``output_tps`` / ``req_conc`` / ``n_samples`` /
  ``a1_family``）与 ``meta_json``；全部由报告期从 StageStats 现算后汇写，
  派生量不落库（C3）。
- 合并规则：同一文件内按 L 覆盖更新（replace-per-L）；不同边界/配置的
  面对比用两个文件，不混装（meta 记录 base_url 与来源运行）。
- 查询：L 超出测量范围时 clamp 到端点并标 ``extrapolated``（对应在线
  模型的包络守卫角色）——外插结果只作参考，不进验收结论。

存储细节：``npz.py`` 只承载数值与 bytes（纯机制，零改动）。字符串
（A1 家族标签）因此编码为整数下标，标签表随 ``meta_json`` 落盘自描述；
缺失的 token 速率存 NaN，读回时还原为 None（不静默填 0）。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .npz import NpzError, load_npz, save_npz

SURFACE_KIND = "throughput_surface_v1"

# A1 家族编码（下标即代码，标签表随 meta_json 自描述）：
# 0=none（该点无失败） 1=timeout（超时族） 2=reject（拒绝族） 3=other
A1_FAMILY_LABELS: Tuple[str, ...] = ("none", "timeout", "reject", "other")
_FAMILY_TO_CODE: Dict[str, int] = {label: code for code, label in enumerate(A1_FAMILY_LABELS)}
_CODE_TO_FAMILY: Dict[int, str] = {code: label for label, code in _FAMILY_TO_CODE.items()}

_REQUIRED_ARRAYS = (
    "lengths", "mu_rpm", "mu_tps", "input_tps", "output_tps",
    "req_conc", "n_samples", "a1_family",
)


class ThroughputError(RuntimeError):
    """容量面文件缺失、损坏或 schema 不匹配。"""


@dataclass
class ThroughputPoint:
    """一次容量运行的膝点事实（固定输入长度下测得）。"""

    input_len: int                 # 该运行的输入长度（token 数）
    mu_rpm: float                   # 峰值有效完成速率 ×60（请求/分钟）
    mu_tps: Optional[float]         # 峰值全量 token 速率（input+output）
    input_tps: Optional[float]      # 峰值输入 token 速率（prefill 侧）
    output_tps: Optional[float]     # 峰值输出 token 速率（decode 侧）
    req_conc: int                   # 取得峰值吞吐的负载水平（闭环为并发数）
    n_samples: int                  # 容量读数级样本数
    a1_family: str = "none"         # 膝点级 A1 的主导家族
    source: str = ""                # 来源运行标识（db 路径），仅入 meta


@dataclass
class CapacityAnswer:
    """``--capacity-plan`` 的反问题答案（长度 L 处扛 target_rpm 的代价）。"""

    target_rpm: float
    input_len: int                  # 请求的长度
    effective_len: int              # clamp 后实际用于查表的长度
    extrapolated: bool              # L 是否超出测量范围（clamp 到端点）
    mu_rpm: Optional[float]         # 该长度处预测容量（RPM）
    mu_tps: Optional[float]         # 分解：全量 token 速率
    input_tps: Optional[float]      # 分解：输入侧
    output_tps: Optional[float]     # 分解：输出侧
    req_conc: Optional[int]         # 该长度处取得峰值吞吐的负载水平（参考并发）
    margin_rpm: Optional[float]     # mu_rpm - target_rpm（负 = 目标超容量）
    achievable: Optional[bool]      # 目标是否在实测容量内
    a1_family: str = "none"         # 最近测量点的过载形态
    notes: List[str] = field(default_factory=list)


def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """折线线性插值；调用方保证 xs 升序、x 在 [xs[0], xs[-1]] 内。"""
    if len(xs) == 1:
        return float(ys[0])
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= x <= x1:
            if x1 == x0:
                return float(ys[i])
            t = (x - x0) / (x1 - x0)
            return float(ys[i]) + t * (float(ys[i + 1]) - float(ys[i]))
    # x 恰好等于右端点时上面循环不命中，落到这里
    return float(ys[-1])


def _nan_to_none(value: Any) -> Optional[float]:
    f = float(value)
    return None if math.isnan(f) else f


def _or_nan(value: Optional[float]) -> float:
    return float("nan") if value is None else float(value)


@dataclass
class ThroughputSurface:
    """容量面：按输入长度索引的膝点表 + meta（边界、来源运行）。"""

    points: List[ThroughputPoint] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def fit(cls, points: Sequence[ThroughputPoint], meta: Optional[Dict[str, Any]] = None) -> "ThroughputSurface":
        """由本次运行（可多个）的膝点构建容量面（按 L 去重，后者覆盖前者）。"""
        return cls(points=_dedup_by_len(points), meta=dict(meta or {}))

    def merge(self, points: Sequence[ThroughputPoint]) -> List[int]:
        """replace-per-L 合并：返回被覆盖（含新增）的长度列表。"""
        merged = _dedup_by_len(list(self.points) + list(points))
        self.points = merged
        return sorted({int(p.input_len) for p in points})

    def query(self, target_rpm: float, input_len: int) -> CapacityAnswer:
        """反问题：长度 input_len 处扛 target_rpm RPM 的预测容量与所需并发。"""
        notes: List[str] = []
        if not self.points:
            return CapacityAnswer(
                target_rpm=target_rpm, input_len=input_len, effective_len=input_len,
                extrapolated=False, mu_rpm=None, mu_tps=None, input_tps=None,
                output_tps=None, req_conc=None, margin_rpm=None, achievable=None,
                notes=["容量面为空（尚无任何膝点数据），先跑一次含吞吐报告的运行。"],
            )
        pts = sorted(self.points, key=lambda p: p.input_len)
        lo, hi = pts[0].input_len, pts[-1].input_len
        eff = min(max(int(input_len), lo), hi)
        extrapolated = int(input_len) < lo or int(input_len) > hi
        if extrapolated:
            notes.append(
                f"请求长度 {input_len} 超出测量范围 [{lo}, {hi}]，已 clamp 到 {eff}；"
                "外插结果只作参考，不作为验收依据。"
            )

        xs = [float(p.input_len) for p in pts]
        mu_rpm = _interp(xs, [p.mu_rpm for p in pts], eff)
        mu_tps = _interp_optional(xs, [p.mu_tps for p in pts], eff)
        input_tps = _interp_optional(xs, [p.input_tps for p in pts], eff)
        output_tps = _interp_optional(xs, [p.output_tps for p in pts], eff)
        req_conc = max(1, round(_interp(xs, [float(p.req_conc) for p in pts], eff)))
        nearest = min(pts, key=lambda p: (abs(p.input_len - eff), p.input_len))

        if mu_tps is None:
            notes.append("该长度附近缺少完整 token 数据（usage 未上报），仅 RPM 口径可信。")
        margin = mu_rpm - target_rpm
        achievable = mu_rpm >= target_rpm
        if not achievable:
            notes.append(
                f"目标 {target_rpm:g} RPM 超出长度 {eff} 处的实测容量 "
                f"{mu_rpm:.1f} RPM：需扩容、缩短输入或重谈窗口承诺。"
            )
        return CapacityAnswer(
            target_rpm=target_rpm, input_len=int(input_len), effective_len=int(eff),
            extrapolated=extrapolated, mu_rpm=mu_rpm, mu_tps=mu_tps,
            input_tps=input_tps, output_tps=output_tps, req_conc=req_conc,
            margin_rpm=margin, achievable=achievable,
            a1_family=nearest.a1_family, notes=notes,
        )

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """写 .npz（npz.py 纯机制；字符串编码为整数、缺失值存 NaN）。"""
        pts = sorted(self.points, key=lambda p: p.input_len)
        arrays: Dict[str, Any] = {
            "lengths": [int(p.input_len) for p in pts],
            "mu_rpm": [float(p.mu_rpm) for p in pts],
            "mu_tps": [_or_nan(p.mu_tps) for p in pts],
            "input_tps": [_or_nan(p.input_tps) for p in pts],
            "output_tps": [_or_nan(p.output_tps) for p in pts],
            "req_conc": [int(p.req_conc) for p in pts],
            "n_samples": [int(p.n_samples) for p in pts],
            "a1_family": [_FAMILY_TO_CODE.get(p.a1_family, _FAMILY_TO_CODE["other"]) for p in pts],
        }
        meta = dict(self.meta)
        meta["kind"] = SURFACE_KIND
        meta["a1_family_labels"] = list(A1_FAMILY_LABELS)
        meta["n_points"] = len(pts)
        meta.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        meta["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        sources = sorted({p.source for p in pts if p.source})
        if sources:
            meta["sources"] = sources
        arrays["meta_json"] = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        save_npz(path, arrays)

    @classmethod
    def load(cls, path: str) -> "ThroughputSurface":
        try:
            data = load_npz(path)
        except NpzError as exc:
            # 统一出口：调用方只处理 ThroughputError，不关心底层容器细节
            raise ThroughputError(str(exc)) from exc
        meta_raw = data.get("meta_json")
        if isinstance(meta_raw, (bytes, bytearray)):
            meta = json.loads(bytes(meta_raw).decode("utf-8"))
        else:
            meta = {}
        if meta.get("kind") != SURFACE_KIND:
            raise ThroughputError(
                f"容量面文件 kind 不匹配: {meta.get('kind')!r}（期望 {SURFACE_KIND!r}）"
            )
        missing = [k for k in _REQUIRED_ARRAYS if k not in data]
        if missing:
            raise ThroughputError(f"容量面文件缺少数组: {missing}")
        n = len(data["lengths"])
        for key in _REQUIRED_ARRAYS:
            if len(data[key]) != n:
                raise ThroughputError(
                    f"容量面文件数组长度不一致: {key} 为 {len(data[key])}，lengths 为 {n}"
                )
        points = [
            ThroughputPoint(
                input_len=int(data["lengths"][i]),
                mu_rpm=float(data["mu_rpm"][i]),
                mu_tps=_nan_to_none(data["mu_tps"][i]),
                input_tps=_nan_to_none(data["input_tps"][i]),
                output_tps=_nan_to_none(data["output_tps"][i]),
                req_conc=int(data["req_conc"][i]),
                n_samples=int(data["n_samples"][i]),
                a1_family=_CODE_TO_FAMILY.get(int(data["a1_family"][i]), "other"),
            )
            for i in range(n)
        ]
        return cls(points=points, meta=meta)


def _interp_optional(xs: Sequence[float], ys: Sequence[Optional[float]], x: float) -> Optional[float]:
    """可缺值插值：任一侧缺失即整体缺失（不静默填 0）。"""
    if any(v is None for v in ys):
        return None
    return _interp(xs, ys, x)


def _dedup_by_len(points: Sequence[ThroughputPoint]) -> List[ThroughputPoint]:
    """按 input_len 去重（同长度后者覆盖前者），按长度升序。"""
    by_len: Dict[int, ThroughputPoint] = {}
    for p in points:
        by_len[int(p.input_len)] = p
    return [by_len[k] for k in sorted(by_len)]


def surface_path_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()
