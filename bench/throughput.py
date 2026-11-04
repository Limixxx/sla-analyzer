"""容量面模型：``throughput_model.npz``（docs/07 §7.10.5）。

膝点/平台只是单次运行的一个**点**；容量规划要反问题——"扛 X RPM @
(L_in, L_out) 需要多少并发？"。本模块把多次容量运行（每次一个固定
(输入, 输出) 长度组合）的读数汇成一张二维表，按双线性插值回答反问题：

- 与在线模型的分工：``qos_model.npz`` 预测**没测过的负载点的质量**
  （MLP，docs/06）；本表面回答**容量规划反问题**（表格 + 插值，不训练）。
- 只存事实点（每次运行一个：``input_len`` / ``output_len`` / ``mu_rpm`` /
  ``mu_tps`` / ``input_tps`` / ``output_tps`` / ``req_conc`` / ``n_samples`` /
  ``a1_family`` / ``origin``）与 ``meta_json``；全部由报告期从 StageStats
  现算后汇写，派生量不落库（C3）。
- 合并规则：同一文件内按 (L_in, L_out) 单元格覆盖更新（replace-per-cell，
  D22——输出长度是批量容量的主控变量，不同 L_out 的 μ 不可混进同一格）；
  不同边界/配置的面对比用两个文件，不混装（meta 记录 base_url 与来源运行）。
- 查询：任一坐标超出测量范围时 clamp 到端点并标 ``extrapolated``（对应
  在线模型的包络守卫角色）——外插结果只作参考，不进验收结论；网格角点
  未全部实测时回退最近实测点并显式说明，不静默插值。

存储细节：``npz.py`` 只承载数值与 bytes（纯机制，零改动）。字符串
（A1 家族、读数来源标签）因此编码为整数下标，标签表随 ``meta_json``
落盘自描述；缺失的 token 速率存 NaN，读回时还原为 None（不静默填 0）。
旧版一维面（无 output_lens/origins 数组）读回时 output_len=0、
origin="knee"，查询自动退化为按输入长度的一维插值。
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

# 读数来源编码（D23）：0=knee（膝点实测） 1=plateau（软退化平台估计）。
# 与 source（db 路径审计字段）分开：来源是判定口径，路径是出处，不能互相覆盖。
ORIGIN_LABELS: Tuple[str, ...] = ("knee", "plateau")
_ORIGIN_TO_CODE: Dict[str, int] = {label: code for code, label in enumerate(ORIGIN_LABELS)}

_REQUIRED_ARRAYS = (
    "lengths", "mu_rpm", "mu_tps", "input_tps", "output_tps",
    "req_conc", "n_samples", "a1_family",
)
# 旧版一维面缺省的数组：读回守卫（缺 → output_len=0 / origin="knee"）
_OPTIONAL_ARRAYS = ("output_lens", "origins")


class ThroughputError(RuntimeError):
    """容量面文件缺失、损坏或 schema 不匹配。"""


@dataclass
class ThroughputPoint:
    """一次容量运行的读数事实（固定 (输入, 输出) 长度组合下测得）。"""

    input_len: int                 # 该运行的输入长度（token 数）
    mu_rpm: float                   # 峰值有效完成速率 ×60（请求/分钟）
    mu_tps: Optional[float]         # 峰值全量 token 速率（input+output）
    input_tps: Optional[float]      # 峰值输入 token 速率（prefill 侧）
    output_tps: Optional[float]     # 峰值输出 token 速率（decode 侧）
    req_conc: int                   # 取得峰值吞吐的负载水平（闭环为并发数）
    n_samples: int                  # 容量读数级样本数
    a1_family: str = "none"         # 膝点级 A1 的主导家族（平台读数恒为 none）
    source: str = ""                # 来源运行标识（db 路径），仅入 meta
    output_len: int = 0             # 输出长度坐标（D22）；0 = 旧版一维面/未记录
    origin: str = "knee"            # 读数来源（D23）：knee=膝点实测 | plateau=平台估计


@dataclass
class CapacityAnswer:
    """``--capacity-plan`` 的反问题答案（(L_in, L_out) 处扛 target_rpm 的代价）。"""

    target_rpm: float
    input_len: int                  # 请求的输入长度
    effective_len: int              # clamp 后实际用于查表的输入长度
    extrapolated: bool              # 坐标是否超出测量范围（clamp 到端点）
    mu_rpm: Optional[float]         # 该坐标处预测容量（RPM）
    mu_tps: Optional[float]         # 分解：全量 token 速率
    input_tps: Optional[float]      # 分解：输入侧
    output_tps: Optional[float]     # 分解：输出侧
    req_conc: Optional[int]         # 该坐标处取得峰值吞吐的负载水平（参考并发）
    margin_rpm: Optional[float]     # mu_rpm - target_rpm（负 = 目标超容量）
    achievable: Optional[bool]      # 目标是否在实测容量内
    a1_family: str = "none"         # 最近测量点的过载形态
    notes: List[str] = field(default_factory=list)
    output_len: int = 0             # 请求的输出长度（0 = 未指定）
    effective_output_len: int = 0   # clamp/回退后实际用于查表的输出长度
    origin: str = "knee"            # 最近测量点的读数来源（knee/plateau）


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
    """容量面：按 (输入, 输出) 长度单元格索引的读数表 + meta（边界、来源运行）。"""

    points: List[ThroughputPoint] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def fit(cls, points: Sequence[ThroughputPoint], meta: Optional[Dict[str, Any]] = None) -> "ThroughputSurface":
        """由本次运行（可多个）的读数构建容量面（按单元格去重，后者覆盖前者）。"""
        return cls(points=_dedup_by_cell(points), meta=dict(meta or {}))

    def merge(self, points: Sequence[ThroughputPoint]) -> List[Tuple[int, int]]:
        """replace-per-(L_in, L_out) 合并：返回被覆盖（含新增）的坐标列表。"""
        merged = _dedup_by_cell(list(self.points) + list(points))
        self.points = merged
        return sorted({(int(p.input_len), int(p.output_len)) for p in points})

    def query(self, target_rpm: float, input_len: int, output_len: int = 0) -> CapacityAnswer:
        """反问题：(L_in, L_out) 处扛 target_rpm RPM 的预测容量与所需并发。

        output_len=0 表示未指定：一维面（未记录输出长度）按输入长度插值；
        二维面取最小输出档作答并提示显式指定——不同输出档的容量不可比，
        不猜坐标（D22）。
        """
        notes: List[str] = []
        if not self.points:
            return CapacityAnswer(
                target_rpm=target_rpm, input_len=input_len, effective_len=input_len,
                extrapolated=False, mu_rpm=None, mu_tps=None, input_tps=None,
                output_tps=None, req_conc=None, margin_rpm=None, achievable=None,
                notes=["容量面为空（尚无任何读数数据），先跑一次含吞吐报告的运行。"],
            )
        if any(p.output_len > 0 for p in self.points):
            return self._query_2d(target_rpm, input_len, output_len, notes)
        return self._query_1d(target_rpm, input_len, output_len, notes)

    def _query_1d(
        self, target_rpm: float, input_len: int, output_len: int, notes: List[str]
    ) -> CapacityAnswer:
        """一维面查询（未记录输出长度的旧版面；按输入长度插值）。"""
        pts = sorted(self.points, key=lambda p: p.input_len)
        lo, hi = pts[0].input_len, pts[-1].input_len
        eff = min(max(int(input_len), lo), hi)
        extrapolated = int(input_len) < lo or int(input_len) > hi
        if extrapolated:
            notes.append(
                f"请求长度 {input_len} 超出测量范围 [{lo}, {hi}]，已 clamp 到 {eff}；"
                "外插结果只作参考，不作为验收依据。"
            )
        if output_len > 0:
            notes.append(
                "该容量面未记录输出长度维度（旧版一维面），--output-len 被忽略；"
                "重新以多输出档运行可升级为二维面（D22）。"
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
            output_len=int(output_len), effective_output_len=0,
            origin=nearest.origin,
        )

    def _query_2d(
        self, target_rpm: float, input_len: int, output_len: int, notes: List[str]
    ) -> CapacityAnswer:
        """二维面查询：网格内双线性插值；轴退化退一维；角点缺失回退最近点。"""
        pts = sorted(self.points, key=lambda p: (p.input_len, p.output_len))
        xs = sorted({float(p.input_len) for p in pts})
        ys = sorted({float(p.output_len) for p in pts})
        by_cell = {(int(p.input_len), int(p.output_len)): p for p in pts}

        x_req, y_req = int(input_len), int(output_len)
        if y_req <= 0:
            y_req = int(ys[0])
            notes.append(
                f"未指定 --output-len：本容量面含输出长度档 "
                f"{[int(v) for v in ys]}，已按最小档 {y_req} 作答；"
                "不同输出档容量不可比（D22），请显式指定 --output-len 获取目标档读数。"
            )

        lo_x, hi_x = xs[0], xs[-1]
        lo_y, hi_y = ys[0], ys[-1]
        eff_x = min(max(float(x_req), lo_x), hi_x)
        eff_y = min(max(float(y_req), lo_y), hi_y)
        extrapolated = float(x_req) < lo_x or float(x_req) > hi_x \
            or float(y_req) < lo_y or float(y_req) > hi_y
        if extrapolated:
            notes.append(
                f"查询坐标 ({x_req}, {y_req}) 超出测量范围 "
                f"x[{lo_x:g}, {hi_x:g}] × y[{lo_y:g}, {hi_y:g}]，"
                f"已 clamp 到 ({eff_x:g}, {eff_y:g})；外插结果只作参考，不作为验收依据。"
            )

        def _bracket(coords: List[float], v: float) -> Tuple[float, float]:
            if len(coords) == 1:
                return coords[0], coords[0]
            for i in range(len(coords) - 1):
                if coords[i] <= v <= coords[i + 1]:
                    return coords[i], coords[i + 1]
            return coords[-2], coords[-1]

        x0, x1 = _bracket(xs, eff_x)
        y0, y1 = _bracket(ys, eff_y)
        tx = 0.0 if x1 == x0 else (eff_x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (eff_y - y0) / (y1 - y0)

        # 只要求权重非零的角点齐全：查询点恰在网格线/节点上时，另一侧
        # 角点缺失不影响读数，不必为之放弃插值（但严格在面内时缺角必须
        # 回退最近点，不做静默的部分插值）。
        xs_needed = (x0,) if tx <= 0 else ((x1,) if tx >= 1 else (x0, x1))
        ys_needed = (y0,) if ty <= 0 else ((y1,) if ty >= 1 else (y0, y1))
        needed = [by_cell.get((int(a), int(b))) for a in xs_needed for b in ys_needed]
        if any(c is None for c in needed):
            near = self._nearest_point(pts, x_req, y_req, xs, ys)
            notes.append(
                f"查询点 ({eff_x:g}, {eff_y:g}) 所需网格角点未全部实测，"
                f"已回退最近实测点 ({near.input_len}, {near.output_len})——"
                "该读数为最近点而非插值；按角点铺床（四角运行）后可改善。"
            )
            return self._answer_from_point(
                target_rpm, x_req, y_req, near, near.input_len, near.output_len,
                extrapolated, notes,
            )

        # 角点允许缺失（.get）：零权重侧的角点不参与插值（_pair 触边取端点），
        # 严格在面内所需的角点已在上面 needed 检查中保证齐全
        p00 = by_cell.get((int(x0), int(y0)))
        p10 = by_cell.get((int(x1), int(y0)))
        p01 = by_cell.get((int(x0), int(y1)))
        p11 = by_cell.get((int(x1), int(y1)))

        def _pair(va: Optional[float], vb: Optional[float], t: float) -> Optional[float]:
            # t 触边时直接取端点值；严格在中间时任一端缺失即整体缺失
            #（不静默填 0，与一维面口径一致）
            if t <= 0:
                return va
            if t >= 1:
                return vb
            if va is None or vb is None:
                return None
            return va + t * (vb - va)

        def _bilin(get) -> Optional[float]:
            # get 必须对 None 角点安全（返回 None），触边的零权重角点不参与
            top = _pair(get(p00), get(p10), tx)     # y0 行沿 x
            bot = _pair(get(p01), get(p11), tx)     # y1 行沿 x
            return _pair(top, bot, ty)              # 沿 y 合成

        mu_rpm = _bilin(lambda p: p.mu_rpm if p is not None else None)
        mu_tps = _bilin(lambda p: p.mu_tps if p is not None else None)
        input_tps = _bilin(lambda p: p.input_tps if p is not None else None)
        output_tps = _bilin(lambda p: p.output_tps if p is not None else None)
        req_conc_f = _bilin(lambda p: float(p.req_conc) if p is not None else None)
        req_conc = max(1, round(req_conc_f if req_conc_f is not None else 1.0))
        # 家族/来源是离散标签，不插值：取单位方块内归一化距离最近的角点
        #（最近角点必在 needed 集合内，不会取到缺失的 None）
        cands = [c for c in (
            (0.0, 0.0, p00), (1.0, 0.0, p10), (0.0, 1.0, p01), (1.0, 1.0, p11),
        ) if c[2] is not None]
        near = min(cands, key=lambda c: (tx - c[0]) ** 2 + (ty - c[1]) ** 2)[2]

        if mu_tps is None:
            notes.append("查询坐标附近缺少完整 token 数据（usage 未上报），仅 RPM 口径可信。")
        margin = mu_rpm - target_rpm
        achievable = mu_rpm >= target_rpm
        if not achievable:
            notes.append(
                f"目标 {target_rpm:g} RPM 超出坐标 ({eff_x:g}, {eff_y:g}) 处的实测容量 "
                f"{mu_rpm:.1f} RPM：需扩容、缩短输入/输出或重谈窗口承诺。"
            )
        return CapacityAnswer(
            target_rpm=target_rpm, input_len=int(input_len),
            effective_len=int(round(eff_x)),
            extrapolated=extrapolated, mu_rpm=mu_rpm, mu_tps=mu_tps,
            input_tps=input_tps, output_tps=output_tps, req_conc=req_conc,
            margin_rpm=margin, achievable=achievable,
            a1_family=near.a1_family, notes=notes,
            output_len=int(y_req), effective_output_len=int(round(eff_y)),
            origin=near.origin,
        )

    @staticmethod
    def _nearest_point(
        pts: Sequence[ThroughputPoint], x: int, y: int, xs: List[float], ys: List[float]
    ) -> ThroughputPoint:
        """归一化欧氏距离最近的实测点（轴跨度归一，避免长轴吞掉短轴差异）。"""
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)

        def dist(p: ThroughputPoint) -> Tuple[float, float, int, int]:
            dx = (float(p.input_len) - float(x)) / span_x if span_x > 0 else 0.0
            dy = (float(p.output_len) - float(y)) / span_y if span_y > 0 else 0.0
            return (dx * dx + dy * dy, abs(dx) + abs(dy), p.input_len, p.output_len)

        return min(pts, key=dist)

    @staticmethod
    def _answer_from_point(
        target_rpm: float, x_req: int, y_req: int, point: ThroughputPoint,
        eff_x: int, eff_y: int, extrapolated: bool, notes: List[str],
    ) -> CapacityAnswer:
        if point.mu_tps is None:
            notes.append("查询坐标附近缺少完整 token 数据（usage 未上报），仅 RPM 口径可信。")
        margin = point.mu_rpm - target_rpm
        achievable = point.mu_rpm >= target_rpm
        if not achievable:
            notes.append(
                f"目标 {target_rpm:g} RPM 超出最近实测点 ({eff_x}, {eff_y}) 处的容量 "
                f"{point.mu_rpm:.1f} RPM：需扩容、缩短输入/输出或重谈窗口承诺。"
            )
        return CapacityAnswer(
            target_rpm=target_rpm, input_len=int(x_req), effective_len=int(eff_x),
            extrapolated=extrapolated, mu_rpm=point.mu_rpm, mu_tps=point.mu_tps,
            input_tps=point.input_tps, output_tps=point.output_tps,
            req_conc=point.req_conc, margin_rpm=margin, achievable=achievable,
            a1_family=point.a1_family, notes=notes,
            output_len=int(y_req), effective_output_len=int(eff_y),
            origin=point.origin,
        )

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """写 .npz（npz.py 纯机制；字符串编码为整数、缺失值存 NaN）。"""
        pts = sorted(self.points, key=lambda p: (p.input_len, p.output_len))
        arrays: Dict[str, Any] = {
            "lengths": [int(p.input_len) for p in pts],
            "output_lens": [int(p.output_len) for p in pts],
            "mu_rpm": [float(p.mu_rpm) for p in pts],
            "mu_tps": [_or_nan(p.mu_tps) for p in pts],
            "input_tps": [_or_nan(p.input_tps) for p in pts],
            "output_tps": [_or_nan(p.output_tps) for p in pts],
            "req_conc": [int(p.req_conc) for p in pts],
            "n_samples": [int(p.n_samples) for p in pts],
            "a1_family": [_FAMILY_TO_CODE.get(p.a1_family, _FAMILY_TO_CODE["other"]) for p in pts],
            "origins": [_ORIGIN_TO_CODE.get(p.origin, 0) for p in pts],
        }
        meta = dict(self.meta)
        meta["kind"] = SURFACE_KIND
        meta["a1_family_labels"] = list(A1_FAMILY_LABELS)
        meta["origin_labels"] = list(ORIGIN_LABELS)
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
        # 旧版一维面守卫：output_lens/origins 缺省时按 0/"knee" 读回，
        # 长度不一致则视为损坏（不静默截断）
        extras: Dict[str, Optional[List[Any]]] = {}
        for key in _OPTIONAL_ARRAYS:
            arr = data.get(key)
            if arr is not None and len(arr) != n:
                raise ThroughputError(
                    f"容量面文件数组长度不一致: {key} 为 {len(arr)}，lengths 为 {n}"
                )
            extras[key] = arr
        output_lens = extras["output_lens"]
        origins = extras["origins"]
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
                output_len=int(output_lens[i]) if output_lens is not None else 0,
                origin=ORIGIN_LABELS[int(origins[i])]
                if origins is not None and 0 <= int(origins[i]) < len(ORIGIN_LABELS)
                else "knee",
            )
            for i in range(n)
        ]
        return cls(points=points, meta=meta)


def _interp_optional(xs: Sequence[float], ys: Sequence[Optional[float]], x: float) -> Optional[float]:
    """可缺值插值：任一侧缺失即整体缺失（不静默填 0）。"""
    if any(v is None for v in ys):
        return None
    return _interp(xs, ys, x)


def _dedup_by_cell(points: Sequence[ThroughputPoint]) -> List[ThroughputPoint]:
    """按 (input_len, output_len) 单元格去重（同格后者覆盖），按坐标升序。"""
    by_cell: Dict[Tuple[int, int], ThroughputPoint] = {}
    for p in points:
        by_cell[(int(p.input_len), int(p.output_len))] = p
    return [by_cell[k] for k in sorted(by_cell)]


def surface_path_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()
