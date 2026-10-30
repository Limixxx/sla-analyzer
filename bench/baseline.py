"""基线剖析与判定门限解析。

两个核心概念：

**长度分桶（bucket）**
    TTFT 与 prompt 长度强相关（prefill 计算量近似线性），TPOT 与已生成上下文
    长度相关。全局单一时延门限会让长 prompt 请求天然落入 A3，把"长度差异"
    混进"排队劣化"里。分桶后每个桶有独立门限，A3 归因才能回答"是所有长度
    都劣化，还是只有长 prompt 劣化"。

**基线相对门限（baseline_relative）**
    门限 = 基线 p99 × 系数。基线测于 Phase 0 的低压阶段，与压力解耦，
    因此同一把尺子可以贯穿整个 ramp，跨级可比；换引擎/模型只需重测基线，
    整条曲线自动平移。系数是显式配置，结论可复现、可审计。

明确**不**支持的模式：用本次运行中观测到的分位数（尤其是高压段）定标。
那会让门限随压力同步抬升，SLO 恒约等于目标值——指标永真、无法验收，
且高压段分位数本身已被排队污染（用病态定义正常）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .config import Config, SloConfig
from .stats import percentile


# ---------------------------------------------------------------------- #
# 长度分桶
# ---------------------------------------------------------------------- #

def bucket_count(buckets: Sequence[int]) -> int:
    """桶数量 = 上界个数 + 1（最后一个桶无上界）。"""
    return len(buckets) + 1


def bucket_of(tokens: Optional[int], buckets: Sequence[int]) -> int:
    """把 prompt token 数映射到桶下标；未知长度归入 0 号桶。"""
    if tokens is None:
        return 0
    for idx, upper in enumerate(buckets):
        if tokens <= upper:
            return idx
    return len(buckets)


def bucket_label(idx: int, buckets: Sequence[int]) -> str:
    """桶的可读标签，如 <=1024 / 1025-4096 / >4096。"""
    if idx <= 0:
        return f"≤{buckets[0]}" if buckets else "全部"
    if idx >= len(buckets):
        return f">{buckets[-1]}"
    return f"{buckets[idx - 1] + 1}-{buckets[idx]}"


def row_tokens(row: Dict[str, Any]) -> Optional[int]:
    """请求的实际 prompt 长度；缺失时退回计划长度。"""
    for key in ("prompt_tokens", "prompt_tokens_planned"):
        value = row.get(key)
        if value:
            return int(value)
    return None


# ---------------------------------------------------------------------- #
# 基线
# ---------------------------------------------------------------------- #

@dataclass
class BucketBaseline:
    """单个长度桶在低压基线下的时延分布。"""

    bucket: int
    n: int = 0
    n_ok: int = 0
    a1: int = 0
    ttft_p50_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    tpot_p50_ms: Optional[float] = None
    tpot_p95_ms: Optional[float] = None
    tpot_p99_ms: Optional[float] = None


@dataclass
class BaselineResult:
    """Phase 0 基线剖析的产出。"""

    concurrency: int = 0
    duration_s: float = 0.0
    # 实际运行时长（含自适应延长）。与 duration_s 不一致时说明基线阶段
    # 为了补齐各长度桶的样本量被延长了——报告与日志都必须如实展示，
    # 否则"配置 30s"与"实际跑了 3 分钟"会对不上。
    duration_actual_s: Optional[float] = None
    buckets: List[BucketBaseline] = field(default_factory=list)
    total_n: int = 0
    total_a1: int = 0
    thin_buckets: List[int] = field(default_factory=list)   # 样本不足的桶
    notes: List[str] = field(default_factory=list)

    def by_bucket(self) -> Dict[int, BucketBaseline]:
        return {b.bucket: b for b in self.buckets}


def compute_baseline(
    rows: Sequence[Dict[str, Any]],
    buckets: Sequence[int],
    min_samples: int,
    min_completion_tokens: int = 2,
) -> BaselineResult:
    """由基线阶段的请求明细计算各桶的时延分位数。

    基线阶段只统计分位数与失败数，**不对 A2/A3 做判定**——基线本身就是
    尺子的来源，再拿它判达标会形成循环。
    """
    result = BaselineResult(buckets=[BucketBaseline(bucket=i) for i in range(bucket_count(buckets))])
    per_bucket_ttft: Dict[int, List[float]] = {i: [] for i in range(bucket_count(buckets))}
    per_bucket_tpot: Dict[int, List[float]] = {i: [] for i in range(bucket_count(buckets))}
    index = result.by_bucket()

    for row in rows:
        idx = bucket_of(row_tokens(row), buckets)
        entry = index[idx]
        entry.n += 1
        if row.get("status") == "ok":
            entry.n_ok += 1
            if row.get("ttft_ms") is not None:
                per_bucket_ttft[idx].append(float(row["ttft_ms"]))
            completion = row.get("completion_tokens") or 0
            if row.get("tpot_ms") is not None and completion >= min_completion_tokens:
                per_bucket_tpot[idx].append(float(row["tpot_ms"]))
        else:
            entry.a1 += 1
            result.total_a1 += 1
        result.total_n += 1

    for entry in result.buckets:
        entry.ttft_p50_ms = percentile(per_bucket_ttft[entry.bucket], 0.50)
        entry.ttft_p95_ms = percentile(per_bucket_ttft[entry.bucket], 0.95)
        entry.ttft_p99_ms = percentile(per_bucket_ttft[entry.bucket], 0.99)
        entry.tpot_p50_ms = percentile(per_bucket_tpot[entry.bucket], 0.50)
        entry.tpot_p95_ms = percentile(per_bucket_tpot[entry.bucket], 0.95)
        entry.tpot_p99_ms = percentile(per_bucket_tpot[entry.bucket], 0.99)
        if entry.n_ok < min_samples:
            result.thin_buckets.append(entry.bucket)

    if result.total_a1 > 0:
        result.notes.append(
            f"基线阶段出现 {result.total_a1} 个失败请求，基线被污染："
            "请降低 load.baseline_concurrency 或排查服务可用性后重测。"
        )
    if result.thin_buckets:
        labels = "、".join(bucket_label(i, buckets) for i in result.thin_buckets)
        result.notes.append(
            f"桶 {labels} 的基线样本不足 {min_samples} 个，p99 置信度低"
            "（baseline_relative 模式下这些桶的门限将回退为 absolute 配置值）。"
        )
    return result


# ---------------------------------------------------------------------- #
# 门限解析
# ---------------------------------------------------------------------- #

@dataclass
class Thresholds:
    """生效的判定门限（按桶）。"""

    mode: str
    ttft_ms: Dict[int, float] = field(default_factory=dict)
    tpot_ms: Dict[int, float] = field(default_factory=dict)
    fallback_buckets: List[int] = field(default_factory=list)   # 回退到 absolute 的桶
    notes: List[str] = field(default_factory=list)
    baseline: Optional[BaselineResult] = None

    def ttft_limit(self, bucket: int) -> Optional[float]:
        return self.ttft_ms.get(bucket)

    def tpot_limit(self, bucket: int) -> Optional[float]:
        return self.tpot_ms.get(bucket)


def resolve_thresholds(
    cfg: Config,
    baseline: Optional[BaselineResult],
    min_samples: Optional[int] = None,
) -> Thresholds:
    """按 target_mode 解析每个桶生效的 TTFT/TPOT 门限。

    baseline_relative 模式下，样本不足或缺失的桶回退到 absolute 配置值，
    并在 notes 中说明——绝不静默使用一个不可信的分位数。

    min_samples 默认值按模式区分：baseline_relative 用
    analysis.baseline_min_samples（按桶判定，门槛天然更严），absolute 根本
    不看样本量。所有调用点（runner 与 --report-only）共用这一默认值，
    保证压测时与重建报告时解析出的门限一致。
    """
    slo: SloConfig = cfg.slo
    buckets = cfg.requests.length_buckets
    n_buckets = bucket_count(buckets)
    if min_samples is None:
        min_samples = (
            cfg.analysis.baseline_min_samples
            if slo.target_mode == "baseline_relative"
            else cfg.analysis.min_samples_per_stage
        )

    absolute_ttft = slo.ttft_target_s * 1000.0
    absolute_tpot = slo.tpot_target_ms
    thresholds = Thresholds(mode=slo.target_mode)

    if slo.target_mode == "absolute":
        for i in range(n_buckets):
            thresholds.ttft_ms[i] = absolute_ttft
            thresholds.tpot_ms[i] = absolute_tpot
        return thresholds

    # ---- baseline_relative ----
    if baseline is None or not baseline.buckets:
        thresholds.notes.append(
            "target_mode=baseline_relative 但未获得基线数据，已整体回退为 absolute 门限。"
        )
        for i in range(n_buckets):
            thresholds.ttft_ms[i] = absolute_ttft
            thresholds.tpot_ms[i] = absolute_tpot
            thresholds.fallback_buckets.append(i)
        return thresholds

    index = baseline.by_bucket()
    for i in range(n_buckets):
        entry = index.get(i)
        usable = (
            entry is not None
            and entry.n_ok >= min_samples
            and entry.ttft_p99_ms is not None
            and entry.tpot_p99_ms is not None
        )
        if usable:
            thresholds.ttft_ms[i] = float(entry.ttft_p99_ms) * slo.ttft_factor
            thresholds.tpot_ms[i] = float(entry.tpot_p99_ms) * slo.tpot_factor
        else:
            thresholds.ttft_ms[i] = absolute_ttft
            thresholds.tpot_ms[i] = absolute_tpot
            thresholds.fallback_buckets.append(i)

    thresholds.baseline = baseline
    if thresholds.fallback_buckets:
        labels = "、".join(bucket_label(i, buckets) for i in thresholds.fallback_buckets)
        thresholds.notes.append(
            f"桶 {labels} 基线样本不足 {min_samples} 个或分位数缺失，"
            "门限回退为 absolute 配置值；如需这些桶也走基线口径，请增大 "
            "load.baseline_duration_s / load.baseline_concurrency 后重测。"
        )
    return thresholds


def describe_thresholds(thresholds: Thresholds, buckets: Sequence[int]) -> List[str]:
    """人类可读的生效门限说明（用于日志；报告层另有逐桶表格）。"""
    lines: List[str] = []
    if thresholds.mode == "absolute":
        lines.append(
            f"absolute 模式：所有桶统一 TTFT ≤ {thresholds.ttft_ms.get(0, 0):.0f}ms，"
            f"TPOT ≤ {thresholds.tpot_ms.get(0, 0):.0f}ms"
        )
    else:
        base = thresholds.baseline
        origin = (
            f"baseline_relative 模式（基线：并发 {base.concurrency}，{base.duration_s:.0f}s）"
            if base is not None else "baseline_relative 模式（基线缺失，已回退）"
        )
        parts = []
        for i in sorted(thresholds.ttft_ms):
            tag = "*" if i in thresholds.fallback_buckets else ""
            parts.append(
                f"{bucket_label(i, buckets)}{tag} TTFT≤{thresholds.ttft_ms[i]:.0f}ms / "
                f"TPOT≤{thresholds.tpot_ms[i]:.0f}ms"
            )
        lines.append(origin + "：" + "；".join(parts) + "（* = 该桶回退为 absolute 门限）")
    lines.extend(thresholds.notes)
    return lines
