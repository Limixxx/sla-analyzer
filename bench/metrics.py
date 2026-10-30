"""指标口径：A1/A2/A3 分类、分位数、分级统计聚合。

分类规则（与服务质量定义一一对应）：
- A1 失败请求            ：HTTP 错误 / 连接失败 / 超时 / 流错误 / 空输出
- A2 成功且满足 SLO 的请求：TTFT、TPOT 双双不超出门限
- A3 成功但不满足 SLO 的请求：其余成功请求

A1 内部按归因家族再分两层（不影响分类本身，只影响"过载长什么样"的解读，
见 docs/05-watermark.md 无限队列一节）：
- 超时族 timeout / connect_timeout：请求已被服务接受，但在客户端预算内
  未完成。Router 配无限队列（无 429/529）时，过载的唯一投影就是这一族；
- 拒绝族 rejected / http_4xx / http_5xx / stream_error：服务端明确回复错误。

门限可以按 prompt 长度分桶（TTFT 与长度强相关，全局单门限会让长 prompt
请求天然落入 A3，掩盖真正的排队信号）；门限来源见 bench/baseline.py。

派生指标：
- SLA（可用性） = (A2 + A3) / 总数
- SLO（质量）   = A2 / 总数

水位判定（服务端水位的黑盒可观测投影，见 docs/05-watermark.md）：
每级按 A1/A3 占比与 TTFT p99 判定 NORMAL / BUSY（软水位）/ OVERLOAD（硬水位），
跨级汇总软/硬/SLA 三个水位边界。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .baseline import Thresholds, bucket_of, row_tokens
from .client import (
    KIND_CONNECT_TIMEOUT,
    KIND_HTTP_4XX,
    KIND_HTTP_5XX,
    KIND_REJECTED,
    KIND_STREAM_ERROR,
    KIND_TIMEOUT,
)
from .config import AnalysisConfig, SloConfig, WatermarkConfig
from .stats import mean, percentile

# A1 的归因家族（error_kind 的两类聚合，回答"过载长什么样"）：
# - 超时族：服务没说"不接"，但请求在客户端预算内没完成。Router 无限队列
#   （无 429/529）时过载只投影为这一族——分类上仍是 A1，仍拉低 SLA。
# - 拒绝族：服务端明确回复错误（过载拒绝 / 4xx / 5xx / 流内 error）。
# 其余 kind（conn_error / incomplete / empty_completion / protocol_error）
# 归"其他"：可能是排队耗尽后的连接被切，也可能是服务崩溃，不强行归因。
A1_TIMEOUT_KINDS = frozenset({KIND_TIMEOUT, KIND_CONNECT_TIMEOUT})
A1_REJECT_KINDS = frozenset({KIND_REJECTED, KIND_HTTP_4XX, KIND_HTTP_5XX, KIND_STREAM_ERROR})


def split_fail_kinds(fail_kinds: Dict[str, int]) -> Tuple[int, int, int]:
    """把 A1 的 error_kind 直方图拆成 (超时族, 拒绝族, 其他)。

    fail_kinds 只含 A1 的 kind（compute_stage_stats 在 A1 分支内填充），
    因此三者之和恒等于 A1 总数。实时跑与 --report-only（fail_kinds_json
    重载）走同一函数，归因结论跨路径一致。
    """
    timeout_n = sum(c for k, c in fail_kinds.items() if k in A1_TIMEOUT_KINDS)
    reject_n = sum(c for k, c in fail_kinds.items() if k in A1_REJECT_KINDS)
    other_n = sum(fail_kinds.values()) - timeout_n - reject_n
    return timeout_n, reject_n, other_n


def classify(
    result: Dict[str, Any],
    slo: SloConfig,
    thresholds: Optional[Thresholds] = None,
    buckets: Optional[Sequence[int]] = None,
) -> str:
    """对单条请求记录打 A1/A2/A3 标签。

    thresholds 为空时退化为全局绝对门限（保持旧行为）。
    """
    if result.get("status") != "ok":
        return "A1"
    ttft = result.get("ttft_ms")
    tpot = result.get("tpot_ms")
    completion = result.get("completion_tokens") or 0

    if thresholds is not None and buckets is not None:
        idx = bucket_of(row_tokens(result), buckets)
        ttft_limit = thresholds.ttft_limit(idx)
        tpot_limit = thresholds.tpot_limit(idx)
        if ttft_limit is None:
            ttft_limit = slo.ttft_target_s * 1000.0
        if tpot_limit is None:
            tpot_limit = slo.tpot_target_ms
    else:
        ttft_limit = slo.ttft_target_s * 1000.0
        tpot_limit = slo.tpot_target_ms

    ttft_ok = ttft is not None and ttft <= ttft_limit
    # 输出 token 过少时 TPOT 不可靠，按配置决定是否参与判定
    if completion < slo.min_completion_tokens:
        tpot_ok = True
    else:
        tpot_ok = tpot is None or tpot <= tpot_limit

    return "A2" if (ttft_ok and tpot_ok) else "A3"


def classify_watermark(stats: "StageStats", wm: WatermarkConfig) -> Tuple[str, List[str]]:
    """按黑盒可观测信号判定本级水位状态（watermark.md 三层语义的外部对偶）。

    OR 逻辑、优先级 OVERLOAD > BUSY > NORMAL：
    - OVERLOAD：A1 占比 ≥ hard_reject_ratio（服务在拒绝，或无限队列下
      排队耗尽投影为客户端超时——两种都算"接不住"）或
      A3 占比 ≥ hard_slo_violation_ratio（质量崩塌）；
    - BUSY：A3 占比 ≥ soft_slo_violation_ratio，或（启用时）TTFT p99 ≥
      soft_ttft_p99_ms——排队信号：不拒绝，但时延/质量已劣化；
    - NORMAL：各项可观测指标均在软门限之内；
    - unknown：无完成样本，无法判定（不参与边界发现）。

    判定只用本级已有的聚合量，不改请求路径；--report-only 从旧库重建时
    同样可判（幂等）。资源级水位（KV/slot/GPU）服务端不可见，不在此列。

    A1 触发硬水位时按归因家族（split_fail_kinds）给出解读：超时族主导
    ⇒ 排队型过载（Router 疑似无限队列）；拒绝族主导 ⇒ 拒绝型过载。
    """
    if stats.n_total == 0:
        return "unknown", []

    a1_ratio = stats.a1 / stats.n_total
    a3_ratio = stats.a3 / stats.n_total

    hard: List[str] = []
    if a1_ratio >= wm.hard_reject_ratio:
        timeout_n, reject_n, other_n = stats.a1_families()
        hard.append(
            f"失败率 {a1_ratio:.2%} ≥ {wm.hard_reject_ratio:.2%}"
            f"（A1={stats.a1}/{stats.n_total}：客户端超时 {timeout_n}、"
            f"服务端拒绝 {reject_n}、其他 {other_n}）"
        )
        if timeout_n > reject_n:
            hard.append(
                "失败以客户端超时为主：服务不拒绝但无法在预算内完成——"
                "Router 疑似无限队列/排队耗尽，过载形态为排队型而非拒绝型"
            )
        elif reject_n > timeout_n:
            hard.append(
                "失败以服务端拒绝为主：过载形态为拒绝型（429/5xx 明确回绝）"
            )
    if a3_ratio >= wm.hard_slo_violation_ratio:
        hard.append(
            f"质量违约率 {a3_ratio:.2%} ≥ {wm.hard_slo_violation_ratio:.2%}"
            f"（A3={stats.a3}/{stats.n_total}）"
        )
    if hard:
        return "OVERLOAD", hard

    soft: List[str] = []
    if a3_ratio >= wm.soft_slo_violation_ratio:
        soft.append(
            f"质量违约率 {a3_ratio:.2%} ≥ {wm.soft_slo_violation_ratio:.2%}"
            f"（A3={stats.a3}/{stats.n_total}）"
        )
    if (
        wm.soft_ttft_p99_ms > 0
        and stats.ttft_p99_ms is not None
        and stats.ttft_p99_ms >= wm.soft_ttft_p99_ms
    ):
        soft.append(f"TTFT p99 {stats.ttft_p99_ms:.0f}ms ≥ {wm.soft_ttft_p99_ms:.0f}ms")
    if soft:
        return "BUSY", soft
    return "NORMAL", ["各项可观测指标均在软门限之内"]


@dataclass
class StageStats:
    stage_id: int
    concurrency: int
    duration_s: float
    warmup_s: float
    # 实际运行时长（含自适应延长）；与 duration_s（配置值）分开，
    # 保证“配置 30s、实际跑了 3 分钟”在日志与报告中都可审计。
    actual_duration_s: Optional[float] = None

    n_total: int = 0
    n_warmup: int = 0
    n_ok: int = 0
    n_failed: int = 0
    n_aborted: int = 0               # 本级结束时仍在途、被 drain 放弃的请求（不参与 SLA/SLO）
    a1: int = 0
    a2: int = 0
    a3: int = 0

    slo: float = 0.0
    sla: float = 0.0

    fail_kinds: Dict[str, int] = field(default_factory=dict)
    a3_ttft_only: int = 0
    a3_tpot_only: int = 0
    a3_both: int = 0

    ttft_p50_ms: Optional[float] = None
    ttft_p90_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    tpot_p50_ms: Optional[float] = None
    tpot_p90_ms: Optional[float] = None
    tpot_p95_ms: Optional[float] = None
    tpot_p99_ms: Optional[float] = None
    e2e_p50_ms: Optional[float] = None
    e2e_p99_ms: Optional[float] = None
    connect_p50_ms: Optional[float] = None   # 建连+写出+等响应头；TTFT/E2E 的客户端视角组成部分
    connect_p99_ms: Optional[float] = None

    rps: Optional[float] = None
    output_tps: Optional[float] = None
    prompt_tokens_avg: Optional[float] = None
    completion_tokens_avg: Optional[float] = None

    ttft_target_ms: float = 0.0
    tpot_target_ms: float = 0.0
    sla_target: float = 0.0
    slo_target: float = 0.0

    # 水位状态（服务端水位的黑盒可观测投影）：NORMAL / BUSY / OVERLOAD / unknown
    watermark_state: str = "unknown"
    # 触发原因（人类可读，随 watermark_reasons_json 落库）
    watermark_reasons: List[str] = field(default_factory=list)

    # 按 prompt 长度桶的拆分（报告用；不入 stages 表，另存 stage_buckets）
    bucket_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    @property
    def sla_met(self) -> bool:
        return self.n_total > 0 and self.sla >= self.sla_target

    @property
    def slo_met(self) -> bool:
        return self.n_total > 0 and self.slo >= self.slo_target

    def a1_families(self) -> Tuple[int, int, int]:
        """A1 按归因家族拆分：(超时族, 拒绝族, 其他)。

        由 fail_kinds 现场推导（live 与 report-only 一致），不落库——
        fail_kinds_json 已是充分统计量，避免为派生量加 schema 列。
        """
        return split_fail_kinds(self.fail_kinds)

    def as_row(self) -> Dict[str, Any]:
        # 只导出标量字段：字典字段（fail_kinds / bucket_stats）另走专用表
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
            if isinstance(getattr(self, key), (int, float, str)) or getattr(self, key) is None
        }


def compute_stage_stats(
    stage_id: int,
    concurrency: int,
    rows: Iterable[Dict[str, Any]],
    warmup_rows: int,
    slo: SloConfig,
    analysis: AnalysisConfig,
    duration_s: float,
    warmup_s: float,
    thresholds: Optional[Thresholds] = None,
    buckets: Optional[Sequence[int]] = None,
    watermark: Optional[WatermarkConfig] = None,
) -> StageStats:
    """由请求明细聚合一级压力位的统计量。"""
    stats = StageStats(
        stage_id=stage_id,
        concurrency=concurrency,
        duration_s=duration_s,
        warmup_s=warmup_s,
        n_warmup=warmup_rows,
        ttft_target_ms=slo.ttft_target_s * 1000.0,
        tpot_target_ms=slo.tpot_target_ms,
        sla_target=analysis.sla_target,
        slo_target=analysis.slo_target,
    )

    rows = list(rows)
    # aborted（drain 阶段被放弃的在途请求）单列：无法判定成败，不进入任何比率与分位数，
    # 否则"最慢的请求被静默丢弃"会造成存活者偏差，低估时延与 A3。
    aborted_rows = [r for r in rows if r.get("status") == "aborted"]
    stats.n_aborted = len(aborted_rows)
    rows = [r for r in rows if r.get("status") != "aborted"]
    stats.n_total = len(rows)
    if not rows:
        return stats

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    stats.n_ok = len(ok_rows)
    stats.n_failed = stats.n_total - stats.n_ok

    for r in rows:
        klass = r.get("latency_class") or classify(r, slo, thresholds, buckets)
        if klass == "A1":
            stats.a1 += 1
            kind = r.get("error_kind") or "unknown"
            stats.fail_kinds[kind] = stats.fail_kinds.get(kind, 0) + 1
        elif klass == "A2":
            stats.a2 += 1
        else:
            stats.a3 += 1

    stats.slo = stats.a2 / stats.n_total
    stats.sla = (stats.a2 + stats.a3) / stats.n_total

    # ---- A3 归因（按桶门限判定超标侧）----
    per_bucket_ttft: Dict[int, List[float]] = {}
    per_bucket_tpot: Dict[int, List[float]] = {}
    for r in rows:
        idx = bucket_of(row_tokens(r), buckets) if buckets else 0
        entry = stats.bucket_stats.setdefault(
            idx, {"n": 0, "a1": 0, "a2": 0, "a3": 0, "a3_ttft_only": 0, "a3_tpot_only": 0, "a3_both": 0}
        )
        entry["n"] += 1
        klass = r.get("latency_class") or classify(r, slo, thresholds, buckets)
        if klass == "A1":
            entry["a1"] += 1
            continue
        if klass == "A2":
            entry["a2"] += 1
        else:
            entry["a3"] += 1
            ttft_limit = (
                thresholds.ttft_limit(idx) if thresholds is not None else None
            ) or slo.ttft_target_s * 1000.0
            tpot_limit = (
                thresholds.tpot_limit(idx) if thresholds is not None else None
            ) or slo.tpot_target_ms
            ttft_bad = r.get("ttft_ms") is not None and r["ttft_ms"] > ttft_limit
            completion = r.get("completion_tokens") or 0
            if completion < slo.min_completion_tokens:
                tpot_bad = False
            else:
                tpot_bad = r.get("tpot_ms") is not None and r["tpot_ms"] > tpot_limit
            if ttft_bad and tpot_bad:
                entry["a3_both"] += 1
                stats.a3_both += 1
            elif ttft_bad:
                entry["a3_ttft_only"] += 1
                stats.a3_ttft_only += 1
            elif tpot_bad:
                entry["a3_tpot_only"] += 1
                stats.a3_tpot_only += 1
        if r.get("ttft_ms") is not None:
            per_bucket_ttft.setdefault(idx, []).append(float(r["ttft_ms"]))
        if r.get("tpot_ms") is not None:
            per_bucket_tpot.setdefault(idx, []).append(float(r["tpot_ms"]))

    for idx, entry in stats.bucket_stats.items():
        entry["ttft_p99_ms"] = percentile(per_bucket_ttft.get(idx, []), 0.99)
        entry["tpot_p99_ms"] = percentile(per_bucket_tpot.get(idx, []), 0.99)
        if thresholds is not None:
            entry["ttft_limit_ms"] = thresholds.ttft_limit(idx)
            entry["tpot_limit_ms"] = thresholds.tpot_limit(idx)

    ttfts = [r["ttft_ms"] for r in ok_rows if r.get("ttft_ms") is not None]
    tpots = [r["tpot_ms"] for r in ok_rows if r.get("tpot_ms") is not None]
    e2es = [r["e2e_ms"] for r in ok_rows if r.get("e2e_ms") is not None]
    connects = [r["connect_ms"] for r in ok_rows if r.get("connect_ms") is not None]

    stats.ttft_p50_ms = percentile(ttfts, 0.50)
    stats.ttft_p90_ms = percentile(ttfts, 0.90)
    stats.ttft_p95_ms = percentile(ttfts, 0.95)
    stats.ttft_p99_ms = percentile(ttfts, 0.99)
    stats.tpot_p50_ms = percentile(tpots, 0.50)
    stats.tpot_p90_ms = percentile(tpots, 0.90)
    stats.tpot_p95_ms = percentile(tpots, 0.95)
    stats.tpot_p99_ms = percentile(tpots, 0.99)
    stats.e2e_p50_ms = percentile(e2es, 0.50)
    stats.e2e_p99_ms = percentile(e2es, 0.99)
    stats.connect_p50_ms = percentile(connects, 0.50)
    stats.connect_p99_ms = percentile(connects, 0.99)

    # 吞吐口径：按请求自身的墙钟时间窗计算（含 drain 尾巴），
    # 避免用名义稳压窗口衡量跨窗口完成的请求导致系统性低估。
    stamped = [r for r in rows if r.get("sent_at")]
    window = None
    if stamped:
        first_start = min(r["sent_at"] for r in stamped)
        last_end = max(r["sent_at"] + (r.get("e2e_ms") or 0.0) / 1000.0 for r in stamped)
        window = last_end - first_start
    if window is None:
        window = duration_s - warmup_s
    if window and window > 0:
        stats.rps = stats.n_total / window
        stats.output_tps = sum(r.get("completion_tokens") or 0 for r in rows) / window
    stats.prompt_tokens_avg = mean([r["prompt_tokens"] for r in ok_rows if r.get("prompt_tokens")])
    stats.completion_tokens_avg = mean(
        [r["completion_tokens"] for r in ok_rows if r.get("completion_tokens")]
    )
    # 水位判定（黑盒可观测投影）：放在所有聚合量就绪之后
    state, reasons = classify_watermark(stats, watermark or WatermarkConfig())
    stats.watermark_state = state
    stats.watermark_reasons = reasons
    return stats


@dataclass
class CapacityFinding:
    """容量结论。

    两个目标必须分开看：
    - SLA（可用性） = (A2+A3)/N：服务"接得住"的比例；
    - SLO（质量）   = A2/N：服务"接得好"的比例。

    SLA 达标但 SLO 归零，意味着服务全靠排队消化压力、没有任何实际价值，
    因此生产并发建议以**质量容量**（两者同时达标的最高连续前缀并发）为准，
    SLA 容量仅作为可用性参考单独陈述。
    """

    sla_target: float
    slo_target: float
    min_samples: int

    # 样本充足的连续前缀容量
    sla_capacity: Optional[int] = None          # 满足 SLA 目标的最高并发
    sla_at_sla_capacity: Optional[float] = None
    slo_at_sla_capacity: Optional[float] = None
    slo_capacity: Optional[int] = None          # 满足 SLO 目标的最高并发
    slo_at_slo_capacity: Optional[float] = None

    # 质量容量：SLA 与 SLO 同时达标的最高连续前缀并发（样本充足）
    quality_capacity: Optional[int] = None
    sla_at_quality_capacity: Optional[float] = None
    slo_at_quality_capacity: Optional[float] = None
    # 放宽样本量后的参考容量（低置信度；样本充足前缀为空时使用）
    reference_capacity: Optional[int] = None
    slo_at_reference_capacity: Optional[float] = None

    binding: str = "none"                       # 质量容量的约束方：slo | sla | both | none
    first_slo_breach: Optional[int] = None      # 首个样本充足且 SLO 跌破目标的并发
    best_sla: float = 0.0
    best_sla_concurrency: Optional[int] = None
    knee_concurrency: Optional[int] = None      # 首个样本充足且 SLA 跌破目标的并发
    reached: bool = False                       # 是否找到质量容量
    notes: List[str] = field(default_factory=list)


def _prefix_last_ok(stages: Sequence[StageStats], predicate) -> Optional[StageStats]:
    """连续前缀规则：从低压往高压，取最后一个满足条件的级别；遇到第一个不满足即停止。"""
    last: Optional[StageStats] = None
    for s in stages:
        if predicate(s):
            last = s
        else:
            break
    return last


def find_capacity(stages: Sequence[StageStats], analysis: AnalysisConfig) -> CapacityFinding:
    """从分级结果中提炼容量结论。

    连续前缀规则：只有从低压到高压每一级都达标的并发才认定为有效容量，
    避免中间抖动造成的高估；若更高级别重新达标会在 notes 中说明。
    样本不足（< min_samples_per_stage）的级别不参与前缀，但会进入
    reference_capacity 作为低置信度参考，防止低压优质级别被整体丢弃。
    """
    finding = CapacityFinding(
        sla_target=analysis.sla_target,
        slo_target=analysis.slo_target,
        min_samples=analysis.min_samples_per_stage,
    )
    if not stages:
        finding.notes.append("没有任何有效的压力级别数据。")
        return finding

    ordered = sorted(stages, key=lambda s: s.concurrency)
    sufficient = [s for s in ordered if s.n_total >= analysis.min_samples_per_stage]

    for s in sufficient:
        if s.n_total > 0 and s.sla >= finding.best_sla:
            finding.best_sla = s.sla
            finding.best_sla_concurrency = s.concurrency

    empty = [s for s in ordered if s.n_total == 0]
    if empty:
        finding.notes.append(
            f"并发 {', '.join(str(s.concurrency) for s in empty)} 无任何完成样本"
            "（在途请求可能全部超过 drain 窗口被放弃）：请增大 load.drain_timeout_s "
            "或 load.stage_duration_s 后复测，该级别数据不参与任何判定。"
        )

    both_ok = lambda s: s.sla >= analysis.sla_target and s.slo >= analysis.slo_target  # noqa: E731

    # --- SLA 容量（样本充足连续前缀）---
    last_sla = _prefix_last_ok(sufficient, lambda s: s.sla >= analysis.sla_target)
    if last_sla is not None:
        finding.sla_capacity = last_sla.concurrency
        finding.sla_at_sla_capacity = last_sla.sla
        finding.slo_at_sla_capacity = last_sla.slo
        if not any(s.concurrency > last_sla.concurrency for s in sufficient):
            higher = [s for s in ordered if s.concurrency > last_sla.concurrency]
            if higher:
                finding.notes.append(
                    f"SLA 容量 {last_sla.concurrency} 之上还有压力级 "
                    f"{', '.join(str(s.concurrency) for s in higher)}，但样本不足 "
                    f"{analysis.min_samples_per_stage} 个、未参与前缀判定。"
                )
            else:
                finding.notes.append(
                    f"已测至最高并发 {ordered[-1].concurrency} 仍满足 SLA 目标 "
                    f"{analysis.sla_target:.3%}，未观测到 SLA 拐点；该值只是测试梯度的上限，"
                    "不代表服务实际上限，如需确认应继续提高压力级。"
                )

    # --- SLO 容量（样本充足连续前缀）---
    last_slo = _prefix_last_ok(sufficient, lambda s: s.slo >= analysis.slo_target)
    if last_slo is not None:
        finding.slo_capacity = last_slo.concurrency
        finding.slo_at_slo_capacity = last_slo.slo

    # --- 质量容量：SLA 与 SLO 同时达标 ---
    last_quality = _prefix_last_ok(sufficient, both_ok)
    if last_quality is not None:
        finding.quality_capacity = last_quality.concurrency
        finding.sla_at_quality_capacity = last_quality.sla
        finding.slo_at_quality_capacity = last_quality.slo
        finding.reached = True
        nxt = next((s for s in sufficient if s.concurrency > last_quality.concurrency), None)
        if nxt is None:
            finding.binding = "none"
            finding.notes.append(
                f"已测至最高并发 {ordered[-1].concurrency} 仍同时满足双目标，未观测到质量拐点；"
                "如需确认上限，应继续提高压力级。"
            )
        else:
            sla_bad = nxt.sla < analysis.sla_target
            slo_bad = nxt.slo < analysis.slo_target
            finding.binding = "both" if (sla_bad and slo_bad) else ("sla" if sla_bad else "slo")
    else:
        # 样本充足级别中没有双达标前缀：给出低置信度参考容量与 SLO 跌破点
        ref = _prefix_last_ok(ordered, both_ok)
        if ref is not None:
            finding.reference_capacity = ref.concurrency
            finding.slo_at_reference_capacity = ref.slo
            finding.notes.append(
                f"样本充足（≥{analysis.min_samples_per_stage}）的压力级中无一同时满足双目标；"
                f"放宽样本量后参考容量为并发 {ref.concurrency}"
                f"（SLO={ref.slo:.4f}，样本 {ref.n_total} 条，低置信度），"
                "需延长低压级别时长后复测确认。"
            )
        else:
            finding.notes.append(
                f"所有压力级均未同时满足 SLA ≥ {analysis.sla_target:.3%} "
                f"与 SLO ≥ {analysis.slo_target:.3%}。"
            )

    # 首个 SLO 跌破点（样本充足）
    finding.first_slo_breach = next(
        (s.concurrency for s in sufficient if s.slo < analysis.slo_target), None
    )

    # SLA 前缀之外的抖动提示
    if finding.sla_capacity is not None:
        later = [
            s for s in sufficient
            if s.concurrency > finding.sla_capacity and s.sla >= analysis.sla_target
        ]
        if later:
            finding.notes.append(
                f"并发 {later[-1].concurrency} 处 SLA 重新达标（{later[-1].sla:.4f}），"
                "中间存在抖动，建议复测确认。"
            )
    else:
        finding.notes.append(
            f"没有样本充足且达到 SLA 目标 {analysis.sla_target:.3%} 的压力级；"
            + (
                f"已测最高 SLA 为 {finding.best_sla:.4f}（并发 {finding.best_sla_concurrency}）。"
                if finding.best_sla_concurrency is not None
                else "且没有任何带完成样本的压力级。"
            )
        )

    finding.knee_concurrency = next(
        (s.concurrency for s in sufficient if s.sla < analysis.sla_target), None
    )
    return finding


@dataclass
class WatermarkFinding:
    """水位边界发现（跨级汇总）。

    三个边界回答三个运维问题：
    - soft_watermark_concurrency：首个 BUSY 级——排队预警线（Router 队列上限参考）；
    - hard_watermark_concurrency：首个 OVERLOAD 级——熔断线（拒绝阈值参考；
      无限队列部署下该点是排队耗尽点，投影为客户端超时）；
    - sla_watermark_concurrency：= 质量容量——生产并发建议（与容量模型同一基准）。

    注意这是"水位的可观测投影"：软水位投影略滞后于服务端软水位触达
    （需积累足够 A3 样本才能分辨），报告中对滞后与未触达均有说明。
    """

    enabled: bool = True
    soft_watermark_concurrency: Optional[int] = None
    hard_watermark_concurrency: Optional[int] = None
    sla_watermark_concurrency: Optional[int] = None
    # concurrency -> state（含 unknown 的样本不足级），报告与审计用
    states: Dict[int, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def find_watermarks(
    stages: Sequence[StageStats],
    wm: WatermarkConfig,
    capacity: Optional[CapacityFinding] = None,
) -> WatermarkFinding:
    """从分级水位状态中发现三个水位边界。

    样本不足（n_total=0 → unknown）的级不参与边界发现（D10：不可信的
    统计量比没有更危险）；连续前缀规则同样适用——软水位边界取"从低压到
    高压第一个非 NORMAL 样本充足级"，避免个别级别抖动造成误报。
    """
    finding = WatermarkFinding(enabled=wm.enabled)
    if not wm.enabled or not stages:
        return finding

    ordered = sorted(stages, key=lambda s: s.concurrency)
    finding.states = {s.concurrency: s.watermark_state for s in ordered}

    # 只让带完成样本的级参与边界发现
    judged = [s for s in ordered if s.watermark_state != "unknown"]
    unknown = [s for s in ordered if s.watermark_state == "unknown"]
    if unknown:
        finding.notes.append(
            f"并发 {', '.join(str(s.concurrency) for s in unknown)} 无完成样本，"
            "水位状态无法判定，不参与水位边界发现。"
        )

    finding.soft_watermark_concurrency = next(
        (s.concurrency for s in judged if s.watermark_state in ("BUSY", "OVERLOAD")), None
    )
    finding.hard_watermark_concurrency = next(
        (s.concurrency for s in judged if s.watermark_state == "OVERLOAD"), None
    )
    if capacity is not None:
        finding.sla_watermark_concurrency = capacity.quality_capacity

    if finding.soft_watermark_concurrency is None and judged:
        finding.notes.append(
            f"已测至最高并发 {judged[-1].concurrency} 仍未触达软水位投影"
            "（全部 NORMAL）：只说明梯度上限内未观测到排队信号，"
            "不代表服务没有软水位，如需确认应继续提高压力级。"
        )
    elif (
        finding.soft_watermark_concurrency is not None
        and finding.hard_watermark_concurrency is None
    ):
        finding.notes.append(
            f"并发 {finding.soft_watermark_concurrency} 起进入软水位投影（BUSY），"
            "但已测范围内未触达硬水位投影（无 A1 失败、无质量崩塌）："
            "服务以排队消化压力，治理重点是扩容/调度而非熔断。"
        )
    if (
        finding.hard_watermark_concurrency is not None
        and finding.soft_watermark_concurrency is not None
        and finding.hard_watermark_concurrency <= finding.soft_watermark_concurrency
    ):
        finding.notes.append(
            f"硬水位投影（并发 {finding.hard_watermark_concurrency}）未晚于软水位投影"
            f"（并发 {finding.soft_watermark_concurrency}）：压力跳过排队区直接触发硬水位"
            "（拒绝或客户端超时），Router 准入策略可能较激进，或配置了无限队列。"
        )
    return finding
