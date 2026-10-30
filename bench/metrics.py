"""指标口径：A1/A2/A3 分类、分位数、分级统计聚合。

分类规则（与服务质量定义一一对应）：
- A1 失败请求            ：HTTP 错误 / 连接失败 / 超时 / 流错误 / 空输出
- A2 成功且满足 SLO 的请求：TTFT <= ttft_target_s 且 TPOT <= tpot_target_ms
- A3 成功但不满足 SLO 的请求：其余成功请求

派生指标：
- SLA（可用性） = (A2 + A3) / 总数
- SLO（质量）   = A2 / 总数
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import AnalysisConfig, SloConfig


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """最近秩分位数；q 取值 (0, 1]。"""
    if not values:
        return None
    ordered = sorted(values)
    if q <= 0:
        return float(ordered[0])
    rank = max(1, int(round(q * len(ordered) + 0.5)))
    rank = min(rank, len(ordered))
    return float(ordered[rank - 1])


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def classify(result: Dict[str, Any], slo: SloConfig) -> str:
    """对单条请求记录打 A1/A2/A3 标签。"""
    if result.get("status") != "ok":
        return "A1"
    ttft = result.get("ttft_ms")
    tpot = result.get("tpot_ms")
    completion = result.get("completion_tokens") or 0

    ttft_ok = ttft is not None and ttft <= slo.ttft_target_s * 1000.0
    # 输出 token 过少时 TPOT 不可靠，按配置决定是否参与判定
    if completion < slo.min_completion_tokens:
        tpot_ok = True
    else:
        tpot_ok = tpot is None or tpot <= slo.tpot_target_ms

    return "A2" if (ttft_ok and tpot_ok) else "A3"


@dataclass
class StageStats:
    stage_id: int
    concurrency: int
    duration_s: float
    warmup_s: float

    n_total: int = 0
    n_warmup: int = 0
    n_ok: int = 0
    n_failed: int = 0
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

    rps: Optional[float] = None
    output_tps: Optional[float] = None
    prompt_tokens_avg: Optional[float] = None
    completion_tokens_avg: Optional[float] = None

    ttft_target_ms: float = 0.0
    tpot_target_ms: float = 0.0
    sla_target: float = 0.0
    slo_target: float = 0.0

    @property
    def sla_met(self) -> bool:
        return self.n_total > 0 and self.sla >= self.sla_target

    @property
    def slo_met(self) -> bool:
        return self.n_total > 0 and self.slo >= self.slo_target

    def as_row(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


def compute_stage_stats(
    stage_id: int,
    concurrency: int,
    rows: Iterable[Dict[str, Any]],
    warmup_rows: int,
    slo: SloConfig,
    analysis: AnalysisConfig,
    duration_s: float,
    warmup_s: float,
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
    stats.n_total = len(rows)
    if not rows:
        return stats

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    stats.n_ok = len(ok_rows)
    stats.n_failed = stats.n_total - stats.n_ok

    for r in rows:
        klass = r.get("latency_class") or classify(r, slo)
        if klass == "A1":
            stats.a1 += 1
            kind = r.get("error_kind") or "unknown"
            stats.fail_kinds[kind] = stats.fail_kinds.get(kind, 0) + 1
        elif klass == "A2":
            stats.a2 += 1
        else:
            stats.a3 += 1
            ttft_bad = r.get("ttft_ms") is not None and r["ttft_ms"] > stats.ttft_target_ms
            completion = r.get("completion_tokens") or 0
            if completion < slo.min_completion_tokens:
                tpot_bad = False
            else:
                tpot_bad = r.get("tpot_ms") is not None and r["tpot_ms"] > stats.tpot_target_ms
            if ttft_bad and tpot_bad:
                stats.a3_both += 1
            elif ttft_bad:
                stats.a3_ttft_only += 1
            elif tpot_bad:
                stats.a3_tpot_only += 1

    stats.slo = stats.a2 / stats.n_total
    stats.sla = (stats.a2 + stats.a3) / stats.n_total

    ttfts = [r["ttft_ms"] for r in ok_rows if r.get("ttft_ms") is not None]
    tpots = [r["tpot_ms"] for r in ok_rows if r.get("tpot_ms") is not None]
    e2es = [r["e2e_ms"] for r in ok_rows if r.get("e2e_ms") is not None]

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

    measured = duration_s - warmup_s
    if measured > 0:
        stats.rps = stats.n_total / measured
        stats.output_tps = sum(r.get("completion_tokens") or 0 for r in rows) / measured
    stats.prompt_tokens_avg = _mean([r["prompt_tokens"] for r in ok_rows if r.get("prompt_tokens")])
    stats.completion_tokens_avg = _mean(
        [r["completion_tokens"] for r in ok_rows if r.get("completion_tokens")]
    )
    return stats


@dataclass
class CapacityFinding:
    """3 个 9（及 SLO 目标）并发容量结论。"""

    sla_target: float
    slo_target: float
    min_samples: int

    sla_capacity: Optional[int] = None          # 满足 SLA 目标的最高并发（连续前缀规则）
    slo_at_sla_capacity: Optional[float] = None
    sla_at_sla_capacity: Optional[float] = None
    slo_capacity: Optional[int] = None          # 满足 SLO 目标的最高并发
    slo_at_slo_capacity: Optional[float] = None

    best_sla: float = 0.0
    best_sla_concurrency: Optional[int] = None
    knee_concurrency: Optional[int] = None      # 首次跌破 SLA 目标的并发
    reached: bool = False
    notes: List[str] = field(default_factory=list)


def find_capacity(stages: Sequence[StageStats], analysis: AnalysisConfig) -> CapacityFinding:
    """从分级结果中提炼容量结论。

    连续前缀规则：只有从低压到高压每一级都达标的并发才认定为有效容量，
    避免中间抖动造成的高估；若更高级别重新达标会在 notes 中说明。
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

    for s in ordered:
        if s.n_total >= analysis.min_samples_per_stage and s.sla >= finding.best_sla:
            finding.best_sla = s.sla
            finding.best_sla_concurrency = s.concurrency

    # SLA 容量：按并发升序扫描，样本不足的级别视为中立（不打断前缀），
    # 记录最后一个达标级别，遇到第一个达标失败级别即停止。
    last_ok: Optional[StageStats] = None
    breached: Optional[StageStats] = None
    for s in ordered:
        if s.n_total < analysis.min_samples_per_stage:
            continue
        if s.sla >= analysis.sla_target:
            last_ok = s
        else:
            breached = s
            break
    if last_ok is not None:
        finding.sla_capacity = last_ok.concurrency
        finding.sla_at_sla_capacity = last_ok.sla
        finding.slo_at_sla_capacity = last_ok.slo
        finding.reached = True
        later = [
            s for s in ordered
            if breached is not None and s.concurrency > breached.concurrency
            and s.n_total >= analysis.min_samples_per_stage and s.sla >= analysis.sla_target
        ]
        if later:
            finding.notes.append(
                f"并发 {later[-1].concurrency} 处 SLA 重新达标（{later[-1].sla:.4f}），"
                "中间存在抖动，建议复测确认。"
            )
    else:
        finding.notes.append(
            f"没有样本充足且达到 SLA 目标 {analysis.sla_target:.3%} 的压力级；"
            f"已测最高 SLA 为 {finding.best_sla:.4f}（并发 {finding.best_sla_concurrency}）。"
        )

    # SLO 容量：同样规则
    last_slo_ok: Optional[StageStats] = None
    for s in ordered:
        if s.n_total < analysis.min_samples_per_stage:
            continue
        if s.slo >= analysis.slo_target:
            last_slo_ok = s
        else:
            break
    if last_slo_ok is not None:
        finding.slo_capacity = last_slo_ok.concurrency
        finding.slo_at_slo_capacity = last_slo_ok.slo

    # 拐点：首个样本充足且跌破 SLA 目标的级别
    finding.knee_concurrency = breached.concurrency if breached is not None else None

    if finding.sla_capacity is not None and finding.knee_concurrency is not None:
        finding.notes.append(
            f"并发 {finding.sla_capacity} 及以下 SLA 达标；并发 {finding.knee_concurrency} 起跌破 "
            f"{analysis.sla_target:.1%}。"
        )
    return finding
