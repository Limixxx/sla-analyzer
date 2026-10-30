"""评估报告生成：由分级统计产出带明确结论的 Markdown 报告。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import metrics
from .config import Config
from .metrics import CapacityFinding, StageStats
from .store import RecordStore


def _pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _ms(value: Optional[float]) -> str:
    return f"{value:.0f}" if value is not None else "n/a"


def _num(value: Optional[float], digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# 结论
# ---------------------------------------------------------------------- #

def build_conclusions(
    cfg: Config,
    stages: Sequence[StageStats],
    finding: CapacityFinding,
) -> List[str]:
    lines: List[str] = []
    sla_target = cfg.analysis.sla_target
    slo_target = cfg.analysis.slo_target

    if not stages:
        return ["没有可用的测试数据，无法给出结论。"]

    if finding.reached and finding.sla_capacity is not None:
        lines.append(
            f"**在并发 ≤ {finding.sla_capacity} 时，服务可用性（SLA，A2+A3 占比）"
            f"达到 {_pct(finding.sla_at_sla_capacity, 3)}，满足 {sla_target:.1%}（3 个 9）目标。**"
        )
        lines.append(
            f"该并发下的质量指标 SLO（A2 占比）为 {_pct(finding.slo_at_sla_capacity)}"
            + (
                f"，同样满足 {slo_target:.1%} 目标。"
                if (finding.slo_at_sla_capacity or 0) >= slo_target
                else f"，**低于** {slo_target:.1%} 目标，说明此点位已有请求超时延门限。"
            )
        )
        if finding.knee_concurrency is not None:
            knee = next(s for s in stages if s.concurrency == finding.knee_concurrency)
            lines.append(
                f"并发升至 {finding.knee_concurrency} 时 SLA 跌破目标（{_pct(knee.sla)}），"
                f"系统进入过载区：Router 开始缓冲（TTFT/TPOT 抬升 → A3 增多）并拒绝请求（A1 增多）。"
            )
        headroom = max(1, int(finding.sla_capacity * 0.8))
        lines.append(
            f"**建议生产并发上限设为 {finding.sla_capacity}（留 20% 余量则取 {headroom}）。**"
        )
    else:
        best = finding.best_sla_concurrency
        lines.append(
            f"**本次测试的所有压力级别均未达到 {sla_target:.1%} 的 SLA 目标。**"
            f"最高 SLA 为 {_pct(finding.best_sla, 3)}（并发 {best}）。"
        )
        if best is not None:
            lines.append(
                "建议先按第八节建议扩容/调优后复测；若业务可接受更低可用性目标，"
                f"可将当前最高 SLA 对应的并发 {best} 作为临时容量参考。"
            )

    if finding.slo_capacity is not None:
        lines.append(
            f"SLO（{slo_target:.1%}）对应的最高并发为 {finding.slo_capacity}"
            f"（SLO={_pct(finding.slo_at_slo_capacity)}）。"
        )
    return lines


def build_recommendations(
    cfg: Config,
    stages: Sequence[StageStats],
    finding: CapacityFinding,
) -> List[str]:
    recs: List[str] = []
    if not stages:
        return ["无数据，无法给出建议。"]

    # A1 构成分析
    total_fail = sum(s.a1 for s in stages)
    if total_fail > 0:
        kinds: Dict[str, int] = {}
        for s in stages:
            for kind, count in s.fail_kinds.items():
                kinds[kind] = kinds.get(kind, 0) + count
        top = sorted(kinds.items(), key=lambda kv: kv[1], reverse=True)[:3]
        desc = "、".join(f"{kind}×{count}" for kind, count in top)
        recs.append(f"失败请求（A1）共 {total_fail} 条，主要构成：{desc}。")
        if kinds.get("rejected", 0) > 0:
            recs.append(
                "存在 Router 过载拒绝：过载区应通过扩容（增加 PD 副本）或调整 Router 队列上限/"
                "拒绝阈值来治理，避免可用性断崖。"
            )
        if kinds.get("timeout", 0) > 0:
            recs.append(
                "存在整体超时：请核对 service.timeout_s 是否覆盖最长合法请求（分钟级），"
                "以及是否存在请求被无限期挂起。"
            )

    # A3 归因
    a3_total = sum(s.a3 for s in stages)
    if a3_total > 0:
        ttft_only = sum(s.a3_ttft_only for s in stages)
        tpot_only = sum(s.a3_tpot_only for s in stages)
        both = sum(s.a3_both for s in stages)
        recs.append(
            f"A3（成功但不满足 SLO）共 {a3_total} 条：TTFT 超标 {ttft_only} 条、"
            f"TPOT 超标 {tpot_only} 条、双超标 {both} 条。"
        )
        if tpot_only + both > ttft_only:
            recs.append(
                "劣化以 TPOT 为主：优化方向为提升 decode 侧吞吐（batch 调度、增大 KV 复用、"
                "扩容 decode 副本）或复核 tpot_target_ms 设定是否过严。"
            )
        elif ttft_only > 0:
            recs.append(
                "劣化以 TTFT 为主：优化方向为降低排队（Router 缓冲策略、扩容 prefill、"
                "检查调度亲和导致的热点）或复核 ttft_target_s 设定。"
            )

    if finding.reached and finding.sla_capacity is not None:
        recs.append(
            f"复测建议：固定并发 {finding.sla_capacity} 做 30 分钟以上长稳验证，"
            "确认 SLA 不随时间漂移；并按第 3 节的长度分布核对生产流量特征是否被覆盖。"
        )
    else:
        recs.append("复测建议：扩容后从最高达标失败点附近重新细化并发梯度（步长减半）定位真实拐点。")

    recs.append(
        "口径提醒：本报告的 SLA/SLO 为全请求口径（含被 Router 拒绝的 A1）；"
        "若需“成功请求口径”的时延分布，可查询 DB 中 status='ok' 的明细。"
    )
    return recs


# ---------------------------------------------------------------------- #
# 报告主体
# ---------------------------------------------------------------------- #

def render_report(
    cfg: Config,
    store: RecordStore,
    stages: Sequence[StageStats],
    finding: CapacityFinding,
    meta: Dict[str, str],
) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    started = meta.get("started_at", "unknown")
    lines: List[str] = []

    lines.append("# PD 分离推理服务质量（SLO/SLA）评估报告")
    lines.append("")
    lines.append(f"- 测试时间：{started} ~ {now}")
    lines.append(f"- 被测服务：`{cfg.service.base_url}`（`{cfg.service.path}`，模型 `{cfg.service.model}`）")
    lines.append(f"- 判定门限：TTFT ≤ {cfg.slo.ttft_target_s}s，TPOT ≤ {cfg.slo.tpot_target_ms}ms")
    lines.append(
        f"- 目标：SLA（A2+A3 占比）≥ {cfg.analysis.sla_target:.1%}（3 个 9），"
        f"SLO（A2 占比）≥ {cfg.analysis.slo_target:.1%}"
    )
    lines.append(f"- 压测模式：{cfg.load.mode}，压力级：{cfg.load.concurrency_steps}")
    lines.append("")

    # 结论 -----------------------------------------------------------------
    lines.append("## 一、结论")
    lines.append("")
    for item in build_conclusions(cfg, stages, finding):
        lines.append(f"- {item}")
    lines.append("")

    # 方法 -----------------------------------------------------------------
    lines.append("## 二、测试方法")
    lines.append("")
    lines.append("1. 以闭环方式逐级提升并发（每级含预热与稳压两段，预热数据不计入统计）。")
    lines.append("2. 每个请求按预置的长度分布合成（prompt 长度 / 输出长度均按权重采样），"
                 "并附加唯一前缀以规避前缀缓存；真实 token 数以服务端 usage 为准。")
    lines.append("3. 流式采集 TTFT（首 token 时间）与 TPOT（逐 token 平均间隔）。")
    lines.append("4. 请求三分类：A1=失败（HTTP 错误/连接失败/超时/流错误/空输出），"
                 "A2=成功且 TTFT、TPOT 双双达标，A3=成功但不达标。")
    lines.append("5. SLA=(A2+A3)/总数，SLO=A2/总数；随并发递增绘制可用性曲线，"
                 "定位 3 个 9 对应的最高并发（连续前缀规则）。")
    lines.append("")

    # 分级结果 -------------------------------------------------------------
    lines.append("## 三、分级压力测试结果")
    lines.append("")
    rows = []
    for s in sorted(stages, key=lambda x: x.concurrency):
        rows.append([
            str(s.concurrency),
            str(s.n_total),
            str(s.a1),
            str(s.a2),
            str(s.a3),
            _pct(s.sla, 3),
            _pct(s.slo, 2),
            "达标" if s.sla_met else "未达标",
            _num(s.rps, 1),
            _num(s.output_tps, 0),
        ])
    lines.append(_table(
        ["并发", "样本数", "A1 失败", "A2 达标", "A3 不达标", "SLA", "SLO", "SLA 判定", "RPS", "输出 tok/s"],
        rows,
    ))
    lines.append("")
    if any(s.n_total < cfg.analysis.min_samples_per_stage for s in stages):
        thin = [str(s.concurrency) for s in stages if s.n_total < cfg.analysis.min_samples_per_stage]
        lines.append(
            f"> 注意：并发 {', '.join(thin)} 的样本数少于 {cfg.analysis.min_samples_per_stage}，"
            "其 SLA/SLO 置信度较低，未纳入容量判定。"
        )
        lines.append("")

    # 延迟分布 -------------------------------------------------------------
    lines.append("## 四、延迟分布（仅统计成功请求）")
    lines.append("")
    rows = []
    for s in sorted(stages, key=lambda x: x.concurrency):
        rows.append([
            str(s.concurrency),
            _ms(s.ttft_p50_ms), _ms(s.ttft_p90_ms), _ms(s.ttft_p99_ms),
            _ms(s.tpot_p50_ms), _ms(s.tpot_p90_ms), _ms(s.tpot_p99_ms),
            _ms(s.e2e_p50_ms), _ms(s.e2e_p99_ms),
        ])
    lines.append(_table(
        ["并发", "TTFT p50", "TTFT p90", "TTFT p99", "TPOT p50", "TPOT p90", "TPOT p99", "E2E p50", "E2E p99"],
        rows,
    ))
    lines.append("")
    lines.append(f"（单位 ms；门限线：TTFT {cfg.slo.ttft_target_s * 1000:.0f}ms / "
                 f"TPOT {cfg.slo.tpot_target_ms:.0f}ms）")
    lines.append("")

    # A1 构成 --------------------------------------------------------------
    lines.append("## 五、失败请求（A1）构成")
    lines.append("")
    kinds: Dict[str, int] = {}
    for s in stages:
        for kind, count in s.fail_kinds.items():
            kinds[kind] = kinds.get(kind, 0) + count
    if kinds:
        rows = [[kind, str(count), _pct(count / max(1, sum(kinds.values())))]
                for kind, count in sorted(kinds.items(), key=lambda kv: kv[1], reverse=True)]
        lines.append(_table(["失败类型", "次数", "占 A1 比例"], rows))
    else:
        lines.append("全部压力级均无失败请求（A1=0）。")
    lines.append("")

    # A3 归因 --------------------------------------------------------------
    lines.append("## 六、A3（成功但不达标）归因")
    lines.append("")
    rows = []
    for s in sorted(stages, key=lambda x: x.concurrency):
        rows.append([
            str(s.concurrency), str(s.a3),
            str(s.a3_ttft_only), str(s.a3_tpot_only), str(s.a3_both),
        ])
    lines.append(_table(["并发", "A3 合计", "仅 TTFT 超标", "仅 TPOT 超标", "双超标"], rows))
    lines.append("")

    # 特征覆盖 -------------------------------------------------------------
    lines.append("## 七、请求特征覆盖")
    lines.append("")
    lines.append(f"- 计划 prompt 长度分布（token）：{cfg.requests.prompt_token_targets}"
                 f"（权重 {cfg.requests.prompt_weights}）")
    lines.append(f"- 计划输出长度分布（token）：{cfg.requests.output_token_choices}"
                 f"（权重 {cfg.requests.output_weights}）")
    lines.append(f"- 上下文上限假设：{cfg.requests.max_context_tokens} token")
    if stages:
        avg_p = stages[-1].prompt_tokens_avg
        avg_c = stages[-1].completion_tokens_avg
        lines.append(f"- 实测平均 prompt token（末级成功请求）：{_num(avg_p, 0)}")
        lines.append(f"- 实测平均输出 token（末级成功请求）：{_num(avg_c, 0)}")
    lines.append("")

    # 建议 -----------------------------------------------------------------
    lines.append("## 八、建议")
    lines.append("")
    for item in build_recommendations(cfg, stages, finding):
        lines.append(f"- {item}")
    lines.append("")

    # 附录 -----------------------------------------------------------------
    lines.append("## 附录 A：复现方式")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python bench.py --config {cfg.source_path or 'config.toml'} \\")
    lines.append(f"    --base-url {cfg.service.base_url} --model {cfg.service.model}")
    lines.append("```")
    lines.append("")
    lines.append("## 附录 B：数据库字段说明")
    lines.append("")
    lines.append("- `stages`：每级压力的配置与聚合结果（SLA/SLO/分位数/吞吐）。")
    lines.append("- `requests`：逐请求特征（长度、TTFT/TPOT/E2E、分类、失败类型），"
                 "**不含原始请求与响应内容**。")
    lines.append("- `is_warmup=1` 的为预热数据，统计时排除。")
    lines.append("")
    lines.append(f"原始数据库：`{meta.get('db_path', 'unknown')}`")
    lines.append("")
    return "\n".join(lines)


def write_report(cfg: Config, store: RecordStore, stages: Sequence[StageStats],
                 finding: CapacityFinding, meta: Dict[str, str], output_path: str) -> str:
    text = render_report(cfg, store, stages, finding, meta)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)
