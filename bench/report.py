"""评估报告生成：由分级统计产出带明确结论的 Markdown 报告。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import metrics
from .baseline import BaselineResult, Thresholds, bucket_label
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


def _ratio(current: Optional[float], base: Optional[float]) -> str:
    if current is None or base is None or base <= 0:
        return "n/a"
    return f"×{current / base:.2f}"


def _threshold_headline(
    cfg: Config,
    thresholds: Optional[Thresholds],
    buckets: Sequence[int],
) -> List[str]:
    """报告头部的门限说明（随 target_mode 变化）。"""
    if thresholds is None or cfg.slo.target_mode == "absolute":
        return [
            f"- 判定门限：TTFT ≤ {cfg.slo.ttft_target_s}s，TPOT ≤ {cfg.slo.tpot_target_ms}ms"
            f"（absolute 模式，所有长度桶统一）"
        ]
    parts = []
    for i in sorted(thresholds.ttft_ms):
        tag = "（回退）" if i in thresholds.fallback_buckets else ""
        parts.append(
            f"{bucket_label(i, buckets)}{tag} TTFT≤{thresholds.ttft_ms[i]:.0f}ms / "
            f"TPOT≤{thresholds.tpot_ms[i]:.0f}ms"
        )
    base = thresholds.baseline
    if base is not None:
        origin = (
            f"- 判定门限：baseline_relative 模式 = 基线并发 {base.concurrency}、"
            f"{base.duration_s:.0f}s 的 p99 × 系数"
            f"（TTFT ×{cfg.slo.ttft_factor}，TPOT ×{cfg.slo.tpot_factor}），按长度桶生效："
        )
    else:
        origin = "- 判定门限：baseline_relative 模式，但基线缺失，已整体回退为 absolute 配置值："
    return [origin + "；".join(parts)]


def _render_baseline_section(
    cfg: Config,
    stages: Sequence[StageStats],
    thresholds: Optional[Thresholds],
    baseline: Optional[BaselineResult],
    buckets: Sequence[int],
) -> List[str]:
    """第二节：基线剖析、生效门限与质量退化率。"""
    lines: List[str] = []
    lines.append("## 二、基线剖析与判定门限")
    lines.append("")

    if baseline is None or not baseline.buckets:
        if cfg.slo.target_mode == "baseline_relative":
            lines.append(
                "> 本次配置为 baseline_relative，但未获得基线数据（可能使用 `--no-baseline` "
                "或数据库由旧版本生成），门限已整体回退为 absolute 配置值。"
            )
        else:
            lines.append(
                "本次使用 absolute 门限（固定值，跨次运行可比，适合作为业务验收契约）。"
                "若需自动适配不同推理引擎/模型规模，可改用 `slo.target_mode=baseline_relative`，"
                "由 Phase 0 低压基线的 p99 × 系数生成各长度桶的门限。"
            )
        lines.append("")
        return lines

    index = {b.bucket: b for b in baseline.buckets}
    lines.append(
        f"Phase 0 在并发 {baseline.concurrency}、{baseline.duration_s:.0f}s 的低压下测量无压力时延。"
        "基线只统计分位数与失败数，不做达标判定——基线是尺子的来源，再拿它判定会形成循环。"
    )
    lines.append("")

    rows = []
    for i in sorted(index):
        b = index[i]
        ttft_limit = thresholds.ttft_limit(i) if thresholds else None
        tpot_limit = thresholds.tpot_limit(i) if thresholds else None
        src = "回退 absolute" if (thresholds and i in thresholds.fallback_buckets) else "基线×系数"
        rows.append([
            bucket_label(i, buckets),
            str(b.n_ok),
            str(b.a1),
            _ms(b.ttft_p50_ms), _ms(b.ttft_p95_ms), _ms(b.ttft_p99_ms),
            _ms(b.tpot_p50_ms), _ms(b.tpot_p99_ms),
            f"{ttft_limit:.0f}" if ttft_limit else "n/a",
            f"{tpot_limit:.0f}" if tpot_limit else "n/a",
            src,
        ])
    lines.append(_table(
        ["长度桶", "样本", "A1", "TTFT p50", "TTFT p95", "TTFT p99",
         "TPOT p50", "TPOT p99", "生效 TTFT 门限", "生效 TPOT 门限", "来源"],
        rows,
    ))
    lines.append("")
    for note in baseline.notes:
        lines.append(f"> {note}")
    if baseline.notes:
        lines.append("")

    # ---- 质量退化率：与门限无关的诊断量 ----
    with_buckets = [s for s in stages if s.bucket_stats]
    if with_buckets:
        lines.append("**质量退化率**（各级 p99 ÷ 基线 p99，与绝对门限无关，跨引擎/模型可比）：")
        lines.append("")
        header = ["并发"]
        header += [f"TTFT ×（{bucket_label(i, buckets)}）" for i in sorted(index)]
        header += [f"TPOT ×（{bucket_label(i, buckets)}）" for i in sorted(index)]
        flat: List[List[str]] = []
        for s in sorted(with_buckets, key=lambda x: x.concurrency):
            row = [str(s.concurrency)]
            for i in sorted(index):
                row.append(_ratio(s.bucket_stats.get(i, {}).get("ttft_p99_ms"), index[i].ttft_p99_ms))
            for i in sorted(index):
                row.append(_ratio(s.bucket_stats.get(i, {}).get("tpot_p99_ms"), index[i].tpot_p99_ms))
            flat.append(row)
        lines.append(_table(header, flat))
        lines.append("")
        lines.append(
            "> 读法：退化率接近 1 表示该长度桶尚未受压力影响；某一桶的 TTFT 退化率显著高于其他桶，"
            "说明劣化集中在对应长度区间（长 prompt 指向 prefill 容量，短 prompt 指向准入排队）。"
        )
        lines.append("")
    return lines


# ---------------------------------------------------------------------- #
# 结论
# ---------------------------------------------------------------------- #

def build_conclusions(
    cfg: Config,
    stages: Sequence[StageStats],
    finding: CapacityFinding,
) -> List[str]:
    """结论：以质量容量（SLA∧SLO 同时达标）为生产并发建议的唯一基准。

    SLA 达标仅表示"接得住"，SLO 达标才表示"接得好"；SLA 100% 而 SLO 归零
    意味着服务全靠排队消化压力，该容量不具备生产意义。
    """
    lines: List[str] = []
    sla_target = cfg.analysis.sla_target
    slo_target = cfg.analysis.slo_target

    if not stages:
        return ["没有可用的测试数据，无法给出结论。"]

    by_conc = {s.concurrency: s for s in stages}

    # 零样本退化场景：没有任何请求完成（典型原因是 E2E 超过每级时长 + drain 窗口）
    if all(s.n_total == 0 for s in stages):
        total_aborted = sum(s.n_aborted for s in stages)
        msg = (
            "**所有压力级均无完成请求，无法评估：单请求时长可能超过每级时长 + drain 窗口，"
            "在途请求全部被放弃。**"
        )
        if total_aborted:
            msg += f"（共放弃 {total_aborted} 个在途请求）"
        msg += "请增大 load.drain_timeout_s / load.stage_duration_s 后复测。"
        return [msg]

    # ---- 1) 质量容量 / 生产并发建议 ----
    q = finding.quality_capacity
    if q is not None:
        sq = by_conc[q]
        lines.append(
            f"**在并发 ≤ {q}（已测试梯度内）时，服务同时满足 SLA ≥ {sla_target:.1%} 与 "
            f"SLO ≥ {slo_target:.1%}，该并发为质量有效容量"
            f"（SLA={_pct(sq.sla, 3)}，SLO={_pct(sq.slo, 2)}）。**"
        )
        headroom = max(1, int(q * 0.8))
        lines.append(f"**建议生产并发上限设为 {q}（留 20% 余量则取 {headroom}）。**")
    else:
        ref = finding.reference_capacity
        if ref is not None:
            sr = by_conc[ref]
            lines.append(
                f"**样本充足的压力级中，无一同时满足 SLA ≥ {sla_target:.1%} 与 "
                f"SLO ≥ {slo_target:.1%}，无法给出高置信度的生产并发建议。**"
            )
            lines.append(
                f"放宽样本量后，低置信度参考容量为并发 {ref}"
                f"（SLA={_pct(sr.sla, 3)}，SLO={_pct(sr.slo, 2)}，仅 {sr.n_total} 个样本）；"
                "请增大低压级别时长后复测确认。"
            )
        else:
            lines.append(
                f"**所有测试并发均未同时满足 SLA ≥ {sla_target:.1%} 与 SLO ≥ {slo_target:.1%}，"
                "当前无可推荐的生产并发。**"
            )
        if finding.first_slo_breach is not None:
            sb = by_conc[finding.first_slo_breach]
            lines.append(
                f"并发 {finding.first_slo_breach} 起 SLO 降至 {_pct(sb.slo, 2)}"
                f"（低于 {slo_target:.1%}），请求质量开始系统性劣化。"
            )

    # ---- 2) 3 个 9（SLA 口径）单独陈述 ----
    if finding.sla_capacity is not None:
        sc = by_conc[finding.sla_capacity]
        tail = (
            f"；但该点 SLO 仅 {_pct(sc.slo, 2)}，此容量无实际服务质量意义。"
            if sc.slo < slo_target else "。"
        )
        lines.append(
            f"3 个 9 口径（SLA ≥ {sla_target:.1%}）对应的最高并发为 "
            f"{finding.sla_capacity}（SLA={_pct(sc.sla, 3)}，该点 SLO={_pct(sc.slo, 2)}）{tail}"
        )
    else:
        lines.append(
            f"3 个 9 口径（SLA ≥ {sla_target:.1%}）未达到："
            + (
                f"样本充足级别中最高 SLA 为 {_pct(finding.best_sla, 3)}"
                f"（并发 {finding.best_sla_concurrency}）。"
                if finding.best_sla_concurrency is not None
                else "没有任何带完成样本的压力级。"
            )
        )

    # ---- 3) SLO 口径容量 ----
    if finding.slo_capacity is not None:
        lines.append(
            f"SLO（{slo_target:.1%}）口径对应的最高并发为 {finding.slo_capacity}"
            f"（SLO={_pct(finding.slo_at_slo_capacity)}）。"
        )

    # ---- 4) 过载形态（按实际数据描述，不用模板硬编码） ----
    if finding.knee_concurrency is not None:
        knee = by_conc.get(finding.knee_concurrency)
        if knee is not None:
            if knee.a1 > 0:
                shape = f"出现失败请求（A1={knee.a1}），Router 进入过载拒绝区"
            else:
                shape = f"请求全部成功但时延超标（A3={knee.a3}），过载表现为排队而非拒绝"
            lines.append(
                f"并发 {finding.knee_concurrency} 起 SLA 跌破 {sla_target:.1%}"
                f"（{_pct(knee.sla, 3)}），{shape}。"
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
        # 主导判据：看"与哪项时延相关的超标请求更多"。
        # 双超标同时计入两侧——排队先爆 TTFT，TPOT 往往是被连带拖垮的。
        ttft_side = ttft_only + both
        tpot_side = tpot_only + both
        if cfg.slo.target_mode == "baseline_relative":
            ttft_knob = f"或复核 slo.ttft_factor（当前 {cfg.slo.ttft_factor}）是否过严"
            tpot_knob = f"或复核 slo.tpot_factor（当前 {cfg.slo.tpot_factor}）是否过严"
        else:
            ttft_knob = "或复核 ttft_target_s 设定是否过严。"
            tpot_knob = "或复核 tpot_target_ms 设定。"
        if ttft_side > tpot_side:
            recs.append(
                "劣化以 TTFT 为主（TTFT 相关超标 "
                f"{ttft_side} 条 vs TPOT 相关 {tpot_side} 条）：优化方向为降低排队——"
                f"扩容 PD 副本、提高 Router 准入/调度并发、排查 prefill 侧热点；{ttft_knob}"
            )
        elif tpot_side > ttft_side:
            recs.append(
                "劣化以 TPOT 为主（TPOT 相关超标 "
                f"{tpot_side} 条 vs TTFT 相关 {ttft_side} 条）：优化方向为提升 decode 侧吞吐"
                f"（batch 调度、增大 KV 复用、扩容 decode 副本）{tpot_knob}"
            )
        else:
            recs.append(
                "TTFT 与 TPOT 相关超标数量相当：需同时治理排队与 decode 吞吐，"
                "或复核两项门限设定。"
            )
        if sum(s.a1 for s in stages) == 0:
            recs.append(
                "本次过载全程无拒绝请求（A1=0），压力全部由排队消化：TTFT 随并发持续抬升"
                "是主要矛盾，容量结论应以 SLO（质量）而非 SLA（可用性）为准。"
            )

    target_conc = finding.quality_capacity or finding.reference_capacity
    if target_conc is not None:
        recs.append(
            f"复测建议：固定并发 {target_conc} 做 30 分钟以上长稳验证，"
            "确认指标不随时间漂移；并按请求特征覆盖一节的长度分布核对生产流量特征是否被覆盖。"
        )
    else:
        recs.append("复测建议：先按上述方向扩容/调优，再从最低压力级开始细化梯度定位质量拐点。")

    if cfg.slo.target_mode == "baseline_relative":
        recs.append(
            "门限来源提醒：本次为 baseline_relative 口径，门限由低压基线 p99 × 系数生成，"
            "已自动适配当前引擎/模型的绝对时延水平。跨引擎/跨模型对比时应改为 absolute "
            "口径（或在报告中并列两种口径），否则“达标”只对当次基线有意义。"
        )
    else:
        recs.append(
            "门限来源提醒：本次为 absolute 口径（固定门限），适合作为业务验收契约；"
            "若需自动适配不同引擎/模型规模，可改用 slo.target_mode=baseline_relative。"
        )

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
    thresholds: Optional[Thresholds] = None,
    baseline: Optional[BaselineResult] = None,
) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    started = meta.get("started_at", "unknown")
    lines: List[str] = []
    buckets = cfg.requests.length_buckets

    lines.append("# PD 分离推理服务质量（SLO/SLA）评估报告")
    lines.append("")
    lines.append(f"- 测试时间：{started} ~ {now}")
    lines.append(f"- 被测服务：`{cfg.service.base_url}`（`{cfg.service.path}`，模型 `{cfg.service.model}`）")
    lines.extend(_threshold_headline(cfg, thresholds, buckets))
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

    # 基线与门限 -----------------------------------------------------------
    lines.extend(_render_baseline_section(cfg, stages, thresholds, baseline, buckets))

    # 方法 -----------------------------------------------------------------
    lines.append("## 三、测试方法")
    lines.append("")
    lines.append("1. 以闭环方式逐级提升并发（每级含预热与稳压两段，预热数据不计入统计）。")
    lines.append("2. 每个请求按预置的长度分布合成（prompt 长度 / 输出长度均按权重采样），"
                 "并附加唯一前缀以规避前缀缓存；真实 token 数以服务端 usage 为准。")
    lines.append("3. 流式采集 TTFT（首 token 时间）与 TPOT（逐 token 平均间隔）。")
    lines.append("4. 请求三分类：A1=失败（HTTP 错误/连接失败/超时/流错误/空输出），"
                 "A2=成功且 TTFT、TPOT 双双不超出门限，A3=成功但不达标。")
    lines.append("5. SLA=(A2+A3)/总数，SLO=A2/总数；随并发递增绘制可用性/质量曲线，"
                 "分别定位 3 个 9（SLA 口径）与 SLO 口径对应的最高并发（连续前缀规则），"
                 "并以两者同时达标的**质量容量**作为生产并发建议基准。")
    lines.append("6. 闭环低压级别的样本产出受单请求 E2E 限制，测量段样本不足时本级自动延长"
                 "（上限 load.max_stage_duration_s）；本级结束时仍在途的请求记为放弃"
                 "（aborted），不计入比率。")
    if cfg.slo.target_mode == "baseline_relative":
        lines.append(
            "7. 判定门限来自第二节的低压基线（基线 p99 × 系数），与压力解耦："
            "同一把尺子贯穿全部压力级，因此跨级曲线可比；换引擎/模型只需重测基线。"
        )
    lines.append("")

    # 分级结果 -------------------------------------------------------------
    lines.append("## 四、分级压力测试结果")
    lines.append("")
    rows = []
    for s in sorted(stages, key=lambda x: x.concurrency):
        rows.append([
            str(s.concurrency),
            str(s.n_total),
            str(s.a1),
            str(s.a2),
            str(s.a3),
            str(s.n_aborted) if s.n_aborted else "0",
            _pct(s.sla, 3),
            "达标" if s.sla_met else "未达标",
            _pct(s.slo, 2),
            "达标" if s.slo_met else "未达标",
            _num(s.rps, 1),
            _num(s.output_tps, 0),
        ])
    lines.append(_table(
        ["并发", "样本数", "A1 失败", "A2 达标", "A3 不达标", "放弃",
         "SLA", "SLA 判定", "SLO", "SLO 判定", "RPS", "输出 tok/s"],
        rows,
    ))
    lines.append("")
    if any(s.n_total < cfg.analysis.min_samples_per_stage for s in stages):
        thin = [str(s.concurrency) for s in stages if s.n_total < cfg.analysis.min_samples_per_stage]
        lines.append(
            f"> 注意：并发 {', '.join(thin)} 的样本数少于 {cfg.analysis.min_samples_per_stage}"
            "（受单请求时长限制，已按 max_stage_duration_s 尽力延长），"
            "其达标判定为低置信度，仅作参考容量依据，不参与高置信度容量前缀。"
        )
        lines.append("")
    total_aborted = sum(s.n_aborted for s in stages)
    if total_aborted > 0:
        lines.append(
            f"> 另有 {total_aborted} 个请求在本级结束时仍在途、超过 drain 上限被放弃"
            "（status=aborted），不计入 SLA/SLO 分母；占比偏高时应增大 load.drain_timeout_s。"
        )
        lines.append("")

    # 延迟分布 -------------------------------------------------------------
    lines.append("## 五、延迟分布（仅统计成功请求）")
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
    if thresholds is not None and cfg.slo.target_mode == "baseline_relative":
        pairs = "；".join(
            f"{bucket_label(i, buckets)} TTFT≤{thresholds.ttft_ms[i]:.0f}ms / "
            f"TPOT≤{thresholds.tpot_ms[i]:.0f}ms"
            for i in sorted(thresholds.ttft_ms)
        )
        lines.append(f"（单位 ms；判定门限按长度桶生效：{pairs}）")
    else:
        lines.append(f"（单位 ms；门限线：TTFT {cfg.slo.ttft_target_s * 1000:.0f}ms / "
                     f"TPOT {cfg.slo.tpot_target_ms:.0f}ms）")
    lines.append("")

    # A1 构成 --------------------------------------------------------------
    lines.append("## 六、失败请求（A1）构成")
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
    lines.append("## 七、A3（成功但不达标）归因")
    lines.append("")
    rows = []
    for s in sorted(stages, key=lambda x: x.concurrency):
        rows.append([
            str(s.concurrency), str(s.a3),
            str(s.a3_ttft_only), str(s.a3_tpot_only), str(s.a3_both),
        ])
    lines.append(_table(["并发", "A3 合计", "仅 TTFT 超标", "仅 TPOT 超标", "双超标"], rows))
    lines.append("")

    # 按长度桶拆分：区分"长度天然慢"与"受压后排队慢"
    if any(s.bucket_stats for s in stages):
        lines.append("**按长度桶拆分**（回答“是所有长度都劣化，还是只有长 prompt 劣化”）：")
        lines.append("")
        all_buckets = sorted({i for s in stages for i in s.bucket_stats})
        rows = []
        for s in sorted(stages, key=lambda x: x.concurrency):
            if not s.bucket_stats:
                continue
            row = [str(s.concurrency)]
            for i in all_buckets:
                e = s.bucket_stats.get(i, {})
                row.append(f"{e.get('a3', 0)}/{e.get('a2', 0) + e.get('a3', 0)}")
            rows.append(row)
        lines.append(_table(
            ["并发"] + [f"A3/A2（{bucket_label(i, cfg.requests.length_buckets)}）" for i in all_buckets],
            rows,
        ))
        lines.append("")
        lines.append(
            "> 读法：若 A3 集中在长 prompt 桶而短 prompt 桶仍全 A2，劣化源于 prefill 侧"
            "（长上下文计算量）；若各桶同步劣化，则是全局排队或准入问题。"
        )
        lines.append("")

    # 特征覆盖 -------------------------------------------------------------
    lines.append("## 八、请求特征覆盖")
    lines.append("")
    derived = set(cfg.derived_lengths)
    prompt_note = "（按 max_context_tokens 自动推导，建议按生产流量显式配置）" \
        if "prompt_token_targets" in derived else ""
    lines.append(f"- 计划 prompt 长度分布（token）：{cfg.requests.prompt_token_targets}"
                 f"（权重 {cfg.requests.prompt_weights}）{prompt_note}")
    lines.append(f"- 计划输出长度分布（token）：{cfg.requests.output_token_choices}"
                 f"（权重 {cfg.requests.output_weights}）")
    lines.append(f"- 上下文上限假设：{cfg.requests.max_context_tokens} token"
                 + ("；长度分桶上界亦为自动推导" if "length_buckets" in derived else ""))

    def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
        total = float(sum(weights))
        if total <= 0:
            return None
        return sum(v * w for v, w in zip(values, weights)) / total

    planned_p = _weighted_mean(cfg.requests.prompt_token_targets, cfg.requests.prompt_weights)
    planned_c = _weighted_mean(cfg.requests.output_token_choices, cfg.requests.output_weights)
    ok_stages = [s for s in stages if s.n_ok > 0 and s.prompt_tokens_avg is not None]
    if ok_stages and planned_p:
        total_ok = sum(s.n_ok for s in ok_stages)
        avg_p = sum(s.prompt_tokens_avg * s.n_ok for s in ok_stages) / total_ok
        dev = (avg_p - planned_p) / planned_p
        lines.append(
            f"- 实测平均 prompt token（全部成功请求按级加权）：{_num(avg_p, 0)}"
            f"（计划 {planned_p:.0f}，偏差 {dev:+.1%}）"
        )
        if abs(dev) > 0.15:
            suggest = cfg.requests.chars_per_token / (1.0 + dev)
            lines.append(
                f"- ⚠️ prompt 长度偏差超过 15%：requests.chars_per_token="
                f"{cfg.requests.chars_per_token} 与服务端 tokenizer 不符，"
                f"建议校准为 ≈{suggest:.2f} 后复测，否则长度分布未按预期覆盖。"
            )
    if ok_stages and planned_c:
        total_ok = sum(s.n_ok for s in ok_stages)
        avg_c = sum(s.completion_tokens_avg * s.n_ok for s in ok_stages) / total_ok
        lines.append(f"- 实测平均输出 token（全部成功请求按级加权）：{_num(avg_c, 0)}"
                     f"（计划 {planned_c:.0f}）")
    lines.append("")

    # 建议 -----------------------------------------------------------------
    lines.append("## 九、建议")
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
    lines.append("- `stages`：每级压力的配置与聚合结果（SLA/SLO/分位数/吞吐）。"
                 "`kind='stage'` 为正式压力级，`kind='baseline'` 为 Phase 0 基线级（不参与容量判定）。")
    lines.append("- `requests`：逐请求特征（长度、TTFT/TPOT/E2E、分类、失败类型），"
                 "**不含原始请求与响应内容**。")
    lines.append("- `stage_buckets`：每级按 prompt 长度桶的拆分（A2/A3/超标归因/该桶生效门限）。")
    lines.append("- `baseline`：Phase 0 各长度桶的时延分位数，是 baseline_relative 门限的来源。")
    lines.append("- `is_warmup=1` 的为预热数据，统计时排除。")
    lines.append("- `status='aborted'` 的为本级结束时仍在途、超过 drain 上限被放弃的请求，"
                 "不计入 SLA/SLO 分母。")
    lines.append("")
    lines.append(f"原始数据库：`{meta.get('db_path', 'unknown')}`")
    lines.append("")
    return "\n".join(lines)


def write_report(cfg: Config, store: RecordStore, stages: Sequence[StageStats],
                 finding: CapacityFinding, meta: Dict[str, str], output_path: str,
                 thresholds: Optional[Thresholds] = None,
                 baseline: Optional[BaselineResult] = None) -> str:
    text = render_report(cfg, store, stages, finding, meta, thresholds, baseline)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)
