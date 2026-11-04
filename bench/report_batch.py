"""吞吐（容量）报告：``report_batch.py``（docs/07 §7.10.6）。

与 ``report.py``（在线/延迟报告）消费**同一批** StageStats 事实，各自
产出报告与工件——测量管道不做场景裁剪，差异只在分析选择
（``report.kinds``，§7.10.1）。本模块因此只做"容量视角的呈现"：

- 服务等级由吞吐单位承载：SLA = 膝点 ``ok_rps``×60（RPM，可用性/容量），
  SLO = 膝点 ``input_tps+output_tps``（全量 TPS，目标/工作量），同在
  A1 拐点（``find_service_levels(..., a1_onset=True)``，§7.10.3）；
- 容量分析不消费质量维度：A3 照常被采集与计数，但本报告不引用；
- 软退化（全程无 A1）服务以平台期第二读数收场（D23，§7.10.3）；
- 报告期执行两道数据质量闸门（D24）并汇写容量面 ``throughput_model.npz``
  （坐标为 (L_in, L_out) 单元格，D22；派生量不落库，C3），使
  ``--report-only`` 可从旧库重建出同一结论。

依赖方向（§7.10.7 隔离规则）：本模块 import 在线侧数据结构
（metrics/format/store）与吞吐模块（throughput）；在线侧模块零容量
import，``bench.py`` 组合层是唯一同时知晓两种分析的地方。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import metrics
from .config import Config
from .format import num, pct, table
from .metrics import CapacityFinding, StageStats, split_fail_kinds
from .store import RecordStore
from .throughput import (
    ThroughputError,
    ThroughputPoint,
    ThroughputSurface,
    surface_path_exists,
)

# 平台佐证判据：连续两级有效完成速率增幅不足 5% 视为吞吐不再随负载增长
# （§7.10.6）。增幅 ∈ [0, 5%) 为趋平（饱和形态），< 0 为回落（已过膝点坍塌），
# 两者都佐证膝点，但成因不同，报告中分开标注。
_PLATEAU_GROWTH = 0.05

# 数据质量闸门（D24）容差与最少样本：偏差超容差记诊断行（只警告不阻断）；
# 样本过少时不判读——小样本均值残差噪声大，喊出来反而误导。
_CONSISTENCY_TOL = 0.15
_MIN_DIAG_SAMPLES = 5


# ---------------------------------------------------------------------- #
# 膝点与吞吐口径
# ---------------------------------------------------------------------- #
def find_knee_stage(
    stages: Sequence[StageStats], a1_onset: bool = True
) -> Tuple[Optional[StageStats], List[str]]:
    """容量膝点：首个出现 A1 的负载水平（a1_onset 口径）。

    返回 (膝点级或 None, 理由)。无完成样本的级不参与；膝点未触达时
    第二个返回值为"只对该梯度范围有效"的诚实说明。
    """
    ordered = sorted((s for s in stages if s.n_total > 0), key=lambda s: s.concurrency)
    if not ordered:
        return None, ["没有任何带完成样本的压力级，无法确定容量膝点。"]
    for s in ordered:
        if s.a1 > 0:
            return s, []
    top = ordered[-1].concurrency
    return None, [
        f"已测至最高负载 {top} 全程未出现 A1：梯度上限内服务未饱和，"
        "膝点未触达——本次结论只对该负载范围有效，需提高梯度上限复测。"
    ]


def _dominant_family(stats: StageStats) -> str:
    """A1 主导家族（timeout / reject / other；无失败为 none）。"""
    timeout_n, reject_n, other_n = stats.a1_families()
    if stats.a1 == 0:
        return "none"
    if timeout_n >= reject_n and timeout_n >= other_n:
        return "timeout"
    if reject_n >= other_n:
        return "reject"
    return "other"


def _run_io_len(cfg: Config) -> Optional[Tuple[int, int]]:
    """本次运行的 (输入, 输出) 长度坐标：仅单组合配置可取（D22）。

    容量面按 (L_in, L_out) 单元格索引且每个点应为"定长运行"的读数
    （§7.10.5）；混合输入档或混合输出档的运行不对应单一 cell，用均值
    近似会让插值面失真，因此不猜坐标——返回 None，由调用方跳过容量面
    写入并说明原因（D21"定点测量"纪律在二维下的推广）。
    """
    ins = list(cfg.requests.prompt_token_targets)
    outs = list(cfg.requests.output_token_choices)
    if len(ins) == 1 and len(outs) == 1:
        return int(ins[0]), int(outs[0])
    return None


def capacity_stage(stages: Sequence[StageStats], knee: StageStats) -> StageStats:
    """容量读数级：膝点级及低压各级中 ok_rps 的峰值所在级。

    文档口径 SLA = ok_rps(c*)×60 成立的前提是"膝点处吞吐即峰值"——
    排队型饱和确实如此；但**硬拒绝型**服务（有界队列 + 429/503）在膝点
    级即坍塌（该级大部分请求被拒），ok_rps 反而远低于低压峰值。统计
    诚实原则（AGENT.md）要求容量读数取峰值：理想膝点处二者一致，
    坍塌时取低压峰值并显式说明，不把坍塌值当成容量。
    """
    candidates = [
        s for s in stages
        if s.n_total > 0 and s.ok_rps is not None and s.concurrency <= knee.concurrency
    ]
    if not candidates:
        return knee
    return max(candidates, key=lambda s: (s.ok_rps, -s.concurrency))


def knee_point(
    cfg: Config, knee: StageStats, cap: StageStats, source: str = ""
) -> Optional[ThroughputPoint]:
    """由膝点构造容量面事实点（token 侧可缺，缺则不编造）。

    mu 取容量读数级（峰值 ok_rps），req_conc 取同一负载水平（取得峰值的
    并发/在途数），n_samples 取容量读数级样本数，坐标取本次运行的
    (L_in, L_out) 组合（_run_io_len，混合档位返回 None 不入面）。膝点
    （开始出现失败的负载水平）见报告第三节，不入容量面——反问题问的
    是"扛得住要多少"。
    """
    io = _run_io_len(cfg)
    if io is None or cap.ok_rps is None:
        return None
    return ThroughputPoint(
        input_len=io[0],
        output_len=io[1],
        mu_rpm=cap.ok_rps * 60.0,
        mu_tps=cap.total_tps,
        input_tps=cap.input_tps,
        output_tps=cap.output_tps,
        req_conc=cap.concurrency,
        n_samples=cap.n_total,
        a1_family=_dominant_family(knee),
        source=source,
        origin="knee",
    )


def plateau_reading(
    stages: Sequence[StageStats], margin: float = _PLATEAU_GROWTH
) -> Tuple[Optional[StageStats], List[str]]:
    """软退化（全程无 A1）服务的平台期第二读数（D23，§7.10.3）。

    判据（两条同时满足才给读数）：
    - 末级边际增幅 < margin（吞吐走平）；
    - 末级 ok_rps 不低于全程峰值的 (1 - margin)（顶部平坦——用"差值带"
      而非"峰值必须落在末级"判定，容忍测量噪声把峰值推到倒数第二级）。

    满足 → 返回峰值级作为"范围内饱和读数"；不满足 → 返回 None 并说明
    卡在哪条判据上（仍在爬升 / 回落可疑形态 / 样本不足以判读）。该读数
    **非膝点**：服务不拒绝、只变慢，峰值 ok_rps 是梯度范围内的饱和平台
    估计，只对已测负载范围与当次超时预算有效（D25）。
    """
    ordered = sorted(
        (s for s in stages if s.n_total > 0 and s.ok_rps is not None),
        key=lambda s: s.concurrency,
    )
    if len(ordered) < 2:
        return None, ["带有效完成速率的压力级不足两级，无法判读平台形态。"]
    last, prev = ordered[-1], ordered[-2]
    peak = max(ordered, key=lambda s: (s.ok_rps, -s.concurrency))
    growth = (last.ok_rps - prev.ok_rps) / prev.ok_rps if prev.ok_rps > 0 else 1.0
    if growth >= margin:
        return None, [
            f"末级（负载 {last.concurrency}）边际增幅 {growth:+.1%} ≥ {margin:.0%}，"
            "吞吐仍在爬升：范围内未饱和，不给平台读数。"
        ]
    if last.ok_rps < peak.ok_rps * (1.0 - margin):
        return None, [
            f"末级（负载 {last.concurrency}）ok_rps 已较峰值回落超过 {margin:.0%}"
            f"（{last.ok_rps:.2f} vs 峰值 {peak.ok_rps:.2f} rps）且全程无 A1："
            "吞吐回落而无失败属可疑形态（疑似计时污染或外部干扰），建议复测核实，"
            "不给平台读数。"
        ]
    return peak, [
        f"负载 {last.concurrency} 之上全程未出现 A1 且吞吐已走平"
        f"（末级边际 {growth:+.1%} < {margin:.0%}，峰值与末级差 ≤ {margin:.0%}）："
        f"软退化平台形态。范围内饱和读数取峰值级 负载 {peak.concurrency} 的 "
        f"ok_rps×60——**非膝点读数**，是排队型服务的平台估计，只对已测负载范围"
        "与本次超时预算有效。"
    ]


def plateau_point(
    cfg: Config, cap: StageStats, source: str = ""
) -> Optional[ThroughputPoint]:
    """由平台峰值级构造容量面事实点（D23）：a1_family=none、origin=plateau。

    软退化服务没有 A1，过载形态无从归因，家族恒为 none；origin=plateau
    使容量面与后续 --capacity-plan 能区分"膝点实测"与"平台估计"两种
    读数口径（与 source 的 db 路径审计含义分开）。
    """
    io = _run_io_len(cfg)
    if io is None or cap.ok_rps is None:
        return None
    return ThroughputPoint(
        input_len=io[0],
        output_len=io[1],
        mu_rpm=cap.ok_rps * 60.0,
        mu_tps=cap.total_tps,
        input_tps=cap.input_tps,
        output_tps=cap.output_tps,
        req_conc=cap.concurrency,
        n_samples=cap.n_total,
        a1_family="none",
        source=source,
        origin="plateau",
    )


# ---------------------------------------------------------------------- #
# 容量面汇写（报告期；--report-only 可重建）
# ---------------------------------------------------------------------- #
def merge_surface_point(
    cfg: Config, point: Optional[ThroughputPoint], reason: str = ""
) -> Tuple[Optional[str], List[str]]:
    """把本次读数并入容量面文件（replace-per-(L_in, L_out) 单元格）。返回 (路径或 None, 说明)。

    不同边界（base_url）的面对比用两个文件、不混装：既有文件记录的是
    别的边界时拒绝合并并原样保留，由报告与 stdout 说明。
    """
    path = cfg.report.throughput_model_path
    notes: List[str] = []
    if point is None:
        return None, notes + [reason or "本次运行未产生可入容量面的读数事实。"]
    try:
        if surface_path_exists(path):
            existing = ThroughputSurface.load(path)
            bound = existing.meta.get("base_url")
            if bound and bound != cfg.service.base_url:
                return None, notes + [
                    f"容量面 {path} 记录的是另一个边界（{bound}），"
                    f"与本次（{cfg.service.base_url}）不混装；未更新。"
                ]
            existing.merge([point])
            existing.meta.setdefault("base_url", cfg.service.base_url)
            existing.save(path)
            return path, notes + [f"本次读数已并入既有容量面（按 (L_in, L_out) 单元格覆盖更新）：{path}"]
        surface = ThroughputSurface.fit([point], {"base_url": cfg.service.base_url})
        surface.save(path)
        return path, notes + [f"容量面已创建：{path}"]
    except (ThroughputError, OSError, ValueError) as exc:
        # 容量面是派生工件，写失败不影响报告结论；显式说明而非静默
        return None, notes + [f"容量面写入失败（{type(exc).__name__}: {exc}）；报告结论不受影响。"]


# ---------------------------------------------------------------------- #
# 报告渲染
# ---------------------------------------------------------------------- #
def _header_lines(cfg: Config, meta: Dict[str, str], sl_notes: Sequence[str]) -> List[str]:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    started = meta.get("started_at", "unknown")
    ins = list(cfg.requests.prompt_token_targets)
    outs = list(cfg.requests.output_token_choices)
    in_txt = (
        f"固定输入长度 {ins[0]} tokens（单输入档）"
        if len(ins) == 1
        else f"混合输入长度 {ins}（多档不入容量面）"
    )
    out_txt = (
        f"，固定输出长度 {outs[0]} tokens"
        if len(outs) == 1
        else f"，混合输出长度 {outs}（多档不入容量面）"
    )
    len_txt = in_txt + out_txt
    lines = [
        "# 推理服务容量（吞吐）评估报告",
        "",
        f"- 测量边界：`{cfg.service.base_url}`（`{cfg.service.path}`，模型 `{cfg.service.model}`）",
        f"- 运行时刻：{started} ~ {now}",
        f"- 配置快照：模式 {cfg.load.mode}，负载阶梯 {cfg.load.concurrency_steps}，"
        f"每级 {cfg.load.stage_duration_s:g}s（预热 {cfg.load.warmup_s:g}s）；{len_txt}；"
        f"门限 {cfg.slo.target_mode}；Phase 0 基线 {'开' if cfg.load.baseline_enabled else '关'}",
        f"- 本容量读数的超时预算：service.timeout_s = {cfg.service.timeout_s:g}s"
        "（膝点 = 服务拒绝**或**客户放弃，D25；同一服务在不同超时预算下膝点与读数不同，"
        "数字必须连同预算一起解读）",
        f"- 原始数据库：`{meta.get('db_path', 'unknown')}`",
        "",
        "> 范围声明：本报告只覆盖 `service.base_url` 指向的**推理服务本身**的容量。"
        "上层系统（批量调度前端等）的队列/调度/轮询开销一般使对客可承诺值**低于**"
        "本报告实测；如需那个数字，属另立测量，不冒充本工具结论（§7.10.4）。",
        "",
    ]
    for note in sl_notes:
        lines.append(f"> 注意：{note}")
    if sl_notes:
        lines.append("")
    return lines


def _bottleneck_lines(cap: StageStats) -> List[str]:
    """瓶颈判读：input/output 分解（膝点与平台两条结论路径共用）。"""
    if cap.input_tps is None or cap.output_tps is None or cap.input_tps <= 0:
        return []
    if cap.input_tps >= cap.output_tps:
        return [
            f"瓶颈判读：输入侧 {cap.input_tps:,.0f} tok/s ≥ 输出侧 {cap.output_tps:,.0f} tok/s，"
            "prefill 主导——扩容/调优优先看预填充批次与长上下文路径。"
        ]
    return [
        f"瓶颈判读：输出侧 {cap.output_tps:,.0f} tok/s > 输入侧 {cap.input_tps:,.0f} tok/s，"
        "decode 主导——扩容/调优优先看连续批处理宽度与 KV 容量。"
    ]


def _conclusion_lines(
    cfg: Config,
    stages: Sequence[StageStats],
    knee: Optional[StageStats],
    cap: Optional[StageStats],
    knee_notes: Sequence[str],
    plateau_peak: Optional[StageStats] = None,
    plateau_notes: Sequence[str] = (),
) -> Tuple[List[str], List[str]]:
    """结论区。返回 (结论行, 附加说明行)——SLA/SLO 与窗口承诺判定。

    knee=None 且平台判定成立（D23）时走平台口径：读数明确标注
    "非膝点、软退化平台估计"；平台判定也不成立时维持"未确定容量膝点"
    的诚实降级，并附平台判定的排除理由。
    """
    lines: List[str] = []
    extra: List[str] = []
    if plateau_peak is not None:
        ok_rps = plateau_peak.ok_rps
        sla_rpm = ok_rps * 60.0 if ok_rps is not None else None
        lines.append(
            "**范围内饱和读数（非膝点：软退化平台估计，D23）**：负载梯度内全程未出现 A1，"
            "服务不拒绝、只变慢（排队型饱和）。"
            + (
                f"SLA 口径读数 = {sla_rpm:,.1f} RPM（峰值 ok_rps {ok_rps:.2f} rps × 60 "
                f"@ 负载 {plateau_peak.concurrency}）。"
                if sla_rpm is not None
                else "有效完成速率不可用，无法给出 RPM 口径读数。"
            )
            + "该数字是梯度范围内的饱和平台估计：负载继续加码服务不会失败、只会更慢，"
            "读数不构成「接得住」意义上的 SLA。"
        )
        if plateau_peak.total_tps is not None:
            lines.append(
                f"吞吐构成：全量 {plateau_peak.total_tps:,.0f} tokens/s"
                + (
                    f"（输入 {plateau_peak.input_tps:,.0f} + 输出 {plateau_peak.output_tps:,.0f}）。"
                    if plateau_peak.input_tps is not None and plateau_peak.output_tps is not None
                    else "。"
                )
            )
        extra.extend(_bottleneck_lines(plateau_peak))
        a = cfg.analysis
        if a.window_requests > 0 and a.window_hours > 0 and sla_rpm is not None:
            required = a.window_requests / (a.window_hours * 60.0)
            line = required * a.window_safety_factor
            verdict = "达标" if sla_rpm >= line else "**未达标**"
            lines.append(
                f"窗口承诺判定（平台口径）：{verdict}——验收线 {line:,.1f} RPM，"
                f"平台读数 {sla_rpm:,.1f} RPM；判定随超时预算（timeout_s）与"
                "已测负载范围变化，不是膝点口径的承诺。"
            )
        elif a.window_requests > 0 or a.window_hours > 0:
            extra.append("窗口承诺只配了一半（window_requests / window_hours 需同时设置），本次不给达标判定。")
        else:
            extra.append("未设窗口承诺（analysis.window_requests / window_hours = 0）：本报告只陈述实测值。")
        return lines, extra

    if knee is None or cap is None:
        lines.append("**未确定容量膝点**：" + "；".join(list(knee_notes) + list(plateau_notes)))
        return lines, extra

    ok_rps = cap.ok_rps
    sla_rpm = ok_rps * 60.0 if ok_rps is not None else None
    slo_tps = cap.total_tps
    where = f"负载 {cap.concurrency}"
    if sla_rpm is not None:
        lines.append(
            f"**实测服务等级（A1 拐点 = 负载 {knee.concurrency}，容量读数 @ {where}）**："
            f"SLA = {sla_rpm:,.1f} RPM（有效完成速率 {ok_rps:.2f} rps × 60）"
            + (f"，SLO = {slo_tps:,.0f} tokens/s（全量 = 输入 {cap.input_tps:,.0f} + "
               f"输出 {cap.output_tps:,.0f}）" if slo_tps is not None else "")
            + "。膝点处提交速率≈完成速率（λ*≈μ*），最大可持续负载与饱和吞吐数值一致。"
        )
    else:
        lines.append(f"负载 {knee.concurrency} 为首个 A1 级，但有效完成速率不可用（无时间窗），"
                     "无法给出 RPM 口径服务等级。")

    # 坍塌提示：容量读数级不是膝点级本身时，说明为何取低压峰值
    if cap.concurrency != knee.concurrency:
        knee_rps = knee.ok_rps
        extra.append(
            f"容量读数取 {where}（膝点级 负载 {knee.concurrency} 的 ok_rps 已受过载回落："
            + (f"{knee_rps:.2f} vs 峰值 {ok_rps:.2f} rps" if knee_rps is not None else "不可用")
            + "）——硬拒绝型服务在膝点级即坍塌，取低压峰值是诚实口径。"
        )

    # 瓶颈判读：input/output 分解
    extra.extend(_bottleneck_lines(cap))

    # 窗口承诺判定（人工只给承诺，验收线由实测反推）
    a = cfg.analysis
    if a.window_requests > 0 and a.window_hours > 0 and sla_rpm is not None:
        required = a.window_requests / (a.window_hours * 60.0)
        line = required * a.window_safety_factor
        verdict = "达标" if sla_rpm >= line else "**未达标**"
        lines.append(
            f"窗口承诺判定：{verdict}——人工价值目标 {a.window_requests:,} 请求 / "
            f"{a.window_hours:g} 小时反推需求 μ ≥ {required:,.1f} RPM，"
            f"加安全余量 ×{a.window_safety_factor:g} 得验收线 {line:,.1f} RPM；"
            f"实测 SLA = {sla_rpm:,.1f} RPM"
            + (f"（余量 {sla_rpm - line:+,.1f} RPM）" if sla_rpm >= line
               else f"（缺口 {line - sla_rpm:,.1f} RPM）")
            + "。"
        )
        # TPS 侧预算核对（同一承诺的 token 视角）
        avg_out = cap.completion_tokens_avg
        io = _run_io_len(cfg)
        if slo_tps is not None and avg_out and io:
            budget_tps = a.window_requests * (io[0] + avg_out) / (a.window_hours * 3600.0)
            verdict_tps = "达标" if slo_tps >= budget_tps else "未达标"
            lines.append(
                f"token 预算核对：{a.window_requests:,} 请求 ×（{io[0]} + {avg_out:.0f}）"
                f"tokens / {a.window_hours:g}h ⇒ 需求 μ ≥ {budget_tps:,.0f} tokens/s；"
                f"实测 SLO = {slo_tps:,.0f} tokens/s（{verdict_tps}）。"
            )
        else:
            extra.append(
                "token 预算核对未执行：需要实测 SLO、实测平均输出长度与可定的 "
                "(L_in, L_out) 坐标（D22，混合档位不猜测）。"
            )
    elif a.window_requests > 0 or a.window_hours > 0:
        extra.append("窗口承诺只配了一半（window_requests / window_hours 需同时设置），本次不给达标判定。")
    else:
        extra.append("未设窗口承诺（analysis.window_requests / window_hours = 0）：本报告只陈述实测值。")
    return lines, extra


def _capacity_table(stages: Sequence[StageStats]) -> List[str]:
    ordered = sorted(stages, key=lambda s: s.concurrency)
    rows = []
    for s in ordered:
        rows.append([
            str(s.concurrency),
            str(s.n_total),
            num(s.ok_rps, 2),
            num(s.ok_rps * 60.0 if s.ok_rps is not None else None, 1),
            num(s.input_tps, 0),
            num(s.output_tps, 0),
            num(s.total_tps, 0),
            pct(s.a1 / s.n_total, 2) if s.n_total else "n/a",
            num(s.rps, 2),
        ])
    lines = [
        "| 负载水平 | 样本数 | ok_rps（有效） | RPM（ok×60） | input_tps | output_tps | total_tps | A1 占比 | rps（全量·含失败） |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    lines.append("")
    lines.append(
        "> `ok_rps`/`input_tps`/`output_tps`/`total_tps` 只计有效完成（容量结论的口径）；"
        "`rps` 为全量发出速率（**含失败**，全部快速失败时反而最高，仅作参照，"
        "不进入任何容量结论）。`total_tps = input_tps + output_tps`。"
    )
    return lines


def _knee_evidence_lines(
    stages: Sequence[StageStats],
    knee: Optional[StageStats],
    plateau_peak: Optional[StageStats] = None,
    plateau_notes: Sequence[str] = (),
) -> List[str]:
    ordered = sorted((s for s in stages if s.n_total > 0), key=lambda s: s.concurrency)
    lines: List[str] = []
    if knee is not None:
        lines.append(
            f"- **首个 A1 级（膝点）= 负载 {knee.concurrency}**：该级出现首个失败请求"
            f"（A1={knee.a1}/{knee.n_total}，{pct(knee.a1 / knee.n_total, 2)}），"
            "服务自该水平起接不住——这就是容量拐点，而非人工设定值。"
        )
    elif plateau_peak is not None:
        lines.append("- **平台判定（D23，软退化）**：" + " ".join(plateau_notes))
    # 平台佐证：连续两级有效完成速率增幅 < 5%
    flat: List[str] = []
    drop: List[str] = []
    for prev, cur in zip(ordered, ordered[1:]):
        if prev.ok_rps and cur.ok_rps and prev.ok_rps > 0:
            growth = (cur.ok_rps - prev.ok_rps) / prev.ok_rps
            if growth < _PLATEAU_GROWTH:
                item = f"{prev.concurrency}→{cur.concurrency}（{growth:+.1%}）"
                (flat if growth >= 0 else drop).append(item)
    if flat:
        lines.append("- 平台佐证·趋平（连续两级 ok_rps 增幅 ∈ [0, 5%)，吞吐饱和的典型形态）："
                     + "；".join(flat) + "。")
    if drop:
        lines.append("- 平台佐证·回落（连续两级 ok_rps 增幅 < 0，已过膝点的坍塌）："
                     + "；".join(drop) + "。")
    if knee is not None and not flat and not drop:
        lines.append("- 未见连续两级 ok_rps 增幅 < 5% 的平台：膝点由首个 A1 直接判定"
                     "（阶梯跨度可能跳过了平台区，可加密梯度复测）。")
    return lines


def _attribution_lines(stages: Sequence[StageStats]) -> List[str]:
    ordered = sorted((s for s in stages if s.n_total > 0), key=lambda s: s.concurrency)
    rows = []
    for s in ordered:
        if s.a1 == 0:
            continue
        timeout_n, reject_n, other_n = s.a1_families()
        kinds = "、".join(f"{k}×{c}" for k, c in sorted(s.fail_kinds.items(), key=lambda kv: -kv[1]))
        rows.append([
            str(s.concurrency), str(s.a1), str(timeout_n), str(reject_n), str(other_n), kinds or "-",
        ])
    if not rows:
        return ["- 本次运行没有任何 A1（失败请求）：梯度上限内服务未出现过载，无归因可陈。"]
    lines = [
        "| 负载水平 | A1 数 | 超时族 | 拒绝族 | 其他 | 失败子类型 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    lines.append("")
    lines.append(
        "> 家族口径（D17）：**拒绝族**（429/5xx/stream_error）说明服务端有界队列在显式"
        "拒绝；**超时族**（客户端超时）说明队列在无限吸收——过载只投影为迟到。"
        "治理动作因此不同：前者调队列容量/准入，后者调客户端超时与背压。"
    )
    return lines


def _advice_lines(cfg: Config, stages: Sequence[StageStats], knee: Optional[StageStats],
                  cap: Optional[StageStats], knee_notes: Sequence[str],
                  plateau_peak: Optional[StageStats] = None) -> List[str]:
    lines: List[str] = []
    if plateau_peak is not None:
        lines.append("- **软退化（排队型）服务**：过载只投影为变慢而不拒绝（D17 超时族形态）。"
                     "治理动作在入口侧：客户端超时与背压（在途上限、退避）、准入控制；"
                     "单纯加并发只会把容量数字变成排队时长。")
        lines.append("- **平台读数的外推纪律（D23/D25）**：读数只对已测负载范围与本次超时预算"
                     "（timeout_s）有效；再加大负载不会产生失败信号，只会拉长时延——"
                     "上线必须配合时延/水位监控，不能指望服务报错。")
        a = cfg.analysis
        if a.window_requests > 0 and a.window_hours > 0 and cap.ok_rps is not None:
            sla_rpm = cap.ok_rps * 60.0
            required = a.window_requests / (a.window_hours * 60.0) * a.window_safety_factor
            if sla_rpm < required:
                lines.append(
                    f"- **容量缺口 {required - sla_rpm:,.1f} RPM（平台口径）**：扩容"
                    "（副本/并行度）、缩短输入/输出长度、或与业务方重谈窗口承诺。"
                )
        lines.append("- **判据校准**：平台判定（末级边际 < 5%）是经验阈值；梯度太粗会伪造"
                     "「未饱和」（建议最小梯度档 ≥ 末级并发的 25%），判读存疑时加密梯度复测。")
        return lines
    if knee is None or cap is None:
        lines.append("- **提高梯度上限复测**：" + "；".join(knee_notes))
        return lines
    family = _dominant_family(knee)
    if family == "reject":
        lines.append("- **过载形态 = 拒绝族**：服务端有界队列显式拒绝。可提高队列容量/准入阈值，"
                     "或在入口侧做负载调参（限流、降级、批量合并）把峰值压到膝点以下。")
    elif family == "timeout":
        lines.append("- **过载形态 = 超时族**：队列无限吸收、过载投影为迟到。客户端超时与背压"
                     "（在途上限、退避）是唯一有效闸门；单纯加并发只会把失败变成慢失败。")
    a = cfg.analysis
    if a.window_requests > 0 and a.window_hours > 0 and cap.ok_rps is not None:
        sla_rpm = cap.ok_rps * 60.0
        required = a.window_requests / (a.window_hours * 60.0) * a.window_safety_factor
        if sla_rpm < required:
            lines.append(
                f"- **容量缺口 {required - sla_rpm:,.1f} RPM**：扩容（副本/并行度）、"
                "缩短输入长度、或与业务方重谈窗口承诺——三者必选其一，"
                "不要把未达标数字当对客承诺。"
            )
        else:
            lines.append(
                f"- **容量满足窗口承诺**（余量 {sla_rpm - required:+,.1f} RPM）："
                f"以膝点负载 {knee.concurrency} 作为生产上限参考，并对 ok_rps 做持续监控"
                "（低于预测 μ×安全系数即早警，见容量面使用场景四）。"
            )
    lines.append("- **复测纪律**：容量结论绑定本次边界（base_url）与输入长度；换引擎/模型/"
                 "长度档后重测，容量面按长度覆盖更新，不要把旧数字套新边界。")
    return lines


def _validate_stage_consistency(
    cfg: Config, stage: StageStats, rows: List[Dict[str, Any]]
) -> List[str]:
    """两道数据质量闸门（D24，仅吞吐路径调用；只警告不阻断）。

    - Little's Law（闭环）：rps × 平均E2E ≈ 施压并发。偏差超阈说明该级
      计时窗被污染（疑似冷启动/预热泄漏/外部干扰）——负载更大不可能
      "更快"，闭环下吞吐与时延的乘积必须守恒。
    - 时延恒等式：E2E ≈ TTFT + TPOT×(输出-1)（逐请求残差均值）。偏差
      超阈说明时延构成不自洽（字段口径错位或数据损坏）。

    两闸门互相独立、可互为佐证；输入全部来自请求明细现场计算，不落库、
    不改变任何容量数字（结果只进本报告诊断行，R4）。前置排除：think_time
    改变闭环守恒式（E2E 需加思考时间）与存在 aborted（窗口计入放弃请求
    但无 E2E）时，Little's Law 不适用，跳过而非误报。
    """
    if not rows:
        return [f"负载 {stage.concurrency}：请求明细缺失，无法执行数据质量校验（D24）。"]
    measured = [r for r in rows if r.get("status") != "aborted"]
    if len(measured) < _MIN_DIAG_SAMPLES:
        return []
    diags: List[str] = []
    if (
        cfg.load.mode == "closed"
        and cfg.load.think_time_s <= 0
        and stage.n_aborted == 0
        and stage.rps is not None
        and stage.concurrency > 0
    ):
        e2es = [r["e2e_ms"] for r in measured if r.get("e2e_ms") is not None]
        if len(e2es) >= _MIN_DIAG_SAMPLES:
            lhs = stage.rps * (sum(e2es) / len(e2es)) / 1000.0
            dev = abs(lhs - stage.concurrency) / stage.concurrency
            if dev > _CONSISTENCY_TOL:
                diags.append(
                    f"负载 {stage.concurrency}：Little's Law 一致性异常——"
                    f"rps×平均E2E = {lhs:.2f}，与施压并发 {stage.concurrency} 偏差 "
                    f"{dev:.0%}（>{_CONSISTENCY_TOL:.0%}），疑似冷启动/计时污染（D24）。"
                )
    resid: List[float] = []
    for r in measured:
        e2e = r.get("e2e_ms")
        ttft = r.get("ttft_ms")
        tpot = r.get("tpot_ms")
        out = r.get("completion_tokens")
        if e2e is None or ttft is None or tpot is None or not out or out < 2 or e2e <= 0:
            continue
        resid.append(abs(e2e - ttft - tpot * (out - 1)) / e2e)
    if len(resid) >= _MIN_DIAG_SAMPLES:
        mean_resid = sum(resid) / len(resid)
        if mean_resid > _CONSISTENCY_TOL:
            diags.append(
                f"负载 {stage.concurrency}：时延恒等式偏差均值 {mean_resid:.1%}"
                f"（>{_CONSISTENCY_TOL:.0%}）——E2E 与 TTFT+TPOT×(输出-1) 不自洽，"
                "时延构成异常（D24）。"
            )
    return diags


def render_report_batch(
    cfg: Config,
    store: RecordStore,
    stages: Sequence[StageStats],
    finding: CapacityFinding,
    meta: Dict[str, str],
) -> Tuple[str, Optional[str], List[str]]:
    """渲染吞吐报告并汇写容量面。返回 (报告文本, 容量面路径或 None, 容量面说明)。"""
    sl_finding = metrics.find_service_levels(stages, finding, cfg.watermark, a1_onset=True)
    knee, knee_notes = find_knee_stage(stages)
    # knee 与 sl_finding.availability_boundary 应一致（同判据）；不一致时以
    # find_service_levels 为准并在结论中说明，避免两处口径漂移
    if knee is not None and sl_finding.availability_boundary is not None \
            and knee.concurrency != sl_finding.availability_boundary:
        knee_notes = knee_notes + [
            f"内部口径不一致：find_knee_stage={knee.concurrency} vs "
            f"find_service_levels={sl_finding.availability_boundary}（应相同判据）"
        ]
    cap = capacity_stage(stages, knee) if knee is not None else None

    # 软退化平台判定（D23）：仅在膝点未触达时启用第二读数口径
    plateau_peak, plateau_notes = (None, [])
    if knee is None:
        plateau_peak, plateau_notes = plateau_reading(stages)
    if plateau_peak is not None:
        cap = plateau_peak

    source = meta.get("db_path", "")
    point = None
    if knee is not None and cap is not None:
        point = knee_point(cfg, knee, cap, source=source)
    elif plateau_peak is not None:
        point = plateau_point(cfg, plateau_peak, source=source)
    skip_reason = ""
    if (knee is not None or plateau_peak is not None) and point is None:
        skip_reason = (
            "本次读数缺少入容量面的坐标（(L_in, L_out) 组合不可定或有效速率缺失），仅出报告。"
        )
    surface_path, surface_notes = merge_surface_point(cfg, point, reason=skip_reason)

    # 数据质量诊断（D24）：仅吞吐路径（R4）；从请求明细现场计算，
    # 不写库、不改变任何容量数字，只进本报告诊断行
    diagnostics: List[str] = []
    for s in stages:
        if s.n_total <= 0:
            continue
        diagnostics.extend(_validate_stage_consistency(cfg, s, store.stage_rows(s.stage_id)))

    lines: List[str] = []
    lines.extend(_header_lines(cfg, meta, list(knee_notes) + list(sl_finding.notes)))
    if surface_path:
        lines.append(f"- 容量面：`{surface_path}`（本次读数已并入，replace-per-(L_in, L_out)）")
    else:
        lines.append(f"- 容量面：未写入——{surface_notes[-1] if surface_notes else '未知原因'}")
    lines.append("")

    lines.append("## 一、结论")
    lines.append("")
    conclusions, extra = _conclusion_lines(
        cfg, stages, knee, cap, knee_notes,
        plateau_peak=plateau_peak, plateau_notes=plateau_notes,
    )
    for item in conclusions:
        lines.append(f"- {item}")
    lines.append("")
    for item in extra:
        lines.append(f"- {item}")
    if extra:
        lines.append("")

    lines.append("## 二、容量表（逐级）")
    lines.append("")
    lines.extend(_capacity_table(stages))
    if diagnostics:
        lines.append("### 数据质量诊断（D24，仅警告不阻断）")
        lines.append("")
        for d in diagnostics:
            lines.append(f"- {d}")
        lines.append("")

    lines.append("## 三、膝点证据")
    lines.append("")
    lines.extend(_knee_evidence_lines(stages, knee, plateau_peak, plateau_notes))
    lines.append("")

    lines.append("## 四、A1 归因（过载形态）")
    lines.append("")
    lines.extend(_attribution_lines(stages))
    lines.append("")

    lines.append("## 五、建议")
    lines.append("")
    lines.extend(_advice_lines(cfg, stages, knee, cap, knee_notes, plateau_peak=plateau_peak))
    lines.append("")

    lines.append("## 六、方法与口径")
    lines.append("")
    lines.append("1. 一次采集两侧消费：时延与吞吐指标在同一次运行中全量落库（§7.10.1）；"
                 "本报告只消费吞吐侧面，A3 等质量读数见在线报告（report.kinds 可同时产出）。")
    lines.append("2. 请求三分类照常执行（A1=失败，A2=成功且达标，A3=成功不达标）；"
                 "容量口径下 A3 不进入结论，比率 SLA=(A2+A3)/N 与 SLO=A2/N 退化为同一数值，"
                 "服务等级改由吞吐单位承载（§7.10.3）。")
    lines.append("3. 有效完成速率 ok_rps = n_ok/时间窗（只计成功请求）；"
                 "input_tps/output_tps 按成功请求的双 token 计数；total_tps=input+output 读取时派生。")
    lines.append("4. 时间窗按请求自身墙钟计算（首个发出到最后完成，含 drain 尾巴），"
                 "避免用名义稳压窗口造成系统性低估。")
    lines.append("5. **容量读数**（SLA/SLO）取膝点级及低压各级 ok_rps 的峰值所在级："
                 "排队型饱和时膝点级即峰值（二者一致）；硬拒绝型服务在膝点级即坍塌"
                 "（该级大部分请求被拒），取低压峰值并在结论中显式说明。")
    lines.append("6. 容量面（throughput_model.npz）由本报告报告期汇写：只存事实点与 meta，"
                 "坐标为 (输入, 输出) 长度组合（D22）；查询按双线性插值，任一坐标超范围 "
                 "clamp 并标 extrapolated，网格角点未实测时回退最近点并显式说明（§7.10.5）。")
    lines.append("7. **膝点 = 服务拒绝 或 客户放弃**（D25）：A1 拒绝族是服务端有界队列显式拒绝；"
                 "超时族是队列无限吸收、过载投影为迟到——膝点位置由客户端 timeout_s 框定"
                 "（见报告头超时预算行），不同预算下的容量读数不可比。")
    lines.append("8. 软退化（全程无 A1）服务以平台期第二读数收场（D23）：末级边际增幅 < 5% 且"
                 "峰值不低于末级 5% 时给「范围内饱和读数」并入容量面（origin=plateau）；"
                 "该读数非膝点，只对已测负载范围与超时预算有效。")
    lines.append("9. 数据质量闸门（D24，仅吞吐路径）：闭环下 rps×平均E2E≈并发（Little's Law）"
                 "与 E2E≈TTFT+TPOT×(输出-1)（时延恒等式）两道独立校验，偏差 >15% 记入第二节"
                 "诊断行；只警告不阻断，不改变任何容量数字。")
    lines.append("")
    return "\n".join(lines), surface_path, surface_notes


def write_report_batch(
    cfg: Config, store: RecordStore, stages: Sequence[StageStats],
    finding: CapacityFinding, meta: Dict[str, str], output_path: str,
) -> Tuple[str, Optional[str]]:
    """渲染并落盘吞吐报告；返回 (报告路径, 容量面路径或 None)。"""
    text, surface_path, _notes = render_report_batch(cfg, store, stages, finding, meta)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path), surface_path
