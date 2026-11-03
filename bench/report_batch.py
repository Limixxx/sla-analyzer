"""吞吐（容量）报告：``report_batch.py``（docs/07 §7.10.6）。

与 ``report.py``（在线/延迟报告）消费**同一批** StageStats 事实，各自
产出报告与工件——测量管道不做场景裁剪，差异只在分析选择
（``report.kinds``，§7.10.1）。本模块因此只做"容量视角的呈现"：

- 服务等级由吞吐单位承载：SLA = 膝点 ``ok_rps``×60（RPM，可用性/容量），
  SLO = 膝点 ``input_tps+output_tps``（全量 TPS，目标/工作量），同在
  A1 拐点（``find_service_levels(..., a1_onset=True)``，§7.10.3）；
- 容量分析不消费质量维度：A3 照常被采集与计数，但本报告不引用；
- 报告期汇写容量面 ``throughput_model.npz``（派生量不落库，C3），
  使 ``--report-only`` 可从旧库重建出同一结论。

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


def _run_input_len(cfg: Config, knee: StageStats) -> Optional[int]:
    """本次运行的输入长度坐标：仅单长度配置可取（配置值本身）。

    容量面按长度索引且每个点应为"固定长度运行"的膝点（§7.10.5）；
    混合长度运行的膝点不对应单一 L，用均值近似会让插值面失真，
    因此不猜坐标——返回 None，由调用方跳过容量面写入并说明原因。
    """
    targets = list(cfg.requests.prompt_token_targets)
    if len(targets) == 1:
        return int(targets[0])
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
    并发/在途数），n_samples 取容量读数级样本数。膝点（开始出现失败的
    负载水平）见报告第三节，不入容量面——反问题问的是"扛得住要多少"。
    """
    input_len = _run_input_len(cfg, knee)
    if input_len is None or cap.ok_rps is None:
        return None
    return ThroughputPoint(
        input_len=input_len,
        mu_rpm=cap.ok_rps * 60.0,
        mu_tps=cap.total_tps,
        input_tps=cap.input_tps,
        output_tps=cap.output_tps,
        req_conc=cap.concurrency,
        n_samples=cap.n_total,
        a1_family=_dominant_family(knee),
        source=source,
    )


# ---------------------------------------------------------------------- #
# 容量面汇写（报告期；--report-only 可重建）
# ---------------------------------------------------------------------- #
def merge_surface_point(
    cfg: Config, point: Optional[ThroughputPoint], reason: str = ""
) -> Tuple[Optional[str], List[str]]:
    """把本次膝点并入容量面文件（replace-per-L）。返回 (路径或 None, 说明)。

    不同边界（base_url）的面对比用两个文件、不混装：既有文件记录的是
    别的边界时拒绝合并并原样保留，由报告与 stdout 说明。
    """
    path = cfg.report.throughput_model_path
    notes: List[str] = []
    if point is None:
        return None, notes + [reason or "本次运行未产生可入容量面的膝点事实。"]
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
            return path, notes + [f"本次膝点已并入既有容量面（按长度覆盖更新）：{path}"]
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
    single = list(cfg.requests.prompt_token_targets)
    len_txt = (
        f"固定输入长度 {single[0]} tokens（单长度档）"
        if len(single) == 1
        else f"混合长度 {single}（容量面按膝点均值取坐标）"
    )
    lines = [
        "# 推理服务容量（吞吐）评估报告",
        "",
        f"- 测量边界：`{cfg.service.base_url}`（`{cfg.service.path}`，模型 `{cfg.service.model}`）",
        f"- 运行时刻：{started} ~ {now}",
        f"- 配置快照：模式 {cfg.load.mode}，负载阶梯 {cfg.load.concurrency_steps}，"
        f"每级 {cfg.load.stage_duration_s:g}s（预热 {cfg.load.warmup_s:g}s）；{len_txt}；"
        f"门限 {cfg.slo.target_mode}；Phase 0 基线 {'开' if cfg.load.baseline_enabled else '关'}",
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


def _conclusion_lines(
    cfg: Config,
    stages: Sequence[StageStats],
    knee: Optional[StageStats],
    cap: Optional[StageStats],
    knee_notes: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """结论区。返回 (结论行, 附加说明行)——SLA/SLO 与窗口承诺判定。"""
    lines: List[str] = []
    extra: List[str] = []
    if knee is None or cap is None:
        lines.append("**未确定容量膝点**：" + "；".join(knee_notes))
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
    if cap.input_tps is not None and cap.output_tps is not None and cap.input_tps > 0:
        if cap.input_tps >= cap.output_tps:
            extra.append(
                f"瓶颈判读：输入侧 {cap.input_tps:,.0f} tok/s ≥ 输出侧 {cap.output_tps:,.0f} tok/s，"
                "prefill 主导——扩容/调优优先看预填充批次与长上下文路径。"
            )
        else:
            extra.append(
                f"瓶颈判读：输出侧 {cap.output_tps:,.0f} tok/s > 输入侧 {cap.input_tps:,.0f} tok/s，"
                "decode 主导——扩容/调优优先看连续批处理宽度与 KV 容量。"
            )

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
        input_len = _run_input_len(cfg, knee)
        if slo_tps is not None and avg_out and input_len:
            budget_tps = a.window_requests * (input_len + avg_out) / (a.window_hours * 3600.0)
            verdict_tps = "达标" if slo_tps >= budget_tps else "未达标"
            lines.append(
                f"token 预算核对：{a.window_requests:,} 请求 ×（{input_len} + {avg_out:.0f}）"
                f"tokens / {a.window_hours:g}h ⇒ 需求 μ ≥ {budget_tps:,.0f} tokens/s；"
                f"实测 SLO = {slo_tps:,.0f} tokens/s（{verdict_tps}）。"
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


def _knee_evidence_lines(stages: Sequence[StageStats], knee: Optional[StageStats]) -> List[str]:
    ordered = sorted((s for s in stages if s.n_total > 0), key=lambda s: s.concurrency)
    lines: List[str] = []
    if knee is not None:
        lines.append(
            f"- **首个 A1 级（膝点）= 负载 {knee.concurrency}**：该级出现首个失败请求"
            f"（A1={knee.a1}/{knee.n_total}，{pct(knee.a1 / knee.n_total, 2)}），"
            "服务自该水平起接不住——这就是容量拐点，而非人工设定值。"
        )
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
                  cap: Optional[StageStats], knee_notes: Sequence[str]) -> List[str]:
    lines: List[str] = []
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

    source = meta.get("db_path", "")
    point = knee_point(cfg, knee, cap, source=source) if (knee is not None and cap is not None) else None
    skip_reason = ""
    if knee is not None and point is None:
        skip_reason = (
            "本次膝点缺少入容量面的坐标（输入长度不可定或有效速率缺失），仅出报告。"
        )
    surface_path, surface_notes = merge_surface_point(cfg, point, reason=skip_reason)

    lines: List[str] = []
    lines.extend(_header_lines(cfg, meta, list(knee_notes) + list(sl_finding.notes)))
    if surface_path:
        lines.append(f"- 容量面：`{surface_path}`（本次膝点已并入，replace-per-L）")
    else:
        lines.append(f"- 容量面：未写入——{surface_notes[-1] if surface_notes else '未知原因'}")
    lines.append("")

    lines.append("## 一、结论")
    lines.append("")
    conclusions, extra = _conclusion_lines(cfg, stages, knee, cap, knee_notes)
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
    lines.append("")

    lines.append("## 三、膝点证据")
    lines.append("")
    lines.extend(_knee_evidence_lines(stages, knee))
    lines.append("")

    lines.append("## 四、A1 归因（过载形态）")
    lines.append("")
    lines.extend(_attribution_lines(stages))
    lines.append("")

    lines.append("## 五、建议")
    lines.append("")
    lines.extend(_advice_lines(cfg, stages, knee, cap, knee_notes))
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
                 "查询按长度线性插值，超范围 clamp 并标 extrapolated（§7.10.5）。")
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
