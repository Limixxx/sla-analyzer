"""评估报告生成：由分级统计产出带明确结论的 Markdown 报告。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import metrics
from . import qos as qos_mod
from .baseline import BaselineResult, Thresholds, bucket_label, bucket_of
from .config import Config
from .metrics import CapacityFinding, StageStats, WatermarkFinding, split_fail_kinds
from .qos import (
    BoundaryScan,
    MixPoint,
    QoSResult,
    StagePrediction,
    apply_limits,
    build_dataset,
    mix_of,
    scan_boundaries,
    simulate_stage,
)
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
    dur_txt = f"配置 {baseline.duration_s:.0f}s"
    actual = baseline.duration_actual_s
    if actual is not None and actual > baseline.duration_s * 1.05:
        dur_txt += (
            f"，为补齐各长度桶样本量自适应延长至 {actual:.0f}s"
            f"（硬上限 load.max_stage_duration_s）"
        )
    lines.append(
        f"Phase 0 在并发 {baseline.concurrency} 的低压下测量无压力时延（{dur_txt}）。"
        "基线只统计分位数与失败数，不做达标判定——基线是尺子的来源，再拿它判定会形成循环。"
    )
    lines.append("")

    rows = []
    for i in sorted(index):
        b = index[i]
        ttft_limit = thresholds.ttft_limit(i) if thresholds else None
        tpot_limit = thresholds.tpot_limit(i) if thresholds else None
        if thresholds is None:
            src = "n/a"
        elif thresholds.mode == "absolute":
            src = "absolute 配置值"
        elif i in thresholds.fallback_buckets:
            src = "回退 absolute"
        else:
            src = "基线×系数"
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
# 服务质量预测模型（MLP）
# ---------------------------------------------------------------------- #
def _qos_bucket_limits(stages: Sequence[StageStats]) -> Dict[int, Tuple[float, float]]:
    """从任一级的按桶统计量取各长度桶的生效门限（TTFT/TPOT，毫秒）。"""
    for s in stages:
        if not s.bucket_stats:
            continue
        limits: Dict[int, Tuple[float, float]] = {}
        for b, entry in s.bucket_stats.items():
            ttft = entry.get("ttft_limit_ms")
            tpot = entry.get("tpot_limit_ms")
            if ttft and tpot:
                limits[int(b)] = (float(ttft), float(tpot))
        if limits:
            return limits
    return {}


def _qos_default_limits(
    cfg: Config, limits: Dict[int, Tuple[float, float]]
) -> Tuple[float, float]:
    if limits:
        return limits[min(limits)]
    return cfg.slo.ttft_target_s * 1000.0, cfg.slo.tpot_target_ms


def _prepare_qos(
    cfg: Config,
    store: RecordStore,
    stages: Sequence[StageStats],
    qos: Optional[QoSResult],
) -> None:
    """预计算预测章节所需的 mix / 分级预测 / 边界细扫（结论区也要引用 scan）。"""
    if qos is None or qos.model is None or not qos.report.usable:
        return
    samples = build_dataset(store, cfg.qos_model)
    if not samples:
        return
    buckets = cfg.requests.length_buckets
    limits = _qos_bucket_limits(stages)
    default_ttft, default_tpot = _qos_default_limits(cfg, limits)

    ref_mix = mix_of(samples, buckets, bucket_of)
    apply_limits(ref_mix, limits)
    qos.reference_mix = ref_mix

    by_stage: Dict[int, List] = {}
    for s in samples:
        by_stage.setdefault(s.stage_id, []).append(s)

    qos.stage_predictions = {}
    for st in stages:
        mix = mix_of(by_stage.get(st.stage_id, []), buckets, bucket_of)
        apply_limits(mix, limits)
        if not mix:
            continue
        qos.stage_mix[st.stage_id] = mix
        qos.stage_predictions[st.stage_id] = simulate_stage(
            qos.model, st.concurrency, mix, default_ttft, default_tpot
        )

    ordered = sorted(stages, key=lambda x: x.concurrency)
    if ordered:
        qos.scan = scan_boundaries(
            qos.model, ref_mix, default_ttft, default_tpot,
            cfg.analysis.slo_target, cfg.watermark,
            ordered[0].concurrency, ordered[-1].concurrency,
        )


def _render_qos_section(
    cfg: Config,
    store: RecordStore,
    stages: Sequence[StageStats],
    qos: Optional[QoSResult],
) -> List[str]:
    """第九节：预测模型卡片、预测 vs 实测对照、预测曲线与水位边界细扫。"""
    lines: List[str] = []
    lines.append("## 九、服务质量预测模型（MLP）")
    lines.append("")

    if qos is None or qos.model is None:
        reason = qos.report.skip_reason if qos is not None else "未启用"
        lines.append(f"本次未生成预测模型：{reason or '未知原因'}。")
        lines.append("")
        lines.append("> 训练需要足够多的 measured 成功样本（`qos_model.min_samples`，默认 200）"
                     "与至少 2 个压力级；增大 `load.stage_duration_s` 或补充压力级后重测即可。")
        lines.append("")
        return lines

    if qos.load_warning:
        lines.append(f"> **警告**：{qos.load_warning}。")
        lines.append("")

    rep = qos.report
    lines.append("> 方法论见 [docs/06-qos-model.md](docs/06-qos-model.md)：输入（并发, prompt 长度, "
                 "输出长度）→ 预测 TTFT/TPOT 分位数与 SLA/SLO 通过率。"
                 "模型绑定当次服务配置，换引擎/模型/并行度必须重训。")
    lines.append("")

    # ---- 9.1 训练与验证 ----
    lines.append("### 9.1 训练与验证")
    lines.append("")
    src_txt = {"trained": "本次训练", "loaded": "加载已有模型"}.get(qos.source, qos.source)
    lines.append(f"- 模型来源：{src_txt}"
                 + (f"（`{qos.model_path}`）" if qos.model_path else ""))
    lines.append(f"- 训练样本：{rep.n_samples} 个 measured 成功请求 / {rep.n_stages} 个压力级"
                 "（A3 保留——膝点区只有 A3；warmup / aborted / A1 剔除）")
    lines.append(f"- 网络结构：{len(cfg.qos_model.hidden) + 2} 层（输入 3 → "
                 f"隐藏 {'→'.join(str(h) for h in cfg.qos_model.hidden)} → 输出 2），"
                 f"log 目标 + 标准化，全批量 Adam")
    lines.append(f"- 训练时长：{rep.seconds:.1f}s"
                 + ("（模型文件记录，非本次训练）" if qos.source == "loaded" else "")
                 + (f"（被 max_train_seconds={cfg.qos_model.max_train_seconds:.0f}s 截断）"
                    if rep.stopped_by_deadline else ""))
    lines.append("")
    if rep.cross_validated:
        lines.append("| 目标 | 验证 R²（折外） | MAE | MAPE |")
        lines.append("| --- | --- | --- | --- |")
        lines.append(f"| TTFT | {rep.r2_ttft:.3f} | {rep.mae_ttft_ms:.1f} ms | {rep.mape_ttft:.1%} |")
        lines.append(f"| TPOT | {rep.r2_tpot:.3f} | {rep.mae_tpot_ms:.1f} ms | {rep.mape_tpot:.1%} |")
        lines.append("")
        lines.append(f"- 交叉验证：按 stage 分组 {len(rep.folds)} 折"
                     f"（防同级样本泄漏）；终训轮数 {rep.best_epochs_median}"
                     f"（各折最优轮数中位数）")
        if rep.folds:
            fold_txt = "；".join(
                f"折{f.fold + 1}@{','.join(str(c) for c in f.val_stage_ids)}({f.best_epoch}ep)"
                for f in rep.folds
            )
            lines.append(f"- 折明细（验证级@stage id）：{fold_txt}")
    else:
        lines.append("> **无有效交叉验证**（压力级少于 2 个）：上述指标为训练集指标，"
                     "偏乐观，不能作为预测可信度依据。")
    for w in rep.warnings:
        lines.append(f"- 注意：{w}")
    lines.append("")

    if not rep.usable:
        lines.append(f"**模型不具备预测价值**（{rep.skip_reason}）。"
                     "本节以下仅渲染数据口径，不给预测结论。")
        lines.append("")
        return lines

    # ---- 数据与 mix ----
    if not qos.reference_mix:
        lines.append("> 数据库中没有可用样本，无法生成预测对照。")
        lines.append("")
        return lines
    buckets = cfg.requests.length_buckets
    limits = _qos_bucket_limits(stages)
    default_ttft, default_tpot = _qos_default_limits(cfg, limits)
    ref_mix = qos.reference_mix

    # ---- 9.2 预测 vs 实测 ----
    lines.append("### 9.2 预测 vs 实测（已测并发）")
    lines.append("")
    rows = []
    for st in sorted(stages, key=lambda x: x.concurrency):
        if st.n_total == 0:
            continue
        sp = qos.stage_predictions.get(st.stage_id)
        if sp is None:
            continue
        dev = abs(sp.pred_slo - st.slo) / max(st.slo, 1e-9)
        flag = " ⚠" if dev > 0.25 else ""
        rows.append([
            str(st.concurrency),
            str(st.n_total),
            _ms(st.ttft_p50_ms), f"{sp.pred_ttft_p50_ms:.0f}",
            _ms(st.ttft_p99_ms), f"{sp.pred_ttft_p99_ms:.0f}",
            _ms(st.tpot_p50_ms), f"{sp.pred_tpot_p50_ms:.1f}",
            _ms(st.tpot_p99_ms), f"{sp.pred_tpot_p99_ms:.1f}",
            _pct(st.slo, 2), _pct(sp.pred_slo, 2) + flag,
        ])
    if rows:
        lines.append(_table(
            ["并发", "样本", "TTFT p50 实测", "TTFT p50 预测", "TTFT p99 实测", "TTFT p99 预测",
             "TPOT p50 实测", "TPOT p50 预测", "TPOT p99 实测", "TPOT p99 预测",
             "SLO 实测", "SLO 预测"],
            rows,
        ))
        lines.append("")
        lines.append("> 预测为该并发实际流量构成下的桶加权值；分位数由折外残差 ECDF 校准。")
        lines.append("> **口径说明**：实测 SLO 的分母含 A1（被拒绝或客户端超时的请求拉低实测值），"
                     "而预测只回答\"成功请求的质量\"——两者在高并发过载区必然分叉，"
                     "这是语义差异而非模型误差；⚠ 标记仅提示偏离幅度，"
                     "低并发区的偏离才真正反映模型精度。")
    else:
        lines.append("> 没有带样本的压力级，跳过对照。")
    lines.append("")

    # ---- 9.3 参考流量下的预测曲线 ----
    ordered = sorted(stages, key=lambda x: x.concurrency)
    if not ordered:
        return lines
    scan = qos.scan
    if scan is None:
        # _prepare_qos 未覆盖（如直接调用渲染）时兜底重算
        scan = scan_boundaries(
            qos.model, ref_mix, default_ttft, default_tpot,
            cfg.analysis.slo_target, cfg.watermark,
            ordered[0].concurrency, ordered[-1].concurrency,
        )
        qos.scan = scan

    lines.append("### 9.3 参考流量下的预测曲线")
    lines.append("")
    mix_txt = "；".join(
        f"{bucket_label(m.bucket, buckets)}({m.prompt_tokens:.0f}/{m.completion_tokens:.0f}) "
        f"{m.weight:.0%}"
        for m in ref_mix
    )
    lines.append(f"- 参考流量构成（本次实测聚合并按桶均值代表）：{mix_txt}")
    lines.append(f"- 扫描范围：并发 {scan.conc_min} ~ {scan.conc_max}（训练包络内，步进 1）")
    lines.append("")
    rows = []
    for c in scan.grid:
        sp = simulate_stage(qos.model, c, ref_mix, default_ttft, default_tpot)
        note = "（外推）" if sp.extrapolated else ""
        rows.append([
            str(c) + note,
            f"{sp.pred_ttft_p50_ms:.0f}", f"{sp.pred_ttft_p99_ms:.0f}",
            f"{sp.pred_tpot_p50_ms:.1f}", f"{sp.pred_tpot_p99_ms:.1f}",
            _pct(sp.pred_slo, 2),
            f"{1 - sp.pred_slo:.1%}",
        ])
    lines.append(_table(
        ["并发", "TTFT p50 (ms)", "TTFT p99 (ms)", "TPOT p50 (ms)", "TPOT p99 (ms)",
         "预测 SLO 通过率", "预测 A3 占比"],
        rows,
    ))
    lines.append("")

    # ---- 水位边界细扫 ----
    lines.append("**水位边界细扫**（与第四节的实测边界互相印证）：")
    lines.append("")
    soft_txt = f"并发 {scan.soft_boundary}" if scan.soft_boundary is not None else "未触达"
    hard_txt = f"并发 {scan.hard_boundary}" if scan.hard_boundary is not None else "未触达"
    cap_txt = (
        f"并发 {scan.quality_capacity}"
        if scan.quality_capacity is not None else "包络内未达到"
    )
    measured_soft = next(
        (s.concurrency for s in ordered if s.watermark_state in ("BUSY", "OVERLOAD")), None
    )
    measured_hard = next(
        (s.concurrency for s in ordered if s.watermark_state == "OVERLOAD"), None
    )
    measured_collapse = next(
        (
            s.concurrency
            for s in ordered
            if s.n_total > 0 and s.a3 / s.n_total >= cfg.watermark.hard_slo_violation_ratio
        ),
        None,
    )
    lines.append(f"- 预测软水位（A3 占比 ≥ {cfg.watermark.soft_slo_violation_ratio:.0%}）："
                 f"{soft_txt}（实测 {measured_soft if measured_soft else '未触达'}）")
    collapse_txt = (
        f"并发 {measured_collapse}" if measured_collapse is not None else "未触达"
    )
    hard_measured_txt = (
        f"；A1 失败型硬水位实测 {measured_hard}"
        if measured_hard is not None else ""
    )
    lines.append(f"- 预测硬水位-质量崩塌（A3 占比 ≥ {cfg.watermark.hard_slo_violation_ratio:.0%}）："
                 f"{hard_txt}（实测质量崩塌 {collapse_txt}{hard_measured_txt}）")
    lines.append(f"- 预测质量容量（SLO ≥ {cfg.analysis.slo_target:.0%} 连续前缀，"
                 f"**仅质量口径**）：{cap_txt}"
                 "（实测容量还要求 SLA 达标，含失败（拒绝/超时）影响，两者在过载区分叉属预期）")
    lines.append("- A1 驱动的硬水位（失败率，含拒绝与客户端超时）**不在模型能力内**："
                 "模型只预测成功请求的质量，不预测失败概率，该边界以实测为准。")
    lines.append("")

    # ---- 9.4 模型文件 ----
    lines.append("### 9.4 模型文件与使用")
    lines.append("")
    lines.append(f"- 文件：`{qos.model_path}`（.npz：ZIP + .npy，"
                 "可被 numpy 生态 `np.load()` 直接消费）")
    lines.append("- 单点预测（无需 numpy）：")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python bench.py --qos-model {qos.model_path} \\")
    lines.append('    --qos-predict "concurrency=16,prompt_tokens=4096,completion_tokens=512"')
    lines.append("```")
    lines.append("")
    lines.append("- 包络："
                 f"并发 {qos.model.feat_min[0]:.0f}~{qos.model.feat_max[0]:.0f}，"
                 f"prompt {qos.model.feat_min[1]:.0f}~{qos.model.feat_max[1]:.0f}，"
                 f"输出 {qos.model.feat_min[2]:.0f}~{qos.model.feat_max[2]:.0f}；"
                 "超出包络的预测一律标记外推且不参与结论。")
    lines.append("")
    return lines

# ---------------------------------------------------------------------- #
# 结论
# ---------------------------------------------------------------------- #
def build_conclusions(
    cfg: Config,
    stages: Sequence[StageStats],
    finding: CapacityFinding,
    watermark: Optional[WatermarkFinding] = None,
    qos: Optional[QoSResult] = None,
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

    # ---- 4) 水位边界（服务端水位的黑盒投影，见 docs/05-watermark.md） ----
    if watermark is not None and watermark.enabled:
        parts = []
        if watermark.soft_watermark_concurrency is not None:
            parts.append(
                f"软水位投影（BUSY，排队预警线）自并发 "
                f"{watermark.soft_watermark_concurrency} 起"
            )
        else:
            parts.append("已测梯度内未触达软水位投影（BUSY）")
        if watermark.hard_watermark_concurrency is not None:
            parts.append(
                f"硬水位投影（OVERLOAD，熔断线）自并发 "
                f"{watermark.hard_watermark_concurrency} 起"
            )
        else:
            parts.append("已测梯度内未触达硬水位投影（OVERLOAD，无 A1 失败/无质量崩塌）")
        if watermark.sla_watermark_concurrency is not None:
            parts.append(
                f"SLA 水位（业务目标口径）= 并发 {watermark.sla_watermark_concurrency}"
                "（与质量容量一致）"
            )
        lines.append("水位边界：" + "；".join(parts) + "。")

    # ---- 5) 过载形态（按实际数据描述，不用模板硬编码） ----
    if finding.knee_concurrency is not None:
        knee = by_conc.get(finding.knee_concurrency)
        if knee is not None:
            if knee.a1 > 0:
                # A1 的家族构成决定"过载长什么样"：拒绝族 ⇒ 服务端明确回绝；
                # 超时族 ⇒ 服务不拒绝但预算内完不成（Router 无限队列的投影）。
                timeout_n, reject_n, other_n = knee.a1_families()
                if timeout_n > 0 and timeout_n >= reject_n:
                    shape = (
                        f"出现失败请求（A1={knee.a1}，其中客户端超时 {timeout_n} 条），"
                        "Router 不拒绝但请求无法在预算内完成"
                        "（疑似无限队列/排队耗尽，过载形态为排队型）"
                    )
                elif reject_n > 0:
                    shape = (
                        f"出现失败请求（A1={knee.a1}，其中服务端拒绝 {reject_n} 条），"
                        "Router 进入过载拒绝区"
                    )
                else:
                    shape = (
                        f"出现失败请求（A1={knee.a1}，连接/流中断类 {other_n} 条），"
                        "过载表现为连接被切而非明确拒绝"
                    )
            else:
                shape = f"请求全部成功但时延超标（A3={knee.a3}），过载表现为排队而非拒绝"
            lines.append(
                f"并发 {finding.knee_concurrency} 起 SLA 跌破 {sla_target:.1%}"
                f"（{_pct(knee.sla, 3)}），{shape}。"
            )

    # ---- 6) 容量判定的补充说明 ----
    # find_capacity 产生的 notes（未观测到拐点、SLA 重新达标抖动、零样本级别、
    # 样本充足级别无双达标等）必须出现在结论里：例如“SLA 容量 = 梯度最后一级”
    # 若不说明，读者会把它当成服务上限。
    # 注意：调用方会给每个 item 加 "- " 前缀，这里不要再自带项目符号或空行。
    if finding.notes:
        lines.append("容量判定的补充说明：")
        for note in finding.notes:
            lines.append(note)
    if watermark is not None and watermark.enabled and watermark.notes:
        lines.append("水位判定的补充说明：")
        for note in watermark.notes:
            lines.append(note)
    if qos is not None and qos.usable:
        scan = qos.scan
        pred_txt = (
            f"预测质量容量边界 = 并发 {scan.quality_capacity}（实测 {finding.quality_capacity}）"
            if scan is not None and scan.quality_capacity is not None
            else "预测边界见第九节"
        )
        lines.append(
            f"服务质量预测模型已训练（验证 R²：TTFT={qos.report.r2_ttft:.3f} / "
            f"TPOT={qos.report.r2_tpot:.3f}，见第九节）：{pred_txt}；"
            "预测仅对训练包络内的负载点有效，容量结论仍以实测连续前缀规则为准。"
        )
    return lines


def build_recommendations(
    cfg: Config,
    stages: Sequence[StageStats],
    finding: CapacityFinding,
    qos: Optional[QoSResult] = None,
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
        # 无限队列部署的识别：全程无 429/529 拒绝，失败以客户端超时为主。
        # 此时过载不投影为拒绝而投影为超时（仍计 A1、拉低 SLA），
        # 若生产同样如此，可用性危机会被转嫁为客户端超时。
        timeout_total, reject_total, _other = split_fail_kinds(kinds)
        if reject_total == 0 and timeout_total > 0 and timeout_total >= total_fail / 2:
            recs.append(
                "失败以客户端超时为主且全程无 429/529 拒绝：Router 可能配置了无限队列——"
                "过载不投影为拒绝而投影为客户端超时（仍计入 A1、拉低 SLA）。"
                "若生产部署同样如此，建议服务侧配置队列上限/排队超时让过载可被观测，"
                "或客户端设置合理超时预算并配合熔断/退避，避免可用性问题被无限队列隐藏。"
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
                "本次过载全程无失败请求（A1=0），压力全部由排队消化：TTFT 随并发持续抬升"
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

    # 预测模型使用建议
    if qos is not None and qos.usable:
        recs.append(
            "预测模型使用建议：容量规划可用模型在包络内做 what-if（换流量构成/并发梯度，"
            "无需重跑压测）；模型对训练包络外的负载点只作参考、不构成结论；"
            "更换引擎/模型/并行度后必须重训（模型绑定当次服务配置）。"
        )
        if qos.scan is not None and qos.scan.quality_capacity is not None:
            recs.append(
                f"预测细扫提示：模型在并发 {qos.scan.quality_capacity} 处预测 SLO 跌破目标，"
                "若该点与实测边界不一致，应在该并发附近补测以校准。"
            )

    # 水位运维建议：把水位边界翻译成 Router / 调度层可消费的配置
    if cfg.watermark.enabled:
        states = {s.concurrency: s.watermark_state for s in stages if s.watermark_state != "unknown"}
        if states:
            soft = next((c for c in sorted(states) if states[c] in ("BUSY", "OVERLOAD")), None)
            hard = next((c for c in sorted(states) if states[c] == "OVERLOAD"), None)
            if hard is not None:
                hard_stage = next(s for s in stages if s.concurrency == hard)
                if hard_stage.a1 > 0:
                    timeout_n, reject_n, _other = hard_stage.a1_families()
                    if timeout_n > 0 and timeout_n >= reject_n:
                        shape = (
                            f"客户端超时：A1={hard_stage.a1}/{hard_stage.n_total}，"
                            f"超时族 {timeout_n} 条"
                        )
                        ops = (
                            "该并发起请求无法在客户端预算内完成（Router 不拒绝但排队耗尽，"
                            "疑似无限队列）：建议将 Router 队列上限/排队超时/熔断线设在该并发之下"
                            "并预留余量——无限队列部署需先给队列设上限，否则过载只表现为客户端超时、"
                            "无法被拒绝类监控发现；治理方向为扩容 PD 副本或提高准入并发。"
                        )
                    else:
                        shape = (
                            f"服务端拒绝：A1={hard_stage.a1}/{hard_stage.n_total}，"
                            f"拒绝族 {reject_n} 条"
                        )
                        ops = (
                            "建议将 Router 拒绝阈值/熔断线设在该并发之下并预留余量；"
                            "治理方向为扩容 PD 副本或提高准入并发，而非单纯调高拒绝阈值。"
                        )
                else:
                    shape = f"质量崩塌：A3={hard_stage.a3}/{hard_stage.n_total}"
                    ops = (
                        "建议将该并发作为熔断线并预留余量；"
                        "治理方向为扩容 PD 副本或提高准入并发。"
                    )
                recs.append(
                    f"水位运维建议：硬水位投影出现在并发 {hard}（{shape}）。{ops}"
                )
            elif soft is not None:
                recs.append(
                    f"水位运维建议：软水位投影出现在并发 {soft}（开始排队，尚无失败），"
                    "建议将该并发作为 Router 排队预警线（触发告警/扩容评估），"
                    "并把生产并发控制在质量容量之内以远离排队区。"
                )
            else:
                recs.append(
                    "水位运维建议：已测梯度内未触达任何水位投影，如需确认服务水位，"
                    "应继续提高压力级或换用更陡的并发梯度。"
                )

    recs.append(
        "口径提醒：本报告的 SLA/SLO 为全请求口径（含被拒绝或客户端超时的 A1）；"
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
    qos: Optional[QoSResult] = None,
) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    started = meta.get("started_at", "unknown")
    lines: List[str] = []
    buckets = cfg.requests.length_buckets
    wm_finding = metrics.find_watermarks(stages, cfg.watermark, finding)
    # 预测模型预计算（mix / 分级预测 / 边界细扫）必须先于结论：结论区引用 qos.scan
    _prepare_qos(cfg, store, stages, qos)

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
    for item in build_conclusions(cfg, stages, finding, wm_finding, qos):
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
    if cfg.watermark.enabled:
        lines.append("6. 每级判定水位状态（服务端水位的黑盒投影，见 docs/05-watermark.md）："
                     "NORMAL（无信号）/ BUSY（软水位投影：A3 占比或 TTFT p99 超软门限，"
                     "服务在排队）/ OVERLOAD（硬水位投影：出现失败——服务端拒绝或客户端超时"
                     "（Router 无限队列时过载只投影为超时）——或 A3 占比超硬门限）；"
                     "跨级汇总软/硬/SLA 三个水位边界，分别对应 Router 的排队预警线、"
                     "熔断线与生产并发建议。")
        adaptive_no, baseline_no = "7.", "8."
    else:
        adaptive_no, baseline_no = "6.", "7."
    lines.append(f"{adaptive_no} 闭环低压级别的样本产出受单请求 E2E 限制，测量段样本不足时本级自动延长"
                 "（上限 load.max_stage_duration_s；基线阶段按“最短长度桶”判定，"
                 "因为门限是按桶解析的）；本级结束时仍在途的请求记为放弃"
                 "（aborted），不计入比率。")
    if cfg.slo.target_mode == "baseline_relative":
        lines.append(
            f"{baseline_no} 判定门限来自第二节的低压基线（基线 p99 × 系数），与压力解耦："
            "同一把尺子贯穿全部压力级，因此跨级曲线可比；换引擎/模型只需重测基线。"
        )
    lines.append("")

    # 分级结果 -------------------------------------------------------------
    lines.append("## 四、分级压力测试结果")
    lines.append("")
    wm_on = cfg.watermark.enabled
    rows = []
    for s in sorted(stages, key=lambda x: x.concurrency):
        row = [
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
        ]
        if wm_on:
            row.append(s.watermark_state)
        rows.append(row)
    headers = ["并发", "样本数", "A1 失败", "A2 达标", "A3 不达标", "放弃",
               "SLA", "SLA 判定", "SLO", "SLO 判定", "RPS", "输出 tok/s"]
    if wm_on:
        headers.append("水位")
    lines.append(_table(headers, rows))
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

    # 水位边界明细（服务端水位的黑盒投影） ----------------------------------
    if wm_on:
        lines.append("**水位边界**（服务端水位的可观测投影，判据见 `[watermark]` 配置；"
                     "资源级水位如 KV Cache / 批 slot / GPU 属服务端内部指标，黑盒不可见，"
                     "详见 docs/05-watermark.md）：")
        lines.append("")
        wrows = []
        for s in sorted(stages, key=lambda x: x.concurrency):
            if s.watermark_state == "unknown":
                trigger = "无完成样本，无法判定"
            elif s.watermark_state == "NORMAL":
                trigger = "—"
            else:
                trigger = "；".join(s.watermark_reasons)
            wrows.append([str(s.concurrency), s.watermark_state, trigger])
        lines.append(_table(["并发", "水位状态", "触发原因"], wrows))
        lines.append("")
        parts = []
        if wm_finding.soft_watermark_concurrency is not None:
            parts.append(f"软水位投影（排队预警线）自并发 {wm_finding.soft_watermark_concurrency}")
        else:
            parts.append("已测梯度内未触达软水位投影")
        if wm_finding.hard_watermark_concurrency is not None:
            parts.append(f"硬水位投影（熔断线）自并发 {wm_finding.hard_watermark_concurrency}")
        else:
            parts.append("已测梯度内未触达硬水位投影")
        if wm_finding.sla_watermark_concurrency is not None:
            parts.append(f"SLA 水位（= 质量容量）为并发 {wm_finding.sla_watermark_concurrency}")
        lines.append("> 边界汇总：" + "；".join(parts) + "。")
        lines.append("")
        for note in wm_finding.notes:
            lines.append(f"> {note}")
        if wm_finding.notes:
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
    conn = [(s.concurrency, s.connect_p50_ms, s.connect_p99_ms) for s in stages
            if s.connect_p50_ms is not None]
    if conn:
        peak = max(conn, key=lambda x: x[2] or 0.0)
        lines.append(
            f"> TTFT/E2E 为客户端视角口径，含建连+请求写出时间（每请求独立连接）。"
            f"各级 connect p50/p99 最高为并发 {peak[0]} 的 "
            f"{_ms(peak[1])}/{_ms(peak[2])}ms；若该项占比可观（跨机房/TLS 场景），"
            "解读 TTFT 时应予扣除，或把压测客户端部署到靠近服务的网络位置。"
        )
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
                # 分母用桶内总数（含 A1）：排除 A1 会让 A3 占比虚高
                row.append(f"{e.get('a3', 0)}/{e.get('n', 0)}")
            rows.append(row)
        lines.append(_table(
            ["并发"] + [f"A3/样本（{bucket_label(i, cfg.requests.length_buckets)}）" for i in all_buckets],
            rows,
        ))
        lines.append("")
        lines.append(
            "> 口径：桶内 A3 数 / 桶内总样本数（含 A1，与分级表的 SLA/SLO 分母一致）。"
        )
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

    # 预测模型 -------------------------------------------------------------
    lines.extend(_render_qos_section(cfg, store, stages, qos))

    # 建议 -----------------------------------------------------------------
    lines.append("## 十、建议")
    lines.append("")
    for item in build_recommendations(cfg, stages, finding, qos):
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
    lines.append("- `stages`：每级压力的配置与聚合结果（SLA/SLO/分位数/吞吐/水位状态）。"
                 "`kind='stage'` 为正式压力级，`kind='baseline'` 为 Phase 0 基线级（不参与容量判定）。")
    lines.append("- `requests`：逐请求特征（长度、TTFT/TPOT/E2E、分类、失败类型），"
                 "**不含原始请求与响应内容**。")
    lines.append("- `stage_buckets`：每级按 prompt 长度桶的拆分（A2/A3/超标归因/该桶生效门限）。")
    lines.append("- `baseline`：Phase 0 各长度桶的时延分位数，是 baseline_relative 门限的来源。")
    lines.append("- `stages.watermark_state` / `watermark_reasons_json`：本级水位状态"
                 "（NORMAL/BUSY/OVERLOAD/unknown）与触发原因——服务端水位的黑盒投影，"
                 "见 docs/05-watermark.md；`--report-only` 会按当前 `[watermark]` 配置重算。")
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
                 baseline: Optional[BaselineResult] = None,
                 qos: Optional[QoSResult] = None) -> str:
    text = render_report(cfg, store, stages, finding, meta, thresholds, baseline, qos)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)
