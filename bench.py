#!/usr/bin/env python3
"""PD 分离推理 QoS 基准测试 CLI。

用法示例：
    # 0) 预检：连通性 + 单请求探针（验证配置，几分钟内完成）
    python bench.py --config config.toml --probe

    # 1) 正式压测：按 config 中的并发梯度逐级施压
    python bench.py --config config.toml

    # 2) 命令行覆盖关键参数（不改配置文件）
    python bench.py --config config.toml --base-url http://router:8000 \
        --model qwen --steps 8,16,32,64 --duration 60 --warmup 10
    #    --duration 会按 5 倍联动单级延长上限；需要其他上限用 --max-duration

    # 3) 仅由已有数据库重新生成报告
    python bench.py --report-only --db results/bench_xxx.db --config config.toml

    # 4) 更换引擎/模型后快速重测基线（仅 Phase 0，不压测）
    python bench.py --config config.toml --baseline-only

    # 5) 跳过 Phase 0，直接用 absolute 门限压测
    python bench.py --config config.toml --no-baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from bench import __version__
from bench.baseline import resolve_thresholds
from bench.config import Config, ConfigError, load_config
from bench.metrics import StageStats, classify_watermark, find_capacity
from bench.prompts import PromptFactory
from bench.report import write_report
from bench.runner import BenchmarkRunner
from bench.store import RecordStore


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PD 分离推理服务质量（SLO/SLA）递增并发压测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.toml", help="配置文件路径（.toml/.json）")
    parser.add_argument("--base-url", help="覆盖 service.base_url")
    parser.add_argument("--model", help="覆盖 service.model")
    parser.add_argument("--api-key", help="覆盖 service.api_key")
    parser.add_argument("--path", help="覆盖 service.path")
    parser.add_argument("--steps", help="覆盖并发梯度，逗号分隔，如 8,16,32,64")
    parser.add_argument("--duration", type=float, help="覆盖每级稳压时长（秒）")
    parser.add_argument("--warmup", type=float, help="覆盖每级预热时长（秒）")
    parser.add_argument(
        "--max-duration",
        type=float,
        help="覆盖单级自适应延长上限（秒）；不指定时随 --duration 按 5 倍联动",
    )
    parser.add_argument("--db", help="覆盖数据库输出路径")
    parser.add_argument("--report", help="覆盖报告输出路径")
    parser.add_argument("--probe", action="store_true", help="仅做连通性/单请求探针，不压测")
    parser.add_argument("--report-only", action="store_true", help="由已有数据库重新生成报告")
    parser.add_argument(
        "--target-mode",
        choices=["absolute", "baseline_relative"],
        help="覆盖 slo.target_mode（门限来源）",
    )
    parser.add_argument("--no-baseline", action="store_true", help="跳过 Phase 0 基线剖析")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="仅执行 Phase 0 基线剖析并打印各桶分位数与生效门限，不做正式压测",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def _apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> None:
    overrides = {
        "base_url": args.base_url,
        "model": args.model,
        "api_key": args.api_key,
        "path": args.path,
        "stage_duration_s": args.duration,
        "warmup_s": args.warmup,
        "db_path": args.db,
        "report_path": args.report,
    }
    cfg.service.base_url = overrides.pop("base_url") or cfg.service.base_url
    cfg.service.model = overrides.pop("model") or cfg.service.model
    cfg.service.api_key = overrides.pop("api_key") or cfg.service.api_key
    cfg.service.path = overrides.pop("path") or cfg.service.path
    cfg.load.stage_duration_s = overrides.pop("stage_duration_s") or cfg.load.stage_duration_s
    cfg.load.warmup_s = overrides.pop("warmup_s") if overrides["warmup_s"] is not None else cfg.load.warmup_s
    # 延长上限与名义时长联动：默认配置即 5 倍（60s → 300s）。若 --duration 单独
    # 覆盖而 max_stage_duration_s 保持配置值，快速验证场景（如 --duration 6）
    # 的单级仍可能被延长到 300s（50 倍），与调用意图相悖。
    if args.max_duration:
        if args.max_duration < cfg.load.stage_duration_s:
            raise ConfigError(
                f"--max-duration({args.max_duration}) 不能小于每级时长({cfg.load.stage_duration_s})"
            )
        cfg.load.max_stage_duration_s = args.max_duration
    elif args.duration:
        cfg.load.max_stage_duration_s = max(args.duration * 5.0, 30.0)
    cfg.storage.db_path = overrides.pop("db_path") or cfg.storage.db_path
    cfg.report.output_path = overrides.pop("report_path") or cfg.report.output_path
    if args.steps:
        try:
            cfg.load.concurrency_steps = sorted({int(x) for x in args.steps.split(",") if x.strip()})
        except ValueError:
            raise ConfigError(f"--steps 解析失败: {args.steps}")
    if args.target_mode:
        cfg.slo.target_mode = args.target_mode
    if args.no_baseline:
        if args.baseline_only:
            raise ConfigError("--no-baseline 与 --baseline-only 互斥")
        cfg.load.baseline_enabled = False
        if cfg.slo.target_mode == "baseline_relative":
            cfg.slo.target_mode = "absolute"
            print("[note] --no-baseline 已跳过 Phase 0 基线，target_mode 由 "
                  "baseline_relative 回退为 absolute 门限。")


def _default_paths(cfg: Config) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not cfg.storage.db_path:
        cfg.storage.db_path = str(Path("results") / f"bench_{stamp}.db")
    if not cfg.report.output_path:
        cfg.report.output_path = str(Path("results") / f"report_{stamp}.md")


async def _probe_impl(cfg: Config) -> int:
    from bench.client import execute_request
    from bench.prompts import RequestSpec

    base = cfg.service.base_url.rstrip("/")
    print(f"[probe] 目标: {base}{cfg.service.path}  模型: {cfg.service.model}")

    # 尝试列出模型（部分部署不带 /v1/models，失败可忽略）
    from bench.httpclient import open_stream

    try:
        url = base + "/v1/models"
        reader, writer, resp = await open_stream(url, "GET", {}, None, cfg.service.connect_timeout_s, cfg.service.verify_tls)
        body = await resp.read_text(limit=65536)
        writer.close()
        print(f"[probe] GET /v1/models -> HTTP {resp.status}")
        try:
            data = json.loads(body)
            models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
            if models:
                print(f"[probe] 可用模型: {models}")
        except ValueError:
            pass
    except Exception as exc:
        print(f"[probe] GET /v1/models 失败（可忽略）: {type(exc).__name__}: {exc}")

    factory = PromptFactory(cfg.requests)
    spec = RequestSpec(prompt_tokens_target=64, max_tokens=64)
    prompt = factory.build(spec)
    print(f"[probe] 发送单请求（prompt≈{spec.prompt_tokens_target} tok, max_tokens={spec.max_tokens}）...")
    result = await execute_request(
        cfg.service, cfg.requests, prompt, spec,
        stage_id=0, concurrency=1, worker_id=0, is_warmup=0, mode="probe",
    )
    print(f"[probe] HTTP={result.http_code} status={result.status} error_kind={result.error_kind}")
    if result.status == "ok":
        print(
            f"[probe] TTFT={result.ttft_ms:.0f}ms TPOT={result.tpot_ms if result.tpot_ms is None else round(result.tpot_ms, 1)}ms "
            f"E2E={result.e2e_ms:.0f}ms prompt_tokens={result.prompt_tokens} "
            f"completion_tokens={result.completion_tokens} (token_source={result.token_source})"
        )
        print("[probe] 结论：服务可用，配置有效，可以开始正式压测。")
        return 0
    print(f"[probe] 请求失败: {result.note}")
    print("[probe] 结论：请检查 base_url / api_key / path / 网络连通性。")
    return 2


def _stats_from_store(store: RecordStore, cfg: Config):
    """从数据库重建分级统计（--report-only 路径）。

    门限按 target_mode 重新解析：absolute 用配置值；baseline_relative 从
    baseline 表重建基线后按桶计算。旧版本数据库没有 baseline 表时自动回退。
    返回 (stats, thresholds, baseline)。
    """
    baseline = store.load_baseline() if cfg.slo.target_mode == "baseline_relative" else None
    thresholds = resolve_thresholds(cfg, baseline)

    stats: list = []
    for row in store.all_stages():
        keys = row.keys()
        entry = StageStats(
            stage_id=row["id"],
            concurrency=row["concurrency"],
            duration_s=row["duration_s"] or 0.0,
            warmup_s=row["warmup_s"] or 0.0,
            n_total=row["n_total"] or 0,
            n_warmup=row["n_warmup"] or 0,
            n_ok=row["n_ok"] or 0,
            n_failed=row["n_failed"] or 0,
            n_aborted=(row["n_aborted"] or 0) if "n_aborted" in keys else 0,
            a1=row["a1"] or 0,
            a2=row["a2"] or 0,
            a3=row["a3"] or 0,
            slo=row["slo"] or 0.0,
            sla=row["sla"] or 0.0,
            fail_kinds=json.loads(row["fail_kinds_json"] or "{}"),
            ttft_p50_ms=row["ttft_p50_ms"], ttft_p90_ms=row["ttft_p90_ms"], ttft_p99_ms=row["ttft_p99_ms"],
            tpot_p50_ms=row["tpot_p50_ms"], tpot_p90_ms=row["tpot_p90_ms"], tpot_p99_ms=row["tpot_p99_ms"],
            e2e_p50_ms=row["e2e_p50_ms"], e2e_p99_ms=row["e2e_p99_ms"],
            connect_p50_ms=row["connect_p50_ms"] if "connect_p50_ms" in keys else None,
            connect_p99_ms=row["connect_p99_ms"] if "connect_p99_ms" in keys else None,
            rps=row["rps"], output_tps=row["output_tps"],
            prompt_tokens_avg=row["prompt_tokens_avg"], completion_tokens_avg=row["completion_tokens_avg"],
            ttft_target_ms=cfg.slo.ttft_target_s * 1000.0,
            tpot_target_ms=cfg.slo.tpot_target_ms,
            sla_target=cfg.analysis.sla_target,
            slo_target=cfg.analysis.slo_target,
        )
        # 分桶明细：DB 里有则按桶重建，否则退化为单桶（兼容旧库）
        bucket_rows = store.stage_buckets(entry.stage_id)
        if bucket_rows:
            for b in bucket_rows:
                entry.bucket_stats[b["bucket"]] = {
                    "n": b["n"] or 0, "a1": b["a1"] or 0, "a2": b["a2"] or 0, "a3": b["a3"] or 0,
                    "a3_ttft_only": b["a3_ttft_only"] or 0,
                    "a3_tpot_only": b["a3_tpot_only"] or 0,
                    "a3_both": b["a3_both"] or 0,
                    "ttft_p99_ms": b["ttft_p99_ms"], "tpot_p99_ms": b["tpot_p99_ms"],
                    "ttft_limit_ms": b["ttft_limit_ms"], "tpot_limit_ms": b["tpot_limit_ms"],
                }
        # 水位状态按当前 [watermark] 配置现场重算（与门限解析一样幂等）；
        # DB 中的 watermark_state 仅作审计留存，不以它为准
        state, reasons = classify_watermark(entry, cfg.watermark)
        entry.watermark_state = state
        entry.watermark_reasons = reasons
        stats.append(entry)
    return stats, thresholds, baseline


async def _amain(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)
    _default_paths(cfg)

    if args.probe:
        return await _probe_impl(cfg)

    if args.baseline_only:
        if cfg.slo.target_mode != "baseline_relative":
            raise ConfigError(
                "--baseline-only 仅用于 baseline_relative 模式（absolute 模式门限与基线无关，"
                "直接用 --probe 或完整压测即可）"
            )
        store = RecordStore(cfg.storage.db_path)
        try:
            store.set_meta("started_at", time.strftime("%Y-%m-%d %H:%M:%S"))
            store.set_meta("config", json.dumps(cfg.summary(), ensure_ascii=False))
            store.set_meta("db_path", cfg.storage.db_path)
            store.set_meta("tool_version", __version__)
            runner = BenchmarkRunner(cfg, store)
            baseline = await runner.run_baseline_only()
            print("=" * 72)
            print(f"[done] 基线已写入数据库: {cfg.storage.db_path}")
            if any(b in baseline.thin_buckets for b in range(len(baseline.buckets))):
                print(
                    "[done] 提示：部分桶样本不足，门限将回退 absolute；"
                    "可增大 load.baseline_duration_s 后重测。"
                )
            print("[done] 未生成正式报告（--baseline-only 不做压测）；确认门限后可运行完整压测。")
            print("=" * 72)
            return 0
        finally:
            store.close()

    if args.report_only:
        if not Path(cfg.storage.db_path).exists():
            print(f"[error] 数据库不存在: {cfg.storage.db_path}", file=sys.stderr)
            return 2
        store = RecordStore(cfg.storage.db_path)
        try:
            stats, thresholds, baseline = _stats_from_store(store, cfg)
            finding = find_capacity(stats, cfg.analysis)
            meta = store.meta_items()
            meta.setdefault("db_path", cfg.storage.db_path)
            path = write_report(
                cfg, store, stats, finding, meta, cfg.report.output_path,
                thresholds=thresholds, baseline=baseline,
            )
            print(f"[done] 报告已生成: {path}")
            return 0
        finally:
            store.close()

    store = RecordStore(cfg.storage.db_path)
    try:
        store.set_meta("started_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        store.set_meta("config", json.dumps(cfg.summary(), ensure_ascii=False))
        store.set_meta("db_path", cfg.storage.db_path)
        store.set_meta("tool_version", __version__)

        print("=" * 72)
        print("PD 分离推理 QoS 压测")
        print(f"  目标      : {cfg.service.base_url}{cfg.service.path}  model={cfg.service.model}")
        if cfg.slo.target_mode == "absolute":
            print(f"  门限      : TTFT<={cfg.slo.ttft_target_s}s  TPOT<={cfg.slo.tpot_target_ms}ms"
                  f"（absolute，全部长度桶统一）")
        else:
            print(f"  门限      : baseline_relative（基线 p99 × TTFT {cfg.slo.ttft_factor} / "
                  f"TPOT {cfg.slo.tpot_factor}，按长度桶 {cfg.requests.length_buckets} 生效）")
        print(f"  目标      : SLA>={cfg.analysis.sla_target:.1%}  SLO>={cfg.analysis.slo_target:.1%}")
        if cfg.watermark.enabled:
            print(f"  水位      : 软=A3>={cfg.watermark.soft_slo_violation_ratio:.0%}，"
                  f"硬=A1>={cfg.watermark.hard_reject_ratio:.0%} 或 "
                  f"A3>={cfg.watermark.hard_slo_violation_ratio:.0%}"
                  + (f"（TTFT p99>={cfg.watermark.soft_ttft_p99_ms:.0f}ms 亦判软）"
                     if cfg.watermark.soft_ttft_p99_ms > 0 else ""))
        print(f"  压力级    : {cfg.load.concurrency_steps}  每级 {cfg.load.stage_duration_s}s"
              f"（预热 {cfg.load.warmup_s}s）")
        if cfg.derived_lengths:
            print(f"  长度分布  : {'、'.join(cfg.derived_lengths)} 未显式配置，"
                  f"按 max_context_tokens={cfg.requests.max_context_tokens} 自动推导"
                  f"（prompt={cfg.requests.prompt_token_targets}，桶={cfg.requests.length_buckets}）")
        print(f"  数据库    : {cfg.storage.db_path}")
        print("=" * 72)

        runner = BenchmarkRunner(cfg, store)
        stats = await runner.run()
        thresholds = runner.thresholds
        baseline = runner.baseline
        finding = find_capacity(stats, cfg.analysis)

        meta = store.meta_items()
        path = write_report(
            cfg, store, stats, finding, meta, cfg.report.output_path,
            thresholds=thresholds, baseline=baseline,
        )
        print("=" * 72)
        print(f"[done] 数据库: {cfg.storage.db_path}")
        print(f"[done] 报告  : {path}")
        if finding.quality_capacity is not None:
            print(
                f"[done] 结论  : 质量有效容量（SLA>={cfg.analysis.sla_target:.1%} 且 "
                f"SLO>={cfg.analysis.slo_target:.1%}）= 并发 {finding.quality_capacity}；"
                f"3 个 9 口径（仅 SLA）容量 = {finding.sla_capacity}"
                + (
                    f"（该点 SLO={finding.slo_at_sla_capacity:.4f}，无质量意义）"
                    if finding.slo_at_sla_capacity is not None
                    and finding.slo_at_sla_capacity < cfg.analysis.slo_target
                    else ""
                )
            )
        elif all(s.n_total == 0 for s in stats):
            print(
                f"[done] 结论  : 所有压力级均无完成请求（共放弃 "
                f"{sum(s.n_aborted for s in stats)} 个在途请求），"
                "请增大 load.drain_timeout_s / load.stage_duration_s 后复测"
            )
        else:
            ref = finding.reference_capacity
            sla_txt = str(finding.sla_capacity) if finding.sla_capacity is not None else "未达到"
            print(
                f"[done] 结论  : 未找到同时满足 SLA/SLO 目标的压力级"
                + (f"；低置信度参考容量 = 并发 {ref}（SLO={finding.slo_at_reference_capacity:.4f}）" if ref is not None else "")
                + f"；3 个 9 口径（仅 SLA）容量 = {sla_txt}"
                + (
                    f"（该点 SLO={finding.slo_at_sla_capacity:.4f}，无质量意义）"
                    if finding.slo_at_sla_capacity is not None
                    and finding.slo_at_sla_capacity < cfg.analysis.slo_target
                    else ""
                )
            )
        print("=" * 72)
        return 0
    finally:
        store.close()


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_amain(args))
    except ConfigError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[abort] 用户中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
