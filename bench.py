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

    # 6) 用压测采样训练出的 MLP 做单点预测（无需 numpy，见 docs/06-qos-model.md）
    python bench.py --qos-model results/qos_model.npz \
        --qos-predict "concurrency=16,prompt_tokens=4096,completion_tokens=512"

    # 7) 容量（吞吐）场景：report.kinds 含 "throughput" 时同一次运行额外产出
    #    吞吐报告与容量面（throughput_model.npz，见 docs/07 §7.10）
    python bench.py --config config.throughput.toml

    # 8) 容量规划反查：扛 X RPM @ 输入长度 L 需要多少并发（无需重跑压测）
    python bench.py --capacity-plan results/throughput_model.npz \
        --target-rpm 500 --input-len 2048
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from bench import __version__, qos
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
    parser.add_argument(
        "--no-qos",
        action="store_true",
        help="关闭服务质量预测模型（仅评估，不训练/不加载 MLP）",
    )
    parser.add_argument(
        "--retrain-qos",
        action="store_true",
        help="忽略已存在的模型文件，强制从当前数据库重训预测模型",
    )
    parser.add_argument("--qos-model", help="覆盖 qos_model.save_path（模型文件路径）")
    parser.add_argument(
        "--qos-predict",
        metavar="SPEC",
        help='用已有模型做单点预测后退出，如 "concurrency=16,prompt_tokens=4096,completion_tokens=512"',
    )
    parser.add_argument(
        "--throughput-model",
        metavar="PATH",
        help="覆盖容量面（throughput_model.npz）输出路径（仅产出吞吐报告时写入）",
    )
    parser.add_argument(
        "--capacity-plan",
        metavar="NPZ",
        help="对已有容量面反查后退出：扛 --target-rpm @ --input-len 需要多少并发",
    )
    parser.add_argument(
        "--target-rpm",
        type=float,
        help="--capacity-plan 的目标吞吐（请求/分钟）",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        help="--capacity-plan 的输入长度（token 数）",
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

    # 预测模型开关（压测后训练/加载 MLP，见 docs/06-qos-model.md）
    if args.qos_model:
        cfg.qos_model.save_path = args.qos_model
    if args.no_qos:
        cfg.qos_model.enabled = False
    if args.retrain_qos and args.no_qos:
        raise ConfigError("--retrain-qos 与 --no-qos 互斥")

    # 容量面路径（吞吐分析的输出工件，见 docs/07 §7.10.5）
    if args.throughput_model:
        cfg.report.throughput_model_path = args.throughput_model


def _qos_predict(cfg: Config, spec: str) -> int:
    """--qos-predict：加载模型做单点预测（无需 numpy）。"""
    from bench.qos import QoSModel

    path = cfg.qos_model.save_path
    if not Path(path).exists():
        print(f"[error] 模型文件不存在: {path}", file=sys.stderr)
        print("       先跑一次完整压测（qos_model.enabled=true）生成模型，"
              "或用 --qos-model 指定路径。", file=sys.stderr)
        return 2
    try:
        fields: Dict[str, float] = {}
        for part in spec.split(","):
            if "=" not in part:
                raise ConfigError(f"预测参数格式错误: {part!r}（应为 key=value）")
            key, value = part.split("=", 1)
            fields[key.strip()] = float(value)
        model = QoSModel.load(path)
        conc = fields.get("concurrency", 1.0)
        prompt = fields.get("prompt_tokens", 1024.0)
        completion = fields.get("completion_tokens", 256.0)
        if conc <= 0 or prompt <= 0 or completion <= 0:
            raise ConfigError("concurrency / prompt_tokens / completion_tokens 必须为正数")
    except (ConfigError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    pred = model.predict(conc, prompt, completion)
    rates = model.pass_rates(
        conc, prompt, completion,
        cfg.slo.ttft_target_s * 1000.0, cfg.slo.tpot_target_ms,
    )
    print("=" * 72)
    print(f"预测输入  : 并发={conc:g}  prompt={prompt:g} tokens  输出={completion:g} tokens")
    print(f"TTFT 预测 : p50={pred.ttft_p50_ms:.1f}ms  p90={pred.ttft_p90_ms:.1f}ms  "
          f"p99={pred.ttft_p99_ms:.1f}ms")
    print(f"TPOT 预测 : p50={pred.tpot_p50_ms:.1f}ms  p90={pred.tpot_p90_ms:.1f}ms  "
          f"p99={pred.tpot_p99_ms:.1f}ms")
    print(f"当前门限  : TTFT<={cfg.slo.ttft_target_s * 1000:.0f}ms  "
          f"TPOT<={cfg.slo.tpot_target_ms:.0f}ms")
    print(f"通过率    : TTFT {rates.ttft:.1%} / TPOT {rates.tpot:.1%} / "
          f"双达标(SLO) {rates.both:.1%}")
    meta = model.meta
    print(f"模型信息  : 训练于 {meta.get('trained_at', '?')}，"
          f"R² TTFT={meta.get('r2_ttft', 0):.3f} / TPOT={meta.get('r2_tpot', 0):.3f}，"
          f"样本 {meta.get('n_samples', 0)}")
    if pred.extrapolated:
        print("⚠ 警告    : 输入超出训练包络（"
              + "；".join(pred.outside) + "），预测不可信、仅供参考")
    print("=" * 72)
    return 0


def _capacity_plan(path: str, target_rpm: float, input_len: int) -> int:
    """--capacity-plan：对容量面反查"扛 X RPM @ 长度 L 需要多少并发"。

    平行于 --qos-predict，但反问题由表格+插值直接回答（不训练，§7.10.5）；
    L 超出测量范围时 clamp 到端点并显式标注 extrapolated。不依赖配置文件
    （只用三个显式参数），因此在 load_config 之前分发。
    """
    from bench.throughput import ThroughputError, ThroughputSurface

    if target_rpm is None or target_rpm <= 0:
        raise ConfigError("--capacity-plan 需要 --target-rpm > 0")
    if input_len is None or input_len <= 0:
        raise ConfigError("--capacity-plan 需要 --input-len > 0")
    if not Path(path).exists():
        print(f"[error] 容量面文件不存在: {path}", file=sys.stderr)
        print("       先跑一次 report.kinds 含 \"throughput\" 的压测生成容量面，"
              "或用 --throughput-model 指定路径。", file=sys.stderr)
        return 2
    try:
        surface = ThroughputSurface.load(path)
    except (ThroughputError, OSError, ValueError) as exc:
        print(f"[error] 容量面加载失败: {exc}", file=sys.stderr)
        return 2

    ans = surface.query(target_rpm, input_len)
    print("=" * 72)
    print(f"反查输入  : 目标 {target_rpm:g} RPM @ 输入长度 {input_len} tokens")
    if ans.mu_rpm is None:
        print("预测容量  : n/a（容量面为空）")
    else:
        print(f"预测容量  : μ = {ans.mu_rpm:,.1f} RPM @ 长度 {ans.effective_len}"
              + (f"（输入 {ans.input_tps:,.0f} + 输出 {ans.output_tps:,.0f} = "
                 f"{ans.mu_tps:,.0f} tokens/s）" if ans.mu_tps is not None else "（token 数据缺失，仅 RPM 可信）"))
        print(f"所需并发  : 约 {ans.req_conc}（该长度取得峰值吞吐的负载水平；"
              "超过它才开始出现失败）")
        verdict = "可达" if ans.achievable else "不可达"
        print(f"判定      : {verdict}"
              + (f"（余量 {ans.margin_rpm:+,.1f} RPM）" if ans.achievable
                 else f"（缺口 {-ans.margin_rpm:,.1f} RPM）"))
        print(f"过载形态  : {ans.a1_family}")
    bound = surface.meta.get("base_url")
    if bound:
        print(f"测量边界  : {bound}（与当前 --base-url 不同则结论不适用）")
    for note in ans.notes:
        print(f"⚠ 注意    : {note}")
    print("=" * 72)
    return 0


def _throughput_report_path(output_path: str) -> str:
    """在线报告 <name>.md → 吞吐报告 <name>_throughput.md（同目录）。"""
    p = Path(output_path)
    return str(p.with_name(p.stem + "_throughput" + p.suffix))


def _emit_reports(cfg: Config, store: RecordStore, stats, finding, meta: Dict[str, str],
                  thresholds=None, baseline=None, force_train_qos: bool = False):
    """按 report.kinds 产出报告（docs/07 §7.10.1：一次采集、两套分析）。

    在线分析与容量分析消费同一批 stats 事实；kinds 之外的分析不产出，
    测量管道本身不做任何场景裁剪。返回 (报告路径列表, qos 结果或 None)。
    """
    from bench.report_batch import write_report_batch

    kinds = set(cfg.report.kinds)
    paths: List[str] = []
    qos_result = None
    if "online" in kinds:
        qos_result = qos.train_or_load(cfg, store, force_train=force_train_qos)
        paths.append(write_report(
            cfg, store, stats, finding, meta, cfg.report.output_path,
            thresholds=thresholds, baseline=baseline, qos=qos_result,
        ))
    if "throughput" in kinds:
        path, _surface = write_report_batch(
            cfg, store, stats, finding, meta, _throughput_report_path(cfg.report.output_path)
        )
        paths.append(path)
    return paths, qos_result


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


def _derive_throughput_fallback(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """旧库（无 ok_rps/input_tps 列）时的报告期推导兜底（docs/07 §7.10.2）。

    ok_rps = rps × n_ok/n_total（精确：rps = n_total/window）；
    input_tps = prompt_tokens_avg × ok_rps（精确：均值×计数=总量，
    成功请求全部带 usage 时与实时聚合值一致）。任一量缺失返回 None，
    由吞吐报告走"诚实退化"（RPM 单口径 + 标注 token 覆盖率），
    不静默填 0（AGENT.md 统计诚实原则）。
    """
    rps = row["rps"]
    n_total = row["n_total"] or 0
    n_ok = row["n_ok"] or 0
    if not rps or not n_total:
        return None, None
    ok_rps = rps * n_ok / n_total
    avg = row["prompt_tokens_avg"]
    input_tps = avg * ok_rps if avg is not None else None
    return ok_rps, input_tps


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
        # 吞吐侧面两列：新库直接读回；旧库缺列时按充分统计量推导兜底
        # （守卫式写法，不假设旧库有新列，也不为兼容铺迁移框架）
        ok_rps = row["ok_rps"] if "ok_rps" in keys else None
        input_tps = row["input_tps"] if "input_tps" in keys else None
        if ok_rps is None or input_tps is None:
            fb_rps, fb_tps = _derive_throughput_fallback(row)
            ok_rps = ok_rps if ok_rps is not None else fb_rps
            input_tps = input_tps if input_tps is not None else fb_tps
        entry.ok_rps = ok_rps
        entry.input_tps = input_tps
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
    # 容量规划反查不依赖配置文件（只用三个显式参数），在加载配置前分发
    if args.capacity_plan:
        return _capacity_plan(args.capacity_plan, args.target_rpm, args.input_len)

    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)
    _default_paths(cfg)

    if args.qos_predict:
        return _qos_predict(cfg, args.qos_predict)

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
            paths, qos_result = _emit_reports(
                cfg, store, stats, finding, meta,
                thresholds=thresholds, baseline=baseline,
                force_train_qos=args.retrain_qos,
            )
            for p in paths:
                print(f"[done] 报告已生成: {p}")
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
        print(f"  报告      : {' + '.join(cfg.report.kinds)}")
        if "throughput" in cfg.report.kinds and cfg.analysis.window_requests > 0:
            print(f"  窗口承诺  : {cfg.analysis.window_requests:,} 请求 / "
                  f"{cfg.analysis.window_hours:g} 小时（安全余量 "
                  f"×{cfg.analysis.window_safety_factor:g}）")
        if cfg.qos_model.enabled:
            print(f"  预测模型  : 开（hidden={cfg.qos_model.hidden}，"
                  f"样本门限 {cfg.qos_model.min_samples}，上限 {cfg.qos_model.max_train_seconds:.0f}s）"
                  f"→ {cfg.qos_model.save_path}")
        else:
            print("  预测模型  : 关（--no-qos / qos_model.enabled=false）")
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
        # 按 report.kinds 产出报告：在线分析与容量分析消费同一批 stats（§7.10.1）；
        # 实时路径下在线分析强制重训（刚测出的数据就该训练新模型，想跳过用 --no-qos），
        # --report-only 路径才优先加载已有模型文件（避免重复训练成本）。
        paths, qos_result = _emit_reports(
            cfg, store, stats, finding, meta,
            thresholds=thresholds, baseline=baseline, force_train_qos=True,
        )
        print("=" * 72)
        print(f"[done] 数据库: {cfg.storage.db_path}")
        for p in paths:
            print(f"[done] 报告  : {p}")
        if qos_result is not None and qos_result.model is not None and qos_result.report.usable:
            print(f"[done] 模型  : {qos_result.model_path}"
                  f"（R² TTFT={qos_result.report.r2_ttft:.3f} / "
                  f"TPOT={qos_result.report.r2_tpot:.3f}，"
                  f"预测见报告第九节）")
        elif qos_result is not None and qos_result.report.skip_reason:
            print(f"[done] 模型  : 未生成（{qos_result.report.skip_reason}）")
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
