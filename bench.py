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

    # 3) 仅由已有数据库重新生成报告
    python bench.py --report-only --db results/bench_xxx.db --config config.toml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from bench import __version__
from bench.config import Config, ConfigError, load_config
from bench.metrics import StageStats, find_capacity
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
    parser.add_argument("--db", help="覆盖数据库输出路径")
    parser.add_argument("--report", help="覆盖报告输出路径")
    parser.add_argument("--probe", action="store_true", help="仅做连通性/单请求探针，不压测")
    parser.add_argument("--report-only", action="store_true", help="由已有数据库重新生成报告")
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
    cfg.storage.db_path = overrides.pop("db_path") or cfg.storage.db_path
    cfg.report.output_path = overrides.pop("report_path") or cfg.report.output_path
    if args.steps:
        try:
            cfg.load.concurrency_steps = sorted({int(x) for x in args.steps.split(",") if x.strip()})
        except ValueError:
            raise ConfigError(f"--steps 解析失败: {args.steps}")


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


def _stats_from_store(store: RecordStore, cfg: Config) -> list:
    stats: list = []
    for row in store.all_stages():
        stats.append(StageStats(
            stage_id=row["id"],
            concurrency=row["concurrency"],
            duration_s=row["duration_s"] or 0.0,
            warmup_s=row["warmup_s"] or 0.0,
            n_total=row["n_total"] or 0,
            n_warmup=row["n_warmup"] or 0,
            n_ok=row["n_ok"] or 0,
            n_failed=row["n_failed"] or 0,
            a1=row["a1"] or 0,
            a2=row["a2"] or 0,
            a3=row["a3"] or 0,
            slo=row["slo"] or 0.0,
            sla=row["sla"] or 0.0,
            fail_kinds=json.loads(row["fail_kinds_json"] or "{}"),
            ttft_p50_ms=row["ttft_p50_ms"], ttft_p90_ms=row["ttft_p90_ms"], ttft_p99_ms=row["ttft_p99_ms"],
            tpot_p50_ms=row["tpot_p50_ms"], tpot_p90_ms=row["tpot_p90_ms"], tpot_p99_ms=row["tpot_p99_ms"],
            e2e_p50_ms=row["e2e_p50_ms"], e2e_p99_ms=row["e2e_p99_ms"],
            rps=row["rps"], output_tps=row["output_tps"],
            prompt_tokens_avg=row["prompt_tokens_avg"], completion_tokens_avg=row["completion_tokens_avg"],
            ttft_target_ms=cfg.slo.ttft_target_s * 1000.0,
            tpot_target_ms=cfg.slo.tpot_target_ms,
            sla_target=cfg.analysis.sla_target,
            slo_target=cfg.analysis.slo_target,
        ))
    return stats


async def _amain(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)
    _default_paths(cfg)

    if args.probe:
        return await _probe_impl(cfg)

    if args.report_only:
        if not Path(cfg.storage.db_path).exists():
            print(f"[error] 数据库不存在: {cfg.storage.db_path}", file=sys.stderr)
            return 2
        store = RecordStore(cfg.storage.db_path)
        try:
            stats = _stats_from_store(store, cfg)
            finding = find_capacity(stats, cfg.analysis)
            meta = store.meta_items()
            meta.setdefault("db_path", cfg.storage.db_path)
            path = write_report(cfg, store, stats, finding, meta, cfg.report.output_path)
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
        print(f"  门限      : TTFT<={cfg.slo.ttft_target_s}s  TPOT<={cfg.slo.tpot_target_ms}ms")
        print(f"  目标      : SLA>={cfg.analysis.sla_target:.1%}  SLO>={cfg.analysis.slo_target:.1%}")
        print(f"  压力级    : {cfg.load.concurrency_steps}  每级 {cfg.load.stage_duration_s}s"
              f"（预热 {cfg.load.warmup_s}s）")
        print(f"  数据库    : {cfg.storage.db_path}")
        print("=" * 72)

        runner = BenchmarkRunner(cfg, store)
        stats = await runner.run()
        finding = find_capacity(stats, cfg.analysis)

        meta = store.meta_items()
        path = write_report(cfg, store, stats, finding, meta, cfg.report.output_path)
        print("=" * 72)
        print(f"[done] 数据库: {cfg.storage.db_path}")
        print(f"[done] 报告  : {path}")
        if finding.reached:
            print(f"[done] 结论  : 3 个 9（SLA>={cfg.analysis.sla_target:.1%}）对应最高并发 "
                  f"{finding.sla_capacity}，该点 SLO={finding.slo_at_sla_capacity:.4f}")
        else:
            print(f"[done] 结论  : 未达到 SLA {cfg.analysis.sla_target:.1%} 目标，"
                  f"最高 SLA={finding.best_sla:.4f}（并发 {finding.best_sla_concurrency}）")
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
