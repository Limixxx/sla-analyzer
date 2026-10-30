"""压力编排器：按递增并发逐级施压，产出每级的请求明细与统计。

闭环（closed）模式：固定并发数，每个 worker 串行发请求直至本级结束，
是探测"Router 缓冲/拒绝边界"的标准方式——并发升高时若后端饱和，
排队会体现为 TTFT/TPOT 抬升（A3 增多），队列打满则出现拒绝（A1 增多）。
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Dict, List, Optional

from . import metrics
from .baseline import (
    BaselineResult,
    Thresholds,
    bucket_label,
    compute_baseline,
    describe_thresholds,
    resolve_thresholds,
)
from .client import KIND_ABORTED, RequestResult, execute_request
from .config import Config
from .prompts import PromptFactory
from .store import RecordStore

_RETRY = object()  # take_slot 的“稍后重试”哨兵
_EXTEND_STEP_S = 30.0  # 样本不足时每次自适应延长的步长


class BenchmarkRunner:
    def __init__(self, cfg: Config, store: RecordStore, log: Optional[Callable[[str], None]] = None):
        self._cfg = cfg
        self._store = store
        self._log = log or (lambda msg: print(msg, flush=True))
        self._factory = PromptFactory(cfg.requests)
        self._stop = asyncio.Event()
        self._baseline: Optional[BaselineResult] = None
        self._thresholds: Optional[Thresholds] = None

    # ------------------------------------------------------------------ #
    @property
    def baseline(self) -> Optional[BaselineResult]:
        """Phase 0 基线结果（未启用基线时为 None）。"""
        return self._baseline

    @property
    def thresholds(self) -> Optional[Thresholds]:
        """本次运行生效的判定门限。"""
        return self._thresholds

    # ------------------------------------------------------------------ #
    async def run(self) -> List[metrics.StageStats]:
        cfg = self._cfg
        all_stats: List[metrics.StageStats] = []

        # Phase 0：低压基线剖析。仅在需要时执行，但其分位数是 baseline_relative
        # 门限的唯一来源；跳过则该模式自动回退为 absolute 门限。
        if cfg.load.baseline_enabled:
            self._baseline = await self._run_baseline()

        self._thresholds = resolve_thresholds(cfg, self._baseline)
        for line in describe_thresholds(self._thresholds, cfg.requests.length_buckets):
            self._log(f"  门限[{cfg.slo.target_mode}] {line}")

        for index, concurrency in enumerate(cfg.load.concurrency_steps, start=1):
            self._log(
                f"[stage {index}/{len(cfg.load.concurrency_steps)}] 并发={concurrency} "
                f"持续={cfg.load.stage_duration_s}s（预热 {cfg.load.warmup_s}s，"
                f"样本不足 {cfg.analysis.min_samples_per_stage} 个时最长延至 "
                f"{cfg.load.max_stage_duration_s}s）..."
            )
            stats = await self._run_stage(index, concurrency)
            all_stats.append(stats)
            abort_txt = f" 放弃={stats.n_aborted}" if stats.n_aborted else ""
            self._log(
                f"  -> 样本={stats.n_total} A1={stats.a1} A2={stats.a2} A3={stats.a3} "
                f"SLA={stats.sla:.4f} SLO={stats.slo:.4f}{abort_txt} "
                f"TTFT_p99={_fmt(stats.ttft_p99_ms)}ms TPOT_p99={_fmt(stats.tpot_p99_ms)}ms"
            )
        return all_stats

    # ------------------------------------------------------------------ #
    async def run_baseline_only(self) -> BaselineResult:
        """仅执行 Phase 0 基线剖析并解析门限（--baseline-only）。

        用于更换推理引擎/模型后快速重测基线：确认各桶分位数与生效门限
        符合预期后，再运行完整压测。基线结果已写入数据库，同一配置的
        后续完整运行会重新测量（不直接复用），避免基线随服务漂移。
        """
        cfg = self._cfg
        self._baseline = await self._run_baseline()
        self._thresholds = resolve_thresholds(cfg, self._baseline)
        for line in describe_thresholds(self._thresholds, cfg.requests.length_buckets):
            self._log(f"  门限[{cfg.slo.target_mode}] {line}")
        return self._baseline

    # ------------------------------------------------------------------ #
    async def _run_baseline(self) -> BaselineResult:
        """Phase 0：低压基线测量，产出各长度桶的时延分位数。

        基线只统计分位数与失败数，不对 A2/A3 判定——基线是尺子的来源，
        再拿它判达标会形成循环论证。
        """
        cfg = self._cfg
        conc = cfg.load.baseline_concurrency
        self._log(
            f"[baseline] Phase 0 低压基线剖析：并发={conc} "
            f"持续={cfg.load.baseline_duration_s}s（预热 {cfg.load.baseline_warmup_s}s）"
        )
        stats = await self._run_stage(
            index=0, concurrency=conc, kind="baseline",
            duration_s=cfg.load.baseline_duration_s, warmup_s=cfg.load.baseline_warmup_s,
        )
        rows = self._store.stage_rows(stats.stage_id, include_warmup=False)
        result = compute_baseline(
            rows,
            cfg.requests.length_buckets,
            min_samples=cfg.analysis.min_samples_per_stage,
            min_completion_tokens=cfg.slo.min_completion_tokens,
        )
        result.concurrency = conc
        result.duration_s = cfg.load.baseline_duration_s
        self._store.save_baseline(stats.stage_id, conc, cfg.load.baseline_duration_s, result)

        for entry in result.buckets:
            label = bucket_label(entry.bucket, cfg.requests.length_buckets)
            self._log(
                f"  -> 桶 {label}: 样本={entry.n_ok} A1={entry.a1} "
                f"TTFT_p50={_fmt(entry.ttft_p50_ms)}ms TTFT_p99={_fmt(entry.ttft_p99_ms)}ms "
                f"TPOT_p50={_fmt(entry.tpot_p50_ms)}ms TPOT_p99={_fmt(entry.tpot_p99_ms)}ms"
            )
        for note in result.notes:
            self._log(f"  .. {note}")
        return result

    # ------------------------------------------------------------------ #
    async def _run_stage(
        self,
        index: int,
        concurrency: int,
        kind: str = "stage",
        duration_s: Optional[float] = None,
        warmup_s: Optional[float] = None,
    ) -> metrics.StageStats:
        cfg = self._cfg
        store = self._store
        self._stop.clear()

        stage_duration = duration_s if duration_s is not None else cfg.load.stage_duration_s
        stage_warmup = warmup_s if warmup_s is not None else cfg.load.warmup_s

        stage_id = store.start_stage(
            concurrency=concurrency,
            mode=cfg.load.mode,
            duration_s=stage_duration,
            warmup_s=stage_warmup,
            kind=kind,
        )

        started = time.monotonic()
        hard_deadline = started + max(cfg.load.max_stage_duration_s, stage_duration)
        warmup_deadline = started + stage_warmup
        stage_deadline = started + stage_duration
        extended = {"on": False}
        issued = {"measured": 0, "warmup": 0}

        async def take_slot() -> Optional[int]:
            """申请一个请求额度；返回 is_warmup 标记，None 表示应退出。"""
            nonlocal stage_deadline
            now = time.monotonic()
            if now >= stage_deadline:
                # 自适应延长：闭环低压级别的样本产出速率受单请求 E2E 限制
                # （并发 1、E2E≈3.5s 时 60s 仅约 17 个样本），若测量段样本不足
                # 则继续施压至 max_stage_duration_s，避免低并发优质级别因样本量
                # 被整体排除在容量判定之外。
                if (
                    issued["measured"] > 0
                    and issued["measured"] < cfg.analysis.min_samples_per_stage
                    and now < hard_deadline
                ):
                    stage_deadline = min(stage_deadline + _EXTEND_STEP_S, hard_deadline)
                    extended["on"] = True
                    return _RETRY
                return None
            if now < warmup_deadline:
                if issued["warmup"] >= cfg.load.max_warmup_requests_per_stage:
                    await asyncio.sleep(0.05)  # 预热额度用尽，等待测量段开始
                    return None if time.monotonic() >= stage_deadline else _RETRY
                issued["warmup"] += 1
                return 1
            if issued["measured"] >= cfg.load.max_requests_per_stage:
                return None
            issued["measured"] += 1
            return 0

        async def _issue(worker_id: int, slot: int, spec) -> RequestResult:
            """发请求；被 drain 取消时先落一条 aborted 记录再抛出。

            否则本级结束时仍在途的（恰是最慢的）请求会被静默丢弃，
            造成存活者偏差：时延分位数与 A3 占比被系统性低估。
            """
            prompt = self._factory.build(spec)
            try:
                return await execute_request(
                    cfg.service, cfg.requests, prompt, spec,
                    stage_id=stage_id, concurrency=concurrency,
                    worker_id=worker_id, is_warmup=slot, mode=cfg.load.mode,
                )
            except asyncio.CancelledError:
                store.add_request(RequestResult(
                    stage_id=stage_id,
                    concurrency=concurrency,
                    worker_id=worker_id,
                    is_warmup=slot,
                    mode=cfg.load.mode,
                    sent_at=time.time(),
                    prompt_tokens_planned=spec.prompt_tokens_target,
                    max_tokens=spec.max_tokens,
                    status="aborted",
                    error_kind=KIND_ABORTED,
                    note="in-flight when stage ended; abandoned after drain timeout",
                ).as_row())
                raise

        async def closed_worker(worker_id: int) -> None:
            while not self._stop.is_set():
                slot = await take_slot()
                if slot is None:
                    return
                if slot is _RETRY:
                    continue
                if cfg.load.think_time_s > 0:
                    await asyncio.sleep(cfg.load.think_time_s)
                spec = self._factory.sample_spec()
                result = await _issue(worker_id, slot, spec)
                store.add_request(result.as_row())

        async def open_worker() -> None:
            interval = 1.0 / cfg.load.target_rps if cfg.load.target_rps > 0 else 0.0
            while not self._stop.is_set():
                slot = await take_slot()
                if slot is None:
                    return
                if slot is _RETRY:
                    continue
                in_flight = len([t for t in asyncio.all_tasks() if t is not asyncio.current_task()])
                if in_flight >= cfg.load.max_inflight:
                    await asyncio.sleep(0.01)
                    continue
                spec = self._factory.sample_spec()
                asyncio.create_task(_issue(0, slot, spec)).add_done_callback(
                    lambda fut: _safe_add(store, fut, stage_id)
                )
                if interval > 0:
                    await asyncio.sleep(interval)

        if cfg.load.mode == "closed":
            workers = [asyncio.create_task(closed_worker(i)) for i in range(concurrency)]
        else:
            workers = [asyncio.create_task(open_worker())]

        # worker 的退出时机由 take_slot 的截止时间控制（含自适应延长），
        # 这里只兜底：整体不超过 max_stage_duration_s + drain_timeout_s。
        try:
            done, pending = await asyncio.wait(workers, timeout=max(cfg.load.max_stage_duration_s, stage_duration))
        except asyncio.CancelledError:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            store.flush_requests()
            raise
        if pending:
            self._stop.set()
            done, pending = await asyncio.wait(pending, timeout=cfg.load.drain_timeout_s)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        store.flush_requests()
        if extended["on"]:
            actual = time.monotonic() - started
            self._log(
                f"  .. 测量段样本不足 {cfg.analysis.min_samples_per_stage} 个，"
                f"本级已自适应延长至 {actual:.0f}s（上限 {hard_deadline - started:.0f}s）"
            )
        rows = store.stage_rows(stage_id, include_warmup=False)
        warmup_rows = store.warmup_count(stage_id)
        stats = metrics.compute_stage_stats(
            stage_id=stage_id,
            concurrency=concurrency,
            rows=rows,
            warmup_rows=warmup_rows,
            slo=cfg.slo,
            analysis=cfg.analysis,
            duration_s=stage_duration,
            warmup_s=stage_warmup,
            thresholds=self._thresholds,
            buckets=cfg.requests.length_buckets,
        )
        # 放弃数按全量统计（含预热段）：预热段被放弃同样说明 drain 窗口不够
        stats.n_aborted = store.aborted_count(stage_id)
        store.finish_stage(stage_id, stats)
        return stats


def _safe_add(store: RecordStore, fut: asyncio.Future, stage_id: int) -> None:
    try:
        result: RequestResult = fut.result()
        if result.stage_id == stage_id:
            store.add_request(result.as_row())
    except Exception:
        pass


def _fmt(value: Optional[float]) -> str:
    return f"{value:.1f}" if value is not None else "n/a"
