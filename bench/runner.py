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
from .client import RequestResult, execute_request
from .config import Config
from .prompts import PromptFactory
from .store import RecordStore

_RETRY = object()  # take_slot 的“稍后重试”哨兵


class BenchmarkRunner:
    def __init__(self, cfg: Config, store: RecordStore, log: Optional[Callable[[str], None]] = None):
        self._cfg = cfg
        self._store = store
        self._log = log or (lambda msg: print(msg, flush=True))
        self._factory = PromptFactory(cfg.requests)
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ #
    async def run(self) -> List[metrics.StageStats]:
        cfg = self._cfg
        all_stats: List[metrics.StageStats] = []
        for index, concurrency in enumerate(cfg.load.concurrency_steps, start=1):
            self._log(
                f"[stage {index}/{len(cfg.load.concurrency_steps)}] 并发={concurrency} "
                f"持续={cfg.load.stage_duration_s}s（预热 {cfg.load.warmup_s}s）..."
            )
            stats = await self._run_stage(index, concurrency)
            all_stats.append(stats)
            self._log(
                f"  -> 样本={stats.n_total} A1={stats.a1} A2={stats.a2} A3={stats.a3} "
                f"SLA={stats.sla:.4f} SLO={stats.slo:.4f} "
                f"TTFT_p99={_fmt(stats.ttft_p99_ms)}ms TPOT_p99={_fmt(stats.tpot_p99_ms)}ms"
            )
        return all_stats

    # ------------------------------------------------------------------ #
    async def _run_stage(self, index: int, concurrency: int) -> metrics.StageStats:
        cfg = self._cfg
        store = self._store
        self._stop.clear()

        stage_id = store.start_stage(
            concurrency=concurrency,
            mode=cfg.load.mode,
            duration_s=cfg.load.stage_duration_s,
            warmup_s=cfg.load.warmup_s,
        )

        started = time.monotonic()
        warmup_deadline = started + cfg.load.warmup_s
        stage_deadline = started + cfg.load.stage_duration_s
        issued = {"measured": 0, "warmup": 0}

        async def take_slot() -> Optional[int]:
            """申请一个请求额度；返回 is_warmup 标记，None 表示应退出。"""
            if time.monotonic() >= stage_deadline:
                return None
            if time.monotonic() < warmup_deadline:
                if issued["warmup"] >= cfg.load.max_warmup_requests_per_stage:
                    await asyncio.sleep(0.05)  # 预热额度用尽，等待测量段开始
                    return None if time.monotonic() >= stage_deadline else _RETRY
                issued["warmup"] += 1
                return 1
            if issued["measured"] >= cfg.load.max_requests_per_stage:
                return None
            issued["measured"] += 1
            return 0

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
                prompt = self._factory.build(spec)
                result = await execute_request(
                    cfg.service, cfg.requests, prompt, spec,
                    stage_id=stage_id, concurrency=concurrency,
                    worker_id=worker_id, is_warmup=slot, mode=cfg.load.mode,
                )
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
                prompt = self._factory.build(spec)
                asyncio.create_task(
                    execute_request(
                        cfg.service, cfg.requests, prompt, spec,
                        stage_id=stage_id, concurrency=concurrency,
                        worker_id=0, is_warmup=slot, mode=cfg.load.mode,
                    )
                ).add_done_callback(lambda fut: _safe_add(store, fut, stage_id))
                if interval > 0:
                    await asyncio.sleep(interval)

        if cfg.load.mode == "closed":
            workers = [asyncio.create_task(closed_worker(i)) for i in range(concurrency)]
        else:
            workers = [asyncio.create_task(open_worker())]

        try:
            await asyncio.wait_for(self._stop.wait(), timeout=cfg.load.stage_duration_s)
        except asyncio.TimeoutError:
            pass
        self._stop.set()
        if workers:
            done, pending = await asyncio.wait(workers, timeout=cfg.load.drain_timeout_s)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        store.flush_requests()
        rows = store.stage_rows(stage_id, include_warmup=False)
        warmup_rows = store.warmup_count(stage_id)
        stats = metrics.compute_stage_stats(
            stage_id=stage_id,
            concurrency=concurrency,
            rows=rows,
            warmup_rows=warmup_rows,
            slo=cfg.slo,
            analysis=cfg.analysis,
            duration_s=cfg.load.stage_duration_s,
            warmup_s=cfg.load.warmup_s,
        )
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
