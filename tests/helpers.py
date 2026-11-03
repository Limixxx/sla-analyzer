"""测试公共构造器：合成请求行 / StageStats / Config。

只用标准库 unittest（AGENT.md：本机无 pytest/CI）。所有合成数据确定性
（固定 sent_at / 固定 token 数），使断言可以精确到小数点后两位。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# 让测试在本仓库根目录下以 `python -m unittest discover -s tests` 运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.config import (  # noqa: E402
    AnalysisConfig,
    Config,
    LoadConfig,
    ReportConfig,
    RequestConfig,
    ServiceConfig,
    SloConfig,
    StorageConfig,
    WatermarkConfig,
)
from bench.metrics import StageStats  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ok_row(sent_at: float, prompt_tokens: int = 1000, completion_tokens: int = 100,
           e2e_ms: float = 100.0, ttft_ms: float = 50.0, tpot_ms: float = 10.0,
           latency_class: str = "A2") -> Dict[str, Any]:
    """一条成功请求的事实行（latency_class 显式给定，绕过 classify）。"""
    return {
        "status": "ok",
        "sent_at": sent_at,
        "connect_ms": 5.0,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "prompt_tokens_planned": prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "max_tokens": completion_tokens,
        "error_kind": None,
        "http_code": 200,
        "latency_class": latency_class,
    }


def fail_row(sent_at: float, error_kind: str = "http_5xx", e2e_ms: float = 5.0) -> Dict[str, Any]:
    """一条失败请求的事实行（A1；无 token 产出，completion 贡献 0）。"""
    return {
        "status": "failed",
        "sent_at": sent_at,
        "connect_ms": 5.0,
        "ttft_ms": None,
        "tpot_ms": None,
        "e2e_ms": e2e_ms,
        "prompt_tokens_planned": 1000,
        "prompt_tokens": None,
        "completion_tokens": None,
        "max_tokens": 100,
        "error_kind": error_kind,
        "http_code": 503,
        "latency_class": "A1",
    }


def stage_rows(n_ok: int, n_fail: int, step_s: float = 0.1, e2e_ms: float = 100.0,
               prompt_tokens: int = 1000, completion_tokens: int = 100,
               fail_kind: str = "http_5xx") -> List[Dict[str, Any]]:
    """一组的请求行：成功与失败交替（某一侧为 0 时另一侧独占），sent_at
    以 step_s 均匀铺开。

    时间窗 = 最后一个 (sent_at + e2e) - 第一个 sent_at，因此
    ok_rps = n_ok / window 可手算校验。
    """
    oks = [
        ok_row(0.0, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
               e2e_ms=e2e_ms)
        for _ in range(n_ok)
    ]
    fails = [fail_row(0.0, error_kind=fail_kind, e2e_ms=e2e_ms) for _ in range(n_fail)]
    rows: List[Dict[str, Any]] = []
    while oks or fails:
        if oks:
            rows.append(oks.pop(0))
        if fails:
            rows.append(fails.pop(0))
    for i, r in enumerate(rows):
        r["sent_at"] = i * step_s
    return rows


def make_stats(
    stage_id: int,
    concurrency: int,
    n_total: int,
    n_ok: int,
    a1: int,
    a3: int = 0,
    ok_rps: Optional[float] = None,
    input_tps: Optional[float] = None,
    output_tps: Optional[float] = None,
    rps: Optional[float] = None,
    prompt_tokens_avg: Optional[float] = 1000.0,
    completion_tokens_avg: Optional[float] = 100.0,
    fail_kinds: Optional[Dict[str, int]] = None,
) -> StageStats:
    """直接构造 StageStats（绕开 compute_stage_stats，用于报告/查找器测试）。"""
    s = StageStats(
        stage_id=stage_id,
        concurrency=concurrency,
        duration_s=60.0,
        warmup_s=10.0,
        n_total=n_total,
        n_ok=n_ok,
        n_failed=n_total - n_ok,
        a1=a1,
        a2=n_ok - a1 - a3,
        a3=a3,
        slo=(n_ok - a1 - a3) / n_total if n_total else 0.0,
        sla=(n_ok - a1) / n_total if n_total else 0.0,
        ok_rps=ok_rps,
        input_tps=input_tps,
        output_tps=output_tps,
        rps=rps,
        prompt_tokens_avg=prompt_tokens_avg,
        completion_tokens_avg=completion_tokens_avg,
        fail_kinds=dict(fail_kinds or ({"http_5xx": a1} if a1 else {})),
        sla_target=0.999,
        slo_target=0.95,
        ttft_target_ms=2000.0,
        tpot_target_ms=50.0,
        watermark_state="unknown",
    )
    return s


def make_config(
    base_url: str = "http://127.0.0.1:8123",
    kinds: Optional[List[str]] = None,
    window_requests: int = 0,
    window_hours: float = 0.0,
    window_safety_factor: float = 1.1,
    prompt_token_targets: Optional[Sequence[int]] = None,
    concurrency_steps: Optional[Sequence[int]] = None,
    throughput_model_path: str = "",
    output_path: str = "results/_test_report.md",
) -> Config:
    """一份最小但合法的 Config（不走 TOML，直接构造 dataclass）。"""
    return Config(
        service=ServiceConfig(base_url=base_url),
        slo=SloConfig(),
        load=LoadConfig(concurrency_steps=sorted(set(concurrency_steps or [2, 4, 8, 16]))),
        requests=RequestConfig(
            prompt_token_targets=list(prompt_token_targets or [4096]),
            prompt_weights=[1.0 / len(prompt_token_targets or [4096])] * len(prompt_token_targets or [4096]),
        ),
        analysis=AnalysisConfig(
            window_requests=window_requests,
            window_hours=window_hours,
            window_safety_factor=window_safety_factor,
        ),
        storage=StorageConfig(),
        report=ReportConfig(
            output_path=output_path,
            kinds=list(kinds or ["online"]),
            throughput_model_path=throughput_model_path,
        ),
    )


def write_toml(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def tmp_dir(name: str) -> Path:
    """测试用临时目录（放系统临时区，退出时由调用方清理）。"""
    import tempfile

    d = Path(tempfile.mkdtemp(prefix=f"sla_t_{name}_"))
    return d


def safe_rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def env_path(name: str) -> str:
    return os.environ.get(name, "")


def load_cli_module():
    """按文件路径加载顶层 bench.py（CLI 组合层）。

    它被同名的 ``bench/`` 包遮蔽（``import bench`` 拿到的是包），
    因此测试里用 importlib 按路径加载，模块名用 ``bench_cli`` 区分。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("bench_cli", PROJECT_ROOT / "bench.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod
