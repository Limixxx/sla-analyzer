"""架构隔离规则与在线路径零影响（docs/07 §7.10.7）。

两条硬约束的测试守护：
1. **无场景分支**：共享模块（runner/client/metrics/store/config/baseline/
   prompts/httpclient/stats/mlp/npq/qos/report）禁止 import 容量模块
   （report_batch / throughput）；bench.py 组合层是唯一例外。
2. **在线路径零影响**：默认配置（report.kinds=["online"]）下
   ``render_report`` 的输出与增强前逐字节一致——由 golden 文件锁定。
   有意变更在线渲染时必须显式重生成 golden（SLA_TEST_REGEN_GOLDEN=1），
   使"顺带改了在线输出"这类回归无法悄悄混过。
"""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from unittest import mock

from helpers import PROJECT_ROOT, make_config  # noqa: E402

from bench import report as report_mod  # noqa: E402
from bench.config import AnalysisConfig, ReportConfig, SloConfig, WatermarkConfig  # noqa: E402
from bench.metrics import (  # noqa: E402
    StageStats,
    classify_watermark,
    find_capacity,
)
from bench.report import render_report  # noqa: E402

# 共享模块：禁止出现任何容量场景 import（§7.10.7 规则 1）
SHARED_MODULES = (
    "runner.py", "client.py", "metrics.py", "store.py", "config.py",
    "baseline.py", "prompts.py", "httpclient.py", "stats.py", "mlp.py",
    "npz.py", "qos.py", "report.py",
)
FORBIDDEN_MODULES = {"report_batch", "throughput"}
GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "online_report.md"


def _imported_modules(source: str) -> List[str]:
    tree = ast.parse(source)
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


class ArchitectureIsolationTest(unittest.TestCase):
    def test_shared_modules_have_no_capacity_imports(self):
        for name in SHARED_MODULES:
            src = (PROJECT_ROOT / "bench" / name).read_text(encoding="utf-8")
            imported = set(_imported_modules(src))
            leaked = imported & FORBIDDEN_MODULES
            self.assertEqual(
                leaked, set(),
                f"{name} 违反了隔离规则：import 了容量模块 {sorted(leaked)}"
                "（场景差异只经配置与报告期表达，§7.10.7）",
            )

    def test_format_module_is_dependency_free(self):
        src = (PROJECT_ROOT / "bench" / "format.py").read_text(encoding="utf-8")
        for imported in _imported_modules(src):
            self.assertFalse(
                imported.startswith("bench") or imported.startswith("."),
                "format.py 应处于依赖图最底层（纯字符串变换）",
            )

    def test_composition_layer_knows_both_analyses(self):
        # 反空虚断言：bench.py 组合层确实同时知晓两种分析
        src = (PROJECT_ROOT / "bench.py").read_text(encoding="utf-8")
        self.assertIn("report_batch", src)
        self.assertIn("throughput", src)

    def test_report_batch_depends_on_online_side_only_downward(self):
        # 依赖方向：report_batch import 在线侧数据结构与吞吐模块，反之不成立
        src = (PROJECT_ROOT / "bench" / "report_batch.py").read_text(encoding="utf-8")
        imported = set(_imported_modules(src))
        self.assertIn("metrics", imported)
        self.assertIn("throughput", imported)
        report_src = (PROJECT_ROOT / "bench" / "report.py").read_text(encoding="utf-8")
        self.assertNotIn("throughput", _imported_modules(report_src))
        self.assertNotIn("report_batch", _imported_modules(report_src))


# ---------------------------------------------------------------------- #
# 在线报告 golden
# ---------------------------------------------------------------------- #
def _online_stages() -> List[StageStats]:
    """确定的在线场景合成数据：低压全好 → A3 出现 → A1 崩塌。"""
    raw = [
        # (conc, n, ok, a1, a3, ttft_p99, tpot_p99)
        (4, 60, 60, 0, 0, 300.0, 20.0),
        (8, 60, 60, 0, 0, 350.0, 22.0),
        (16, 60, 58, 0, 2, 2100.0, 45.0),   # A3：排队开始产生实质伤害
        (32, 60, 30, 30, 0, None, None),    # A1：服务接不住
    ]
    stages: List[StageStats] = []
    for i, (conc, n, ok, a1, a3, ttft_p99, tpot_p99) in enumerate(raw, start=1):
        s = StageStats(
            stage_id=i, concurrency=conc, duration_s=60.0, warmup_s=10.0,
            n_total=n, n_ok=ok, n_failed=n - ok, a1=a1, a2=ok - a1 - a3, a3=a3,
            slo=(ok - a1 - a3) / n, sla=(ok - a1) / n,
            ttft_p50_ms=200.0, ttft_p90_ms=280.0, ttft_p95_ms=290.0, ttft_p99_ms=ttft_p99,
            tpot_p50_ms=18.0, tpot_p90_ms=19.0, tpot_p95_ms=19.5, tpot_p99_ms=tpot_p99,
            e2e_p50_ms=1500.0, e2e_p99_ms=4000.0,
            connect_p50_ms=8.0, connect_p99_ms=15.0,
            rps=1.5 * conc / 4, output_tps=40.0 * conc / 4,
            ok_rps=1.4 * conc / 4, input_tps=1400.0 * conc / 4,
            prompt_tokens_avg=1000.0, completion_tokens_avg=80.0,
            fail_kinds={"http_5xx": a1} if a1 else {},
            ttft_target_ms=1000.0, tpot_target_ms=50.0,
            sla_target=0.999, slo_target=0.95,
        )
        state, reasons = classify_watermark(s, WatermarkConfig())
        s.watermark_state, s.watermark_reasons = state, reasons
        stages.append(s)
    return stages


def _render_online() -> str:
    cfg = make_config(kinds=["online"], output_path="results/golden.md")
    cfg.slo = SloConfig(target_mode="absolute")
    cfg.report = ReportConfig(output_path="results/golden.md")
    stages = _online_stages()
    finding = find_capacity(stages, AnalysisConfig())
    meta: Dict[str, str] = {
        "started_at": "2026-11-03 10:00:00",
        "db_path": "results/golden.db",
        "config": "{}",
        "tool_version": "test",
    }
    fake_time = SimpleNamespace(strftime=lambda fmt: "2026-11-03 10:30:00")
    with mock.patch.object(report_mod, "time", fake_time):
        return render_report(cfg, _NULL_STORE, stages, finding, meta)


class _NullStore:
    """render_report 在 qos=None 时不触碰 store；给个哨兵即可。"""


_NULL_STORE = _NullStore()


class OnlineGoldenTest(unittest.TestCase):
    """默认路径输出逐字节锁定（§7.10.7 在线零影响硬约束）。"""

    def test_matches_golden(self):
        if os.environ.get("SLA_TEST_REGEN_GOLDEN") == "1":
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(_render_online(), encoding="utf-8")
            self.skipTest("golden 已重生成（SLA_TEST_REGEN_GOLDEN=1）")
        self.assertTrue(GOLDEN_PATH.exists(),
                        f"golden 不存在: {GOLDEN_PATH}；首次生成请设 "
                        "SLA_TEST_REGEN_GOLDEN=1 运行一次")
        expected = GOLDEN_PATH.read_text(encoding="utf-8")
        actual = _render_online()
        if actual != expected:
            # 给出第一个差异行，便于定位是哪个章节被改
            exp_lines, act_lines = expected.splitlines(), actual.splitlines()
            for i, (e, a) in enumerate(zip(exp_lines, act_lines)):
                if e != a:
                    self.fail(
                        f"在线报告输出与 golden 不一致（首个差异在第 {i + 1} 行）：\n"
                        f"  golden : {e!r}\n  实际   : {a!r}\n"
                        "若为有意变更，请审阅 diff 后设 SLA_TEST_REGEN_GOLDEN=1 重生成。"
                    )
            self.fail(
                f"在线报告输出与 golden 不一致（行数 {len(act_lines)} vs "
                f"{len(exp_lines)}）"
            )


if __name__ == "__main__":
    unittest.main()
