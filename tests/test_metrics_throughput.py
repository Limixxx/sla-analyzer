"""metrics 吞吐侧面与 a1_onset 的单元测试（docs/07 §7.10.2 / §7.10.3）。"""

from __future__ import annotations

import unittest

from helpers import make_config, make_stats, stage_rows  # noqa: E402

from bench.config import AnalysisConfig, SloConfig, WatermarkConfig  # noqa: E402
from bench.metrics import (  # noqa: E402
    CapacityFinding,
    compute_stage_stats,
    find_capacity,
    find_service_levels,
)


def _stats_from_rows(rows):
    return compute_stage_stats(
        stage_id=1,
        concurrency=8,
        rows=rows,
        warmup_rows=0,
        slo=SloConfig(),
        analysis=AnalysisConfig(),
        duration_s=60.0,
        warmup_s=10.0,
    )


class OkRpsInputTpsTest(unittest.TestCase):
    def test_basic_dual_counting(self):
        # 10 成功 + 2 失败，sent_at 步长 0.1s、e2e 0.1s：
        # 窗口 = (1.1 + 0.1) - 0 = 1.2s（最后一行 sent_at=1.1）
        rows = stage_rows(n_ok=10, n_fail=2, step_s=0.1, e2e_ms=100.0)
        s = _stats_from_rows(rows)
        self.assertEqual(s.n_total, 12)
        self.assertEqual(s.n_ok, 10)
        self.assertEqual(s.a1, 2)
        # rps 含失败；ok_rps 只计成功（fast-fail 不刷吞吐）
        self.assertAlmostEqual(s.rps, 12 / 1.2, places=6)
        self.assertAlmostEqual(s.ok_rps, 10 / 1.2, places=6)
        # 双 token 计数：input 按成功请求，output 失败贡献 0
        self.assertAlmostEqual(s.input_tps, 10 * 1000 / 1.2, places=6)
        self.assertAlmostEqual(s.output_tps, 10 * 100 / 1.2, places=6)
        # total 为读取时派生
        self.assertAlmostEqual(s.total_tps, s.input_tps + s.output_tps, places=6)

    def test_fast_fail_does_not_inflate_throughput(self):
        # 全部快速失败：rps 很高，ok_rps/input_tps/output_tps 全为 0
        rows = stage_rows(n_ok=0, n_fail=20, step_s=0.01, e2e_ms=1.0)
        s = _stats_from_rows(rows)
        self.assertEqual(s.n_ok, 0)
        window = 19 * 0.01 + 0.001  # 最后 sent_at=0.19 + e2e 1ms
        self.assertAlmostEqual(s.rps, 20 / window, places=6)
        self.assertAlmostEqual(s.ok_rps, 0.0, places=6)
        self.assertAlmostEqual(s.input_tps, 0.0, places=6)
        self.assertAlmostEqual(s.output_tps, 0.0, places=6)
        self.assertAlmostEqual(s.total_tps, 0.0, places=6)

    def test_sent_at_zero_is_a_valid_timestamp(self):
        # sent_at=0.0 是合法时刻（纪元起点），不得被真值判断当成缺失——
        # 回归：曾用 `if r.get("sent_at")` 把首行剔除，窗口整体右移
        rows = stage_rows(n_ok=5, n_fail=0, step_s=1.0, e2e_ms=100.0)
        self.assertEqual(rows[0]["sent_at"], 0.0)
        s = _stats_from_rows(rows)
        window = 4.0 + 0.1
        self.assertAlmostEqual(s.ok_rps, 5 / window, places=6)

    def test_window_uses_request_wallclock_not_nominal(self):
        # sent_at 跨 5s（远超名义测量窗口 50s 的一半），窗口必须按请求自身
        # 墙钟计算，否则低估速率
        rows = stage_rows(n_ok=5, n_fail=0, step_s=1.0, e2e_ms=100.0)
        s = _stats_from_rows(rows)
        window = 4.0 + 0.1  # 最后 sent_at=4.0，加 e2e 0.1
        self.assertAlmostEqual(s.ok_rps, 5 / window, places=6)

    def test_no_rows_returns_none_throughput(self):
        s = _stats_from_rows([])
        self.assertEqual(s.n_total, 0)
        self.assertIsNone(s.ok_rps)
        self.assertIsNone(s.input_tps)
        self.assertIsNone(s.total_tps)

    def test_total_tps_none_when_either_side_missing(self):
        s = make_stats(1, 8, 100, 100, 0, ok_rps=5.0, input_tps=None, output_tps=100.0)
        self.assertIsNone(s.total_tps)


class A1OnsetTest(unittest.TestCase):
    """a1_onset：容量膝点口径；默认 False 时在线行为一字不变。"""

    def _stages(self):
        # 低压全好；负载 32 出现 0.5% 的 A1（低于默认硬门限 1%）；
        # 负载 64 出现 20% 的 A1（超硬门限）
        return [
            make_stats(1, 8, 200, 200, 0, ok_rps=5.0, input_tps=5000.0, output_tps=1000.0),
            make_stats(2, 16, 200, 200, 0, ok_rps=9.0, input_tps=9000.0, output_tps=1800.0),
            make_stats(3, 32, 200, 199, 1, ok_rps=8.5, input_tps=8500.0, output_tps=1700.0,
                       fail_kinds={"timeout": 1}),
            make_stats(4, 64, 200, 160, 40, ok_rps=7.0, input_tps=7000.0, output_tps=1400.0,
                       fail_kinds={"http_5xx": 40}),
        ]

    def test_default_behavior_unchanged(self):
        # 默认（a1_onset=False）：可用性边界 = 首个 OVERLOAD 级 = 64
        # （0.5% 的 A1 不触硬门限，与增强前完全一致）
        stages = self._stages()
        finding = find_capacity(stages, AnalysisConfig())
        sl = find_service_levels(stages, finding, WatermarkConfig())
        self.assertEqual(sl.availability_boundary, 64)
        # 显式传 False 与默认等价
        sl_explicit = find_service_levels(stages, finding, WatermarkConfig(), a1_onset=False)
        self.assertEqual(sl_explicit.availability_boundary, 64)

    def test_a1_onset_takes_first_failure(self):
        # a1_onset=True：可用性边界 = 首个 A1 级 = 32（饱和起点）
        stages = self._stages()
        finding = find_capacity(stages, AnalysisConfig())
        sl = find_service_levels(stages, finding, WatermarkConfig(), a1_onset=True)
        self.assertEqual(sl.availability_boundary, 32)

    def test_a1_onset_note_when_no_failure(self):
        stages = [
            make_stats(1, 8, 200, 200, 0, ok_rps=5.0),
            make_stats(2, 16, 200, 200, 0, ok_rps=9.0),
        ]
        finding = find_capacity(stages, AnalysisConfig())
        sl = find_service_levels(stages, finding, WatermarkConfig(), a1_onset=True)
        self.assertIsNone(sl.availability_boundary)
        self.assertTrue(any("未出现 A1" in n for n in sl.notes))

    def test_no_samples_note(self):
        sl = find_service_levels([], CapacityFinding(
            sla_target=0.999, slo_target=0.95, min_samples=60), WatermarkConfig())
        self.assertTrue(any("没有任何带完成样本" in n for n in sl.notes))


if __name__ == "__main__":
    unittest.main()
