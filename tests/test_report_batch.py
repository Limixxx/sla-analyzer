"""吞吐报告的渲染、容量读数与容量面汇写（docs/07 §7.10.6 / §7.10.5）。

覆盖：膝点读数（D21）、平台第二读数（D23）、(L_in, L_out) 坐标纪律
（D22）、数据质量闸门（D24）、超时预算同印（D25）。
"""

from __future__ import annotations

import unittest

from helpers import make_config, make_stats, ok_row, safe_rmtree, tmp_dir  # noqa: E402

from bench.metrics import find_capacity  # noqa: E402
from bench.report_batch import (  # noqa: E402
    _validate_stage_consistency,
    capacity_stage,
    find_knee_stage,
    knee_point,
    plateau_reading,
    render_report_batch,
)
from bench.store import RecordStore  # noqa: E402
from bench.throughput import ThroughputSurface  # noqa: E402


def ramp_stages():
    """标准 ramp：8/16 全好，32 首个 A1（0.5%），64 大量 A1。

    膝点 = 32；但 32 级 ok_rps 已回落（8.5 < 9.0），容量读数应取 16 级
    （峰值 9.0）——用于锁定"坍塌时取低压峰值"的诚实口径。
    """
    return [
        make_stats(1, 8, 200, 200, 0, ok_rps=5.0, input_tps=5000.0, output_tps=1000.0),
        make_stats(2, 16, 200, 200, 0, ok_rps=9.0, input_tps=9000.0, output_tps=1800.0),
        make_stats(3, 32, 200, 199, 1, ok_rps=8.5, input_tps=8500.0, output_tps=1700.0,
                   fail_kinds={"timeout": 1}),
        make_stats(4, 64, 200, 160, 40, ok_rps=7.0, input_tps=7000.0, output_tps=1400.0,
                   fail_kinds={"http_5xx": 40}),
    ]


class KneeAndCapacityReadingTest(unittest.TestCase):
    def test_knee_is_first_a1_level(self):
        knee, notes = find_knee_stage(ramp_stages())
        self.assertEqual(knee.concurrency, 32)
        self.assertEqual(notes, [])

    def test_no_knee_when_no_failure(self):
        stages = [make_stats(1, 8, 200, 200, 0, ok_rps=5.0),
                  make_stats(2, 16, 200, 200, 0, ok_rps=9.0)]
        knee, notes = find_knee_stage(stages)
        self.assertIsNone(knee)
        self.assertTrue(any("未出现 A1" in n for n in notes))

    def test_capacity_reading_takes_peak_before_knee(self):
        stages = ramp_stages()
        knee, _ = find_knee_stage(stages)
        cap = capacity_stage(stages, knee)
        # 32 级 ok_rps 已回落（8.5 < 9.0），峰值在 16 级
        self.assertEqual(cap.concurrency, 16)
        self.assertAlmostEqual(cap.ok_rps, 9.0, places=9)

    def test_capacity_reading_is_knee_when_no_collapse(self):
        # 排队型饱和：膝点级仍是峰值（ok_rps 不降）→ 读数即膝点级
        stages = [
            make_stats(1, 8, 200, 200, 0, ok_rps=5.0),
            make_stats(2, 16, 200, 200, 0, ok_rps=9.0),
            make_stats(3, 32, 200, 199, 1, ok_rps=9.2, fail_kinds={"timeout": 1}),
        ]
        knee, _ = find_knee_stage(stages)
        cap = capacity_stage(stages, knee)
        self.assertEqual(cap.concurrency, 32)

    def test_knee_point_uses_capacity_reading(self):
        stages = ramp_stages()
        knee, _ = find_knee_stage(stages)
        cap = capacity_stage(stages, knee)
        cfg = make_config(output_token_choices=[512])
        point = knee_point(cfg, knee, cap, source="run.db")
        self.assertEqual(point.input_len, 4096)          # 单长度配置
        self.assertEqual(point.output_len, 512)          # 单输出档配置（D22）
        self.assertEqual(point.origin, "knee")
        self.assertAlmostEqual(point.mu_rpm, 9.0 * 60.0, places=9)
        self.assertAlmostEqual(point.mu_tps, 10800.0, places=9)
        self.assertEqual(point.req_conc, 16)              # 取得峰值的负载水平
        self.assertEqual(point.n_samples, 200)
        self.assertEqual(point.a1_family, "timeout")      # 膝点级主导家族
        self.assertEqual(point.source, "run.db")


def plateau_stages():
    """软退化平台：无任何 A1，末两级吞吐走平（末级边际 +1.75% < 5%）。

    峰值在末级（8.14 rps @ 并发 80）——平台判定应给出"范围内饱和读数"。
    """
    return [
        make_stats(1, 16, 200, 200, 0, ok_rps=5.0, input_tps=5000.0, output_tps=1000.0),
        make_stats(2, 48, 200, 200, 0, ok_rps=8.0, input_tps=8000.0, output_tps=1600.0),
        make_stats(3, 80, 200, 200, 0, ok_rps=8.14, input_tps=8140.0, output_tps=1628.0),
    ]


class PlateauReadingTest(unittest.TestCase):
    """平台期第二读数（D23）：软退化服务的容量语言。"""

    def test_fires_on_flat_top(self):
        peak, notes = plateau_reading(plateau_stages())
        self.assertIsNotNone(peak)
        self.assertEqual(peak.concurrency, 80)
        self.assertTrue(any("非膝点读数" in n for n in notes))

    def test_fires_when_peak_one_level_below_top(self):
        # 顶部平坦但峰值落在倒数第二级（噪声容忍：差值带 ≤ 5%）
        stages = [
            make_stats(1, 8, 200, 200, 0, ok_rps=5.0),
            make_stats(2, 16, 200, 200, 0, ok_rps=8.0),
            make_stats(3, 32, 200, 200, 0, ok_rps=7.9),
        ]
        peak, _notes = plateau_reading(stages)
        self.assertIsNotNone(peak)
        self.assertEqual(peak.concurrency, 16)

    def test_not_fires_when_still_climbing(self):
        stages = [
            make_stats(1, 8, 200, 200, 0, ok_rps=5.0),
            make_stats(2, 16, 200, 200, 0, ok_rps=9.0),
            make_stats(3, 32, 200, 200, 0, ok_rps=11.0),
        ]
        peak, notes = plateau_reading(stages)
        self.assertIsNone(peak)
        self.assertTrue(any("仍在爬升" in n for n in notes))

    def test_not_fires_on_suspicious_drop(self):
        # 吞吐回落超 5% 却全程无 A1：可疑形态，拒绝给读数
        stages = [
            make_stats(1, 8, 200, 200, 0, ok_rps=5.0),
            make_stats(2, 16, 200, 200, 0, ok_rps=9.0),
            make_stats(3, 32, 200, 200, 0, ok_rps=6.0),
        ]
        peak, notes = plateau_reading(stages)
        self.assertIsNone(peak)
        self.assertTrue(any("可疑形态" in n for n in notes))

    def test_not_fires_with_single_level(self):
        peak, notes = plateau_reading([make_stats(1, 8, 200, 200, 0, ok_rps=5.0)])
        self.assertIsNone(peak)
        self.assertTrue(any("不足两级" in n for n in notes))


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.dir = tmp_dir("rep")
        self.addCleanup(safe_rmtree, self.dir)
        self.store = RecordStore(str(self.dir / "r.db"))
        self.addCleanup(self.store.close)
        self.meta = {"started_at": "2026-11-03 10:00:00", "db_path": str(self.dir / "r.db")}

    def _render(self, cfg, stages=None):
        stages = stages if stages is not None else ramp_stages()
        finding = find_capacity(stages, cfg.analysis)
        return render_report_batch(cfg, self.store, stages, finding, self.meta)

    def test_sections_and_key_numbers(self):
        cfg = make_config(kinds=["throughput"], output_token_choices=[512],
                          throughput_model_path=str(self.dir / "s.npz"))
        text, surface_path, _notes = self._render(cfg)
        # 头部：边界行 + 范围声明 + 配置快照 + 超时预算（D25）
        self.assertIn("测量边界：`http://127.0.0.1:8123`", text)
        self.assertIn("范围声明", text)
        self.assertIn("固定输入长度 4096", text)
        self.assertIn("超时预算", text)
        self.assertIn("服务拒绝", text)
        self.assertIn("固定输出长度 512", text)
        # 结论：SLA/SLO 与膝点
        self.assertIn("A1 拐点 = 负载 32", text)
        self.assertIn("容量读数 @ 负载 16", text)
        self.assertIn("SLA = 540.0 RPM", text)            # 9.0 rps × 60
        self.assertIn("SLO = 10,800 tokens/s", text)      # 9000 + 1800
        # 坍塌说明（诚实口径）
        self.assertIn("已受过载回落", text)
        # 容量表与 A1 归因
        self.assertIn("| 负载水平 | 样本数 | ok_rps（有效） |", text)
        self.assertIn("| 超时族 | 拒绝族 |", text)
        # 章节齐全
        for sec in ("## 一、结论", "## 二、容量表", "## 三、膝点证据",
                    "## 四、A1 归因", "## 五、建议", "## 六、方法与口径"):
            self.assertIn(sec, text)
        self.assertTrue(surface_path)

    def test_fast_fail_column_annotation(self):
        # rps 全量列必须在表内且被标注"含失败"
        cfg = make_config(throughput_model_path=str(self.dir / "s.npz"))
        text, _sp, _n = self._render(cfg)
        self.assertIn("rps（全量·含失败）", text)
        self.assertIn("全部快速失败时反而最高", text)

    def test_window_verdict_not_met(self):
        # 540 RPM < 100000/(1×60)×1.1 = 1833 → 未达标 + 缺口 + 重谈建议
        cfg = make_config(window_requests=100000, window_hours=1.0,
                          throughput_model_path=str(self.dir / "s.npz"))
        text, _sp, _n = self._render(cfg)
        self.assertIn("窗口承诺判定：**未达标**", text)
        self.assertIn("缺口", text)
        self.assertIn("重谈窗口承诺", text)

    def test_window_verdict_met(self):
        # 540 RPM ≥ 400/(1×60)×1.1 = 7.33 → 达标
        cfg = make_config(window_requests=400, window_hours=1.0,
                          throughput_model_path=str(self.dir / "s.npz"))
        text, _sp, _n = self._render(cfg)
        self.assertIn("窗口承诺判定：达标", text)
        self.assertIn("余量", text)

    def test_no_window_only_states_facts(self):
        cfg = make_config(throughput_model_path=str(self.dir / "s.npz"))
        text, _sp, _n = self._render(cfg)
        self.assertIn("未设窗口承诺", text)
        self.assertNotIn("窗口承诺判定", text)

    def test_token_budget_check_present(self):
        # token 预算核对需要单 (L_in, L_out) 坐标（D22），显式给单输出档
        cfg = make_config(output_token_choices=[512], window_requests=100000,
                          window_hours=1.0,
                          throughput_model_path=str(self.dir / "s.npz"))
        text, _sp, _n = self._render(cfg)
        self.assertIn("token 预算核对", text)

    def test_no_knee_report_degradation(self):
        stages = [make_stats(1, 8, 200, 200, 0, ok_rps=5.0),
                  make_stats(2, 16, 200, 200, 0, ok_rps=9.0)]
        cfg = make_config(throughput_model_path=str(self.dir / "s.npz"))
        text, surface_path, _n = self._render(cfg, stages)
        self.assertIn("未确定容量膝点", text)
        self.assertIn("提高梯度上限复测", text)
        self.assertIsNone(surface_path)  # 无膝点不写容量面

    def test_missing_token_data_degrades_rpm_only(self):
        # usage 缺失：input_tps 全 None → SLO 不可给，RPM 仍可陈述。
        # 容量读数取峰值（负载 8：5.0 rps），不是膝点级（负载 16：4.0）
        stages = [make_stats(1, 8, 200, 200, 0, ok_rps=5.0, input_tps=None, output_tps=None),
                  make_stats(2, 16, 200, 199, 1, ok_rps=4.0, input_tps=None, output_tps=None,
                             fail_kinds={"timeout": 1})]
        cfg = make_config(throughput_model_path=str(self.dir / "s.npz"))
        text, _sp, _n = self._render(cfg, stages)
        self.assertIn("SLA = 300.0 RPM", text)
        self.assertNotIn("SLO =", text)  # 不编造 TPS

    def test_surface_written_and_mergeable(self):
        cfg = make_config(output_token_choices=[512],
                          throughput_model_path=str(self.dir / "s.npz"))
        _t, sp, notes = self._render(cfg)
        self.assertTrue(sp)
        surface = ThroughputSurface.load(sp)
        self.assertEqual(len(surface.points), 1)
        self.assertEqual(surface.points[0].input_len, 4096)
        self.assertEqual(surface.points[0].output_len, 512)   # D22 坐标
        self.assertEqual(surface.points[0].origin, "knee")
        self.assertAlmostEqual(surface.points[0].mu_rpm, 540.0, places=9)
        self.assertEqual(surface.meta.get("base_url"), cfg.service.base_url)
        # 二次渲染（replace-per-cell）：仍是一个点
        _t2, _sp2, notes2 = self._render(cfg)
        self.assertEqual(len(ThroughputSurface.load(sp).points), 1)
        self.assertTrue(any("并入既有容量面" in n for n in notes2))

    def test_surface_refuses_cross_boundary_merge(self):
        cfg = make_config(output_token_choices=[512],
                          throughput_model_path=str(self.dir / "s.npz"))
        _t, sp, _n = self._render(cfg)
        # 另一个边界（不同 base_url）不允许混装同一文件
        other = make_config(base_url="http://other:9000", output_token_choices=[512],
                            throughput_model_path=str(self.dir / "s.npz"))
        _t2, sp2, notes2 = render_report_batch(
            other, self.store, ramp_stages(),
            find_capacity(ramp_stages(), other.analysis), self.meta)
        self.assertIsNone(sp2)
        self.assertTrue(any("不混装" in n for n in notes2))
        # 原文件未被改写
        self.assertEqual(len(ThroughputSurface.load(sp).points), 1)
        self.assertEqual(ThroughputSurface.load(sp).meta.get("base_url"),
                         "http://127.0.0.1:8123")

    def test_mixed_length_run_has_no_surface_point(self):
        # 混合输入长度：输入坐标不可定（不用均值近似，避免插值面失真）
        # → 只出报告，不入容量面（诚实降级）
        cfg = make_config(prompt_token_targets=[1024, 4096],
                          throughput_model_path=str(self.dir / "s.npz"))
        _t, sp, notes = self._render(cfg)
        self.assertIsNone(sp)
        self.assertTrue(any("坐标" in n for n in notes))

    def test_mixed_output_choices_have_no_surface_point(self):
        # 混合输出档（D22）：(L_in, L_out) 组合不可定 → 同样不入容量面
        cfg = make_config(output_token_choices=[128, 512],
                          throughput_model_path=str(self.dir / "s.npz"))
        _t, sp, notes = self._render(cfg)
        self.assertIsNone(sp)
        self.assertTrue(any("坐标" in n for n in notes))

    def test_plateau_report_renders_saturation_reading(self):
        # D23：无膝点但吞吐走平 → 范围内饱和读数 + 容量面 origin=plateau
        cfg = make_config(output_token_choices=[512],
                          throughput_model_path=str(self.dir / "s.npz"))
        text, sp, _notes = self._render(cfg, plateau_stages())
        self.assertIn("范围内饱和读数", text)
        self.assertIn("非膝点", text)
        self.assertIn("软退化平台估计", text)
        self.assertIn("超时预算", text)          # D25：平台读数同样与 timeout_s 同印
        self.assertIsNotNone(sp)
        s = ThroughputSurface.load(sp)
        self.assertEqual(len(s.points), 1)
        p = s.points[0]
        self.assertEqual(p.origin, "plateau")
        self.assertEqual(p.a1_family, "none")
        self.assertEqual(p.input_len, 4096)
        self.assertEqual(p.output_len, 512)
        self.assertAlmostEqual(p.mu_rpm, 8.14 * 60.0, places=6)
        self.assertEqual(p.req_conc, 80)

    def test_plateau_still_degrades_when_climbing(self):
        # 无膝点且末级仍在爬升：不给平台读数，结论说明两条排除理由
        stages = [
            make_stats(1, 8, 200, 200, 0, ok_rps=5.0),
            make_stats(2, 16, 200, 200, 0, ok_rps=9.0),
            make_stats(3, 32, 200, 200, 0, ok_rps=11.0),
        ]
        cfg = make_config(throughput_model_path=str(self.dir / "s.npz"))
        text, sp, _notes = self._render(cfg, stages)
        self.assertIn("未确定容量膝点", text)
        self.assertIn("仍在爬升", text)
        self.assertIsNone(sp)

    def test_diagnostics_rendered_for_polluted_stage(self):
        # D24：Little's Law 异常级 → 吞吐报告出现诊断行（只警告不阻断）
        # 行构造：40 行均匀 e2e=1.04s；rps 人为置 6.0 → rps×E2E=6.24，
        # 与并发 8 偏差 22% > 15%
        rows = [
            ok_row(i * 0.1, prompt_tokens=1000, completion_tokens=100,
                   e2e_ms=1040.0, ttft_ms=50.0, tpot_ms=10.0)
            for i in range(40)
        ]
        for i, r in enumerate(rows):
            row = dict(r)
            row.update(stage_id=1, concurrency=8, worker_id=i, is_warmup=0,
                       mode="closed", token_source="usage", finish_reason="stop",
                       note=None)
            self.store.add_request(row)
        self.store.flush_requests()
        stats = make_stats(1, 8, 40, 40, 0, rps=6.0, ok_rps=7.0,
                           input_tps=7000.0, output_tps=1400.0)
        cfg = make_config(throughput_model_path=str(self.dir / "s.npz"))
        finding = find_capacity([stats], cfg.analysis)
        text, _sp, _n = render_report_batch(cfg, self.store, [stats], finding, self.meta)
        self.assertIn("数据质量诊断（D24", text)
        self.assertIn("Little's Law", text)


def _uniform_rows(n=40, e2e_ms=1040.0, ttft_ms=50.0, tpot_ms=10.0, out=100, step_s=0.1):
    """恒等式自洽的行集：e2e = ttft + tpot×(out−1) = 50 + 10×99 = 1040。"""
    return [
        ok_row(i * step_s, prompt_tokens=1000, completion_tokens=out,
               e2e_ms=e2e_ms, ttft_ms=ttft_ms, tpot_ms=tpot_ms)
        for i in range(n)
    ]


class ConsistencyGateTest(unittest.TestCase):
    """数据质量闸门（D24）：Little's Law 与时延恒等式，只警告不阻断。"""

    def setUp(self):
        self.cfg = make_config()

    def test_clean_rows_pass_both_gates(self):
        # e2e 均值 1.04s，rps = 8/1.04 → rps×E2E = 8 = 并发（Little's 达标）；
        # 恒等式残差为 0
        rows = _uniform_rows()
        stats = make_stats(1, 8, 40, 40, 0, rps=8.0 / 1.04)
        self.assertEqual(_validate_stage_consistency(self.cfg, stats, rows), [])

    def test_littles_law_violation_flagged(self):
        rows = _uniform_rows()
        stats = make_stats(1, 8, 40, 40, 0, rps=6.0)   # lhs=6.24 vs 8 → 偏差 22%
        diags = _validate_stage_consistency(self.cfg, stats, rows)
        self.assertTrue(any("Little's Law" in d for d in diags))
        self.assertFalse(any("时延恒等式" in d for d in diags))

    def test_latency_identity_violation_flagged(self):
        # e2e=1500 但 ttft+tpot×99=1040：残差 460/1500 ≈ 30.7%；
        # rps 同步取 8/1.5 使 Little's Law 保持达标——两闸门独立
        rows = _uniform_rows(e2e_ms=1500.0)
        stats = make_stats(1, 8, 40, 40, 0, rps=8.0 / 1.5)
        diags = _validate_stage_consistency(self.cfg, stats, rows)
        self.assertTrue(any("时延恒等式" in d for d in diags))
        self.assertFalse(any("Little's Law" in d for d in diags))

    def test_littles_law_skipped_with_think_time(self):
        # think_time 改变闭环守恒式（应含思考时间），跳过而非误报
        rows = _uniform_rows()
        stats = make_stats(1, 8, 40, 40, 0, rps=6.0)
        self.cfg.load.think_time_s = 2.0
        self.assertEqual(_validate_stage_consistency(self.cfg, stats, rows), [])

    def test_littles_law_skipped_with_aborted(self):
        # aborted 计入窗口但无 E2E，守恒式不适用，跳过而非误报
        rows = _uniform_rows()
        stats = make_stats(1, 8, 42, 40, 0, rps=6.0)
        stats.n_aborted = 2
        self.assertEqual(_validate_stage_consistency(self.cfg, stats, rows), [])

    def test_missing_rows_reported(self):
        stats = make_stats(1, 8, 40, 40, 0, rps=8.0)
        diags = _validate_stage_consistency(self.cfg, stats, [])
        self.assertEqual(len(diags), 1)
        self.assertIn("请求明细缺失", diags[0])

    def test_too_few_rows_not_judged(self):
        rows = _uniform_rows(n=3)
        stats = make_stats(1, 8, 3, 3, 0, rps=8.0)
        self.assertEqual(_validate_stage_consistency(self.cfg, stats, rows), [])


if __name__ == "__main__":
    unittest.main()
