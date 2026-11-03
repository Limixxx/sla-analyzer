"""容量面模型：fit/merge/query/save/load（docs/07 §7.10.5）。

重点锁定三条纪律：
1. 往返无损（含 NaN↔None 的缺失值语义、a1_family 整数编码）；
2. 外插必须显式（clamp + extrapolated 标记），对应在线模型的包络守卫；
3. 缺失不编造（任一侧 token 数据缺失 → 整体缺失 + 说明）。
"""

from __future__ import annotations

import unittest

from helpers import safe_rmtree, tmp_dir  # noqa: E402

from bench.throughput import (  # noqa: E402
    SURFACE_KIND,
    ThroughputError,
    ThroughputPoint,
    ThroughputSurface,
)


def _two_points():
    return [
        ThroughputPoint(1024, 600.0, 12000.0, 8000.0, 4000.0, 8, 120, "reject", "a.db"),
        ThroughputPoint(4096, 300.0, 9000.0, 7000.0, 2000.0, 16, 90, "timeout", "b.db"),
    ]


class FitMergeTest(unittest.TestCase):
    def test_fit_sorts_and_dedups(self):
        s = ThroughputSurface.fit([
            ThroughputPoint(4096, 300.0, 9000.0, 7000.0, 2000.0, 16, 90),
            ThroughputPoint(1024, 600.0, 12000.0, 8000.0, 4000.0, 8, 120),
            ThroughputPoint(1024, 650.0, 13000.0, 8500.0, 4500.0, 10, 130),  # 同长度后者覆盖
        ])
        self.assertEqual([p.input_len for p in s.points], [1024, 4096])
        self.assertEqual(s.points[0].mu_rpm, 650.0)

    def test_merge_replace_per_length(self):
        s = ThroughputSurface.fit(_two_points())
        replaced = s.merge([ThroughputPoint(2048, 450.0, 10000.0, 6000.0, 4000.0, 12, 100)])
        self.assertEqual(replaced, [2048])
        self.assertEqual([p.input_len for p in s.points], [1024, 2048, 4096])
        # 再合并同长度：覆盖
        s.merge([ThroughputPoint(2048, 480.0, 10500.0, 6300.0, 4200.0, 13, 110)])
        self.assertEqual(len(s.points), 3)
        self.assertEqual(s.points[1].mu_rpm, 480.0)


class SaveLoadTest(unittest.TestCase):
    def setUp(self):
        self.dir = tmp_dir("surf")
        self.addCleanup(safe_rmtree, self.dir)
        self.path = str(self.dir / "s.npz")

    def test_roundtrip_lossless(self):
        s = ThroughputSurface.fit(_two_points(), {"base_url": "http://x:8000"})
        s.save(self.path)
        s2 = ThroughputSurface.load(self.path)
        self.assertEqual(s2.meta["kind"], SURFACE_KIND)
        self.assertEqual(s2.meta["base_url"], "http://x:8000")
        self.assertIn("updated_at", s2.meta)
        self.assertEqual([p.input_len for p in s2.points], [1024, 4096])
        p0, p1 = s2.points
        self.assertAlmostEqual(p0.mu_rpm, 600.0, places=9)
        self.assertAlmostEqual(p0.mu_tps, 12000.0, places=9)
        self.assertAlmostEqual(p0.input_tps, 8000.0, places=9)
        self.assertAlmostEqual(p0.output_tps, 4000.0, places=9)
        self.assertEqual((p0.req_conc, p0.n_samples, p0.a1_family), (8, 120, "reject"))
        self.assertEqual(p1.a1_family, "timeout")

    def test_roundtrip_none_token_values(self):
        # 缺失 token 数据：存 NaN，读回为 None（不静默填 0）
        s = ThroughputSurface.fit([
            ThroughputPoint(1024, 600.0, None, None, None, 8, 120),
        ])
        s.save(self.path)
        s2 = ThroughputSurface.load(self.path)
        p = s2.points[0]
        self.assertIsNone(p.mu_tps)
        self.assertIsNone(p.input_tps)
        self.assertIsNone(p.output_tps)
        self.assertAlmostEqual(p.mu_rpm, 600.0, places=9)

    def test_load_rejects_wrong_kind(self):
        s = ThroughputSurface.fit(_two_points())
        s.save(self.path)
        # 篡改 meta.kind 后应被拒绝
        import json

        from bench.npz import load_npz, save_npz

        data = load_npz(self.path)
        meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
        meta["kind"] = "something_else"
        data["meta_json"] = json.dumps(meta).encode("utf-8")
        save_npz(self.path, data)
        with self.assertRaises(ThroughputError):
            ThroughputSurface.load(self.path)

    def test_load_rejects_missing_arrays(self):
        from bench.npz import save_npz

        save_npz(self.path, {"lengths": [1024]})
        with self.assertRaises(ThroughputError):
            ThroughputSurface.load(self.path)

    def test_load_missing_file(self):
        with self.assertRaises(ThroughputError):
            ThroughputSurface.load(str(self.dir / "nope.npz"))


class QueryTest(unittest.TestCase):
    def setUp(self):
        self.surface = ThroughputSurface.fit(_two_points())

    def test_exact_length(self):
        a = self.surface.query(400.0, 1024)
        self.assertEqual(a.effective_len, 1024)
        self.assertFalse(a.extrapolated)
        self.assertAlmostEqual(a.mu_rpm, 600.0, places=9)
        self.assertEqual(a.req_conc, 8)
        self.assertTrue(a.achievable)
        self.assertAlmostEqual(a.margin_rpm, 200.0, places=9)
        self.assertEqual(a.a1_family, "reject")

    def test_interpolation(self):
        a = self.surface.query(250.0, 2048)
        self.assertEqual(a.effective_len, 2048)
        # 2048 在 1024 与 4096 间的 1/3 处：mu=500，conc=8+8/3≈11，
        # mu_tps=12000+(9000-12000)/3=11000
        self.assertAlmostEqual(a.mu_rpm, 500.0, places=9)
        self.assertEqual(a.req_conc, 11)
        self.assertAlmostEqual(a.mu_tps, 11000.0, places=9)

    def test_extrapolation_clamps_and_flags(self):
        a = self.surface.query(100.0, 99999)
        self.assertTrue(a.extrapolated)
        self.assertEqual(a.effective_len, 4096)
        self.assertTrue(any("clamp" in n for n in a.notes))
        # 低端外插同样 clamp
        b = self.surface.query(100.0, 1)
        self.assertTrue(b.extrapolated)
        self.assertEqual(b.effective_len, 1024)

    def test_unachievable_states_gap(self):
        a = self.surface.query(5000.0, 4096)
        self.assertFalse(a.achievable)
        self.assertAlmostEqual(a.margin_rpm, 300.0 - 5000.0, places=9)
        self.assertTrue(any("超出" in n for n in a.notes))

    def test_missing_token_data_note(self):
        s = ThroughputSurface.fit([ThroughputPoint(1024, 600.0, None, None, None, 8, 120)])
        a = s.query(100.0, 1024)
        self.assertIsNone(a.mu_tps)
        self.assertTrue(any("缺少完整 token 数据" in n for n in a.notes))

    def test_empty_surface(self):
        a = ThroughputSurface.fit([]).query(100.0, 1024)
        self.assertIsNone(a.mu_rpm)
        self.assertTrue(any("容量面为空" in n for n in a.notes))


if __name__ == "__main__":
    unittest.main()
