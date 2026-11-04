"""容量面模型：fit/merge/query/save/load（docs/07 §7.10.5）。

重点锁定四条纪律：
1. 往返无损（含 NaN↔None 的缺失值语义、a1_family/origin 整数编码）；
2. 外插必须显式（clamp + extrapolated 标记），对应在线模型的包络守卫；
3. 缺失不编造（任一侧 token 数据缺失 → 整体缺失 + 说明；角点未实测 →
   回退最近点 + 说明，不静默插值）；
4. 二维坐标 (L_in, L_out)（D22）：merge 按 cell 覆盖、query 双线性插值，
   旧版一维面（无 output_lens/origins 数组）读回退化为 1D 行为。
"""

from __future__ import annotations

import json
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

    def test_merge_replace_per_cell(self):
        s = ThroughputSurface.fit(_two_points())
        replaced = s.merge([ThroughputPoint(2048, 450.0, 10000.0, 6000.0, 4000.0, 12, 100)])
        self.assertEqual(replaced, [(2048, 0)])   # merge 返回 (L_in, L_out) 坐标
        self.assertEqual([p.input_len for p in s.points], [1024, 2048, 4096])
        # 再合并同长度：覆盖
        s.merge([ThroughputPoint(2048, 480.0, 10500.0, 6300.0, 4200.0, 13, 110)])
        self.assertEqual(len(s.points), 3)
        self.assertEqual(s.points[1].mu_rpm, 480.0)

    def test_merge_keeps_distinct_output_cells(self):
        # D22：同输入长度、不同输出档是两个 cell，共存不互覆
        s = ThroughputSurface.fit([
            ThroughputPoint(1024, 600.0, 12000.0, 8000.0, 4000.0, 8, 120, output_len=128),
        ])
        replaced = s.merge([
            ThroughputPoint(1024, 350.0, 9000.0, 5000.0, 4000.0, 10, 90, output_len=512),
        ])
        self.assertEqual(replaced, [(1024, 512)])
        self.assertEqual(
            [(p.output_len, p.mu_rpm) for p in s.points], [(128, 600.0), (512, 350.0)]
        )
        # 同 cell 再合并：覆盖
        s.merge([ThroughputPoint(1024, 360.0, 9100.0, 5100.0, 4000.0, 11, 95, output_len=512)])
        self.assertEqual(len(s.points), 2)
        self.assertEqual([p for p in s.points if p.output_len == 512][0].mu_rpm, 360.0)


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

    def test_roundtrip_origin_and_output_len(self):
        # D22/D23：output_len 与 origin（读数来源）随面往返无损
        s = ThroughputSurface.fit([
            ThroughputPoint(2048, 500.0, 11000.0, 7000.0, 4000.0, 12, 100,
                            output_len=256, origin="plateau"),
        ])
        s.save(self.path)
        s2 = ThroughputSurface.load(self.path)
        p = s2.points[0]
        self.assertEqual(p.output_len, 256)
        self.assertEqual(p.origin, "plateau")
        self.assertIn("origin_labels", s2.meta)

    def test_load_legacy_surface_without_output_arrays(self):
        # 旧版一维面（无 output_lens/origins 数组）：守卫读回 0/"knee"，
        # 查询退化为按输入长度的一维插值
        s = ThroughputSurface.fit(_two_points())
        s.save(self.path)
        from bench.npz import load_npz, save_npz

        data = load_npz(self.path)
        data.pop("output_lens", None)
        data.pop("origins", None)
        save_npz(self.path, data)
        s2 = ThroughputSurface.load(self.path)
        self.assertTrue(all(p.output_len == 0 and p.origin == "knee" for p in s2.points))
        a = s2.query(500.0, 1024)   # 1D 路径：正常给读数，不要求 --output-len
        self.assertAlmostEqual(a.mu_rpm, 600.0, places=9)
        self.assertEqual(a.effective_output_len, 0)

    def test_load_rejects_output_lens_length_mismatch(self):
        s = ThroughputSurface.fit(_two_points())
        s.save(self.path)
        from bench.npz import load_npz, save_npz

        data = load_npz(self.path)
        data["output_lens"] = [8]     # 与 lengths 长度不一致
        save_npz(self.path, data)
        with self.assertRaises(ThroughputError):
            ThroughputSurface.load(self.path)

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


def _grid_points():
    """满 2×2 网格：x∈{1024, 4096} × y∈{128, 512}，四角 mu_rpm 各异。"""
    return [
        ThroughputPoint(1024, 600.0, 12000.0, 8000.0, 4000.0, 8, 120, "reject", "a.db", 128),
        ThroughputPoint(1024, 500.0, 11000.0, 7000.0, 4000.0, 10, 110, "none", "b.db", 512),
        ThroughputPoint(4096, 300.0, 9000.0, 7000.0, 2000.0, 16, 90, "timeout", "c.db", 128),
        ThroughputPoint(4096, 250.0, 8500.0, 6500.0, 2000.0, 18, 80, "none", "d.db", 512),
    ]


class Query2DTest(unittest.TestCase):
    """二维面查询（D22）：双线性插值、未指定输出档、外插、退化轴、角点回退。"""

    def setUp(self):
        self.surface = ThroughputSurface.fit(_grid_points())

    def test_bilinear_interior(self):
        # x=2048（tx=1/3）、y=320（ty=0.5）：
        # y=128 行沿 x → 600−300/3=500；y=512 行 → 500−250/3≈416.667；
        # 沿 y 取中点 → 458.333；req_conc = (10.667+12.667)/2 → 12
        a = self.surface.query(400.0, 2048, 320)
        self.assertAlmostEqual(a.mu_rpm, 458.333333, places=6)
        self.assertEqual(a.req_conc, 12)
        self.assertFalse(a.extrapolated)
        self.assertEqual(a.effective_len, 2048)
        self.assertEqual(a.effective_output_len, 320)
        # token 侧四角齐全 → 双线性
        self.assertAlmostEqual(a.mu_tps, (12000 - 3000 / 3 + 11000 - 2500 / 3) / 2, places=6)

    def test_exact_node_returns_point_values(self):
        a = self.surface.query(100.0, 4096, 512)
        self.assertAlmostEqual(a.mu_rpm, 250.0, places=9)
        self.assertEqual(a.a1_family, "none")
        self.assertEqual(a.origin, "knee")
        self.assertEqual(a.req_conc, 18)

    def test_exact_node_on_sparse_grid_needs_no_full_corners(self):
        # 查询点恰在实测节点上：另一侧角点缺失不影响读数（权重为零）
        sparse = ThroughputSurface.fit([
            ThroughputPoint(1024, 600.0, 12000.0, 8000.0, 4000.0, 8, 120, output_len=128),
            ThroughputPoint(4096, 300.0, 9000.0, 7000.0, 2000.0, 16, 90, output_len=512),
        ])
        a = sparse.query(100.0, 4096, 512)
        self.assertAlmostEqual(a.mu_rpm, 300.0, places=9)
        self.assertFalse(any("最近实测点" in n for n in a.notes))

    def test_unspecified_output_takes_min_and_notes(self):
        a = self.surface.query(100.0, 1024)
        self.assertEqual(a.effective_output_len, 128)
        self.assertAlmostEqual(a.mu_rpm, 600.0, places=9)
        self.assertTrue(any("未指定 --output-len" in n for n in a.notes))

    def test_clamp_both_axes_flags_extrapolated(self):
        a = self.surface.query(100.0, 1, 99999)
        self.assertTrue(a.extrapolated)
        self.assertEqual(a.effective_len, 1024)
        self.assertEqual(a.effective_output_len, 512)
        self.assertTrue(any("超出测量范围" in n for n in a.notes))
        self.assertTrue(any("clamp" in n for n in a.notes))

    def test_degenerate_x_axis_interpolates_along_y(self):
        s = ThroughputSurface.fit([
            ThroughputPoint(2048, 600.0, 12000.0, 8000.0, 4000.0, 8, 120, output_len=128),
            ThroughputPoint(2048, 200.0, 9000.0, 6000.0, 3000.0, 14, 90, output_len=512),
        ])
        a = s.query(100.0, 2048, 320)
        self.assertAlmostEqual(a.mu_rpm, 400.0, places=9)
        self.assertEqual(a.req_conc, 11)   # 8 与 14 的中点
        self.assertFalse(a.extrapolated)

    def test_missing_interior_corner_falls_back_to_nearest(self):
        # 只有对角两点，查询严格在面内：缺角 → 回退最近实测点 + 显式说明
        sparse = ThroughputSurface.fit([
            ThroughputPoint(1024, 600.0, 12000.0, 8000.0, 4000.0, 8, 120, output_len=128),
            ThroughputPoint(4096, 300.0, 9000.0, 7000.0, 2000.0, 16, 90, output_len=512),
        ])
        a = sparse.query(100.0, 2048, 512)
        self.assertAlmostEqual(a.mu_rpm, 300.0, places=9)
        self.assertEqual(a.effective_len, 4096)
        self.assertEqual(a.effective_output_len, 512)
        self.assertTrue(any("最近实测点" in n for n in a.notes))

    def test_origin_and_family_from_nearest_corner(self):
        # 离散标签不插值：取归一化距离最近的角点
        s = ThroughputSurface.fit([
            ThroughputPoint(1024, 600.0, 12000.0, 8000.0, 4000.0, 8, 120,
                            "none", "a.db", 128, "plateau"),
            ThroughputPoint(1024, 500.0, 11000.0, 7000.0, 4000.0, 10, 110,
                            "timeout", "b.db", 512),
            ThroughputPoint(4096, 300.0, 9000.0, 7000.0, 2000.0, 16, 90,
                            "none", "c.db", 128),
            ThroughputPoint(4096, 250.0, 8500.0, 6500.0, 2000.0, 18, 80,
                            "none", "d.db", 512),
        ])
        # x=2048, y=320：tx=1/3、ty=0.5 → 最近角点 (0,0) 与 (0,1) 距离并列，
        # 确定性 tie-break 取前者（origin=plateau）
        a = s.query(100.0, 2048, 320)
        self.assertEqual(a.origin, "plateau")
        self.assertEqual(a.a1_family, "none")

    def test_1d_surface_with_output_len_notes_ignored(self):
        a = ThroughputSurface.fit(_two_points()).query(100.0, 1024, 512)
        self.assertAlmostEqual(a.mu_rpm, 600.0, places=9)
        self.assertTrue(any("未记录输出长度维度" in n for n in a.notes))


if __name__ == "__main__":
    unittest.main()
