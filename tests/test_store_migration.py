"""stages 表吞吐两列的迁移、落库与守卫读回（docs/07 §7.7 / §7.10.2）。

覆盖三条路径：
1. 新库：SCHEMA 直接含 ok_rps/input_tps，finish_stage 原样落库；
2. 旧库（无两列）：_migrate 补齐；--report-only 的 _stats_from_store
   按守卫读回，缺列时用充分统计量推导兜底（ok_rps=rps×n_ok/n_total，
   input_tps=prompt_tokens_avg×ok_rps），不静默填 0；
3. 两条路径口径一致：实时落库值 = report-only 读回值。
"""

from __future__ import annotations

import sqlite3
import unittest

from helpers import load_cli_module, make_config, make_stats, safe_rmtree, tmp_dir  # noqa: E402

from bench.store import RecordStore  # noqa: E402

_cli = load_cli_module()
_derive_throughput_fallback = _cli._derive_throughput_fallback
_stats_from_store = _cli._stats_from_store

SQLITE_SUPPORTS_DROP_COLUMN = sqlite3.sqlite_version_info >= (3, 35, 0)


def _insert_stage(store: RecordStore, stats) -> int:
    sid = store.start_stage(
        concurrency=stats.concurrency, mode="closed",
        duration_s=stats.duration_s, warmup_s=stats.warmup_s,
    )
    store.finish_stage(sid, stats)
    return sid


class NewSchemaTest(unittest.TestCase):
    def setUp(self):
        self.dir = tmp_dir("store_new")
        self.addCleanup(safe_rmtree, self.dir)
        self.store = RecordStore(str(self.dir / "new.db"))

    def tearDown(self):
        self.store.close()

    def test_columns_exist_in_fresh_db(self):
        cols = {r["name"] for r in self.store._conn.execute("PRAGMA table_info(stages)")}
        self.assertIn("ok_rps", cols)
        self.assertIn("input_tps", cols)

    def test_finish_stage_roundtrip(self):
        stats = make_stats(1, 8, 120, 110, 10, ok_rps=2.5, input_tps=2500.0,
                           output_tps=500.0, rps=2.7)
        _insert_stage(self.store, stats)
        rows = self.store.all_stages()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["ok_rps"], 2.5, places=6)
        self.assertAlmostEqual(rows[0]["input_tps"], 2500.0, places=6)

    def test_null_columns_allowed(self):
        # 无完成样本的级：吞吐列可空（不编造 0）
        stats = make_stats(1, 8, 0, 0, 0, ok_rps=None, input_tps=None,
                           output_tps=None, rps=None)
        _insert_stage(self.store, stats)
        row = self.store.all_stages()[0]
        self.assertIsNone(row["ok_rps"])
        self.assertIsNone(row["input_tps"])

    def test_report_only_reads_stored_values_verbatim(self):
        # 新库：读回即落库值，不走推导（推导只在缺列时兜底）
        stats = make_stats(1, 8, 120, 110, 10, ok_rps=2.5, input_tps=2500.0,
                           output_tps=500.0, rps=2.75)
        _insert_stage(self.store, stats)
        entries, _thr, _base = _stats_from_store(self.store, make_config())
        self.assertEqual(len(entries), 1)
        self.assertAlmostEqual(entries[0].ok_rps, 2.5, places=9)
        self.assertAlmostEqual(entries[0].input_tps, 2500.0, places=9)


@unittest.skipUnless(SQLITE_SUPPORTS_DROP_COLUMN,
                     f"SQLite {sqlite3.sqlite_version} 不支持 DROP COLUMN，跳过旧库模拟")
class OldLibraryTest(unittest.TestCase):
    """旧库（缺两列）的迁移与守卫推导。"""

    def setUp(self):
        self.dir = tmp_dir("store_old")
        self.addCleanup(safe_rmtree, self.dir)
        self.db = self.dir / "old.db"
        # 先按新 schema 写入一行，再删掉两列模拟旧库
        store = RecordStore(str(self.db))
        _insert_stage(store, make_stats(1, 8, 120, 110, 10, ok_rps=2.5,
                                        input_tps=2500.0, output_tps=500.0, rps=2.75))
        store.close()
        conn = sqlite3.connect(str(self.db))
        conn.execute("ALTER TABLE stages DROP COLUMN ok_rps")
        conn.execute("ALTER TABLE stages DROP COLUMN input_tps")
        conn.commit()
        conn.close()

    def test_migrate_restores_columns(self):
        store = RecordStore(str(self.db))
        try:
            cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(stages)")}
            self.assertIn("ok_rps", cols)
            self.assertIn("input_tps", cols)
        finally:
            store.close()

    def test_report_only_derives_fallback_from_old_library(self):
        # 旧库读回：缺列 → 由充分统计量推导（rps=2.75, n_ok=110/120）
        store = RecordStore(str(self.db))
        try:
            entries, _thr, _base = _stats_from_store(store, make_config())
            self.assertEqual(len(entries), 1)
            expected_ok_rps = 2.75 * 110 / 120
            self.assertAlmostEqual(entries[0].ok_rps, expected_ok_rps, places=9)
            # input_tps = prompt_tokens_avg(1000) × ok_rps
            self.assertAlmostEqual(entries[0].input_tps, 1000.0 * expected_ok_rps, places=9)
        finally:
            store.close()

    def test_derivation_helper_exact(self):
        row = {"rps": 2.75, "n_total": 120, "n_ok": 110, "prompt_tokens_avg": 1000.0}
        ok_rps, input_tps = _derive_throughput_fallback(row)
        self.assertAlmostEqual(ok_rps, 2.75 * 110 / 120, places=9)
        self.assertAlmostEqual(input_tps, 1000.0 * ok_rps, places=9)

    def test_derivation_helper_missing_data_returns_none(self):
        # 无 rps（无时间窗）或 avg 缺失：返回 None，由报告走诚实退化
        self.assertEqual(
            _derive_throughput_fallback({"rps": None, "n_total": 0, "n_ok": 0,
                                         "prompt_tokens_avg": None}), (None, None))
        self.assertEqual(
            _derive_throughput_fallback({"rps": None, "n_total": 100, "n_ok": 90,
                                         "prompt_tokens_avg": 500.0}), (None, None))
        ok_rps, input_tps = _derive_throughput_fallback(
            {"rps": 2.0, "n_total": 100, "n_ok": 50, "prompt_tokens_avg": None})
        self.assertAlmostEqual(ok_rps, 1.0, places=9)
        self.assertIsNone(input_tps)


if __name__ == "__main__":
    unittest.main()
