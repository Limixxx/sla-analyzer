"""SQLite 存储层：只记录请求特征（长度、时延、分类），不保存原始请求/响应内容。"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS stages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    concurrency     INTEGER NOT NULL,
    mode            TEXT NOT NULL DEFAULT 'closed',
    kind            TEXT NOT NULL DEFAULT 'stage',
    started_at      REAL,
    ended_at        REAL,
    duration_s      REAL,
    warmup_s        REAL,
    actual_duration_s REAL,
    n_warmup        INTEGER DEFAULT 0,
    n_total         INTEGER DEFAULT 0,
    n_ok            INTEGER DEFAULT 0,
    n_failed        INTEGER DEFAULT 0,
    n_aborted       INTEGER DEFAULT 0,
    a1              INTEGER DEFAULT 0,
    a2              INTEGER DEFAULT 0,
    a3              INTEGER DEFAULT 0,
    slo             REAL,
    sla             REAL,
    ttft_p50_ms     REAL,
    ttft_p90_ms     REAL,
    ttft_p99_ms     REAL,
    tpot_p50_ms     REAL,
    tpot_p90_ms     REAL,
    tpot_p99_ms     REAL,
    e2e_p50_ms      REAL,
    e2e_p99_ms      REAL,
    connect_p50_ms  REAL,
    connect_p99_ms  REAL,
    rps             REAL,
    output_tps      REAL,
    prompt_tokens_avg   REAL,
    completion_tokens_avg REAL,
    fail_kinds_json TEXT
);
CREATE TABLE IF NOT EXISTS stage_buckets (
    stage_id      INTEGER NOT NULL,
    bucket        INTEGER NOT NULL,
    n             INTEGER DEFAULT 0,
    a1            INTEGER DEFAULT 0,
    a2            INTEGER DEFAULT 0,
    a3            INTEGER DEFAULT 0,
    a3_ttft_only  INTEGER DEFAULT 0,
    a3_tpot_only  INTEGER DEFAULT 0,
    a3_both       INTEGER DEFAULT 0,
    ttft_p99_ms   REAL,
    tpot_p99_ms   REAL,
    ttft_limit_ms REAL,
    tpot_limit_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_stage_buckets ON stage_buckets(stage_id);
CREATE TABLE IF NOT EXISTS baseline (
    stage_id      INTEGER NOT NULL,
    concurrency   INTEGER,
    duration_s    REAL,
    duration_actual_s REAL,
    bucket        INTEGER NOT NULL,
    n             INTEGER DEFAULT 0,
    n_ok          INTEGER DEFAULT 0,
    a1            INTEGER DEFAULT 0,
    ttft_p50_ms   REAL,
    ttft_p95_ms   REAL,
    ttft_p99_ms   REAL,
    tpot_p50_ms   REAL,
    tpot_p95_ms   REAL,
    tpot_p99_ms   REAL
);
CREATE INDEX IF NOT EXISTS idx_baseline_stage ON baseline(stage_id);
CREATE TABLE IF NOT EXISTS requests (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    stage_id             INTEGER NOT NULL,
    concurrency          INTEGER NOT NULL,
    worker_id            INTEGER,
    is_warmup            INTEGER DEFAULT 0,
    mode                 TEXT,
    sent_at              REAL,
    connect_ms           REAL,
    ttft_ms              REAL,
    tpot_ms              REAL,
    e2e_ms               REAL,
    prompt_tokens_planned INTEGER,
    prompt_tokens        INTEGER,
    completion_tokens    INTEGER,
    max_tokens           INTEGER,
    token_source         TEXT,
    status               TEXT,
    error_kind           TEXT,
    http_code            INTEGER,
    finish_reason        TEXT,
    latency_class        TEXT,
    note                 TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_stage ON requests(stage_id, is_warmup);
CREATE INDEX IF NOT EXISTS idx_requests_class ON requests(stage_id, latency_class);
"""

_INSERT_REQUEST = """
INSERT INTO requests (
    stage_id, concurrency, worker_id, is_warmup, mode, sent_at, connect_ms,
    ttft_ms, tpot_ms, e2e_ms, prompt_tokens_planned, prompt_tokens,
    completion_tokens, max_tokens, token_source, status, error_kind,
    http_code, finish_reason, latency_class, note
) VALUES (
    :stage_id, :concurrency, :worker_id, :is_warmup, :mode, :sent_at, :connect_ms,
    :ttft_ms, :tpot_ms, :e2e_ms, :prompt_tokens_planned, :prompt_tokens,
    :completion_tokens, :max_tokens, :token_source, :status, :error_kind,
    :http_code, :finish_reason, :latency_class, :note
)
"""

_INSERT_STAGE = """
INSERT INTO stages (
    concurrency, mode, kind, started_at, ended_at, duration_s, warmup_s, n_warmup
) VALUES (:concurrency, :mode, :kind, :started_at, :ended_at, :duration_s, :warmup_s, :n_warmup)
"""

_INSERT_BUCKET = """
INSERT INTO stage_buckets (
    stage_id, bucket, n, a1, a2, a3, a3_ttft_only, a3_tpot_only, a3_both,
    ttft_p99_ms, tpot_p99_ms, ttft_limit_ms, tpot_limit_ms
) VALUES (
    :stage_id, :bucket, :n, :a1, :a2, :a3, :a3_ttft_only, :a3_tpot_only, :a3_both,
    :ttft_p99_ms, :tpot_p99_ms, :ttft_limit_ms, :tpot_limit_ms
)
"""

_INSERT_BASELINE = """
INSERT INTO baseline (
    stage_id, concurrency, duration_s, duration_actual_s, bucket, n, n_ok, a1,
    ttft_p50_ms, ttft_p95_ms, ttft_p99_ms, tpot_p50_ms, tpot_p95_ms, tpot_p99_ms
) VALUES (
    :stage_id, :concurrency, :duration_s, :duration_actual_s, :bucket, :n, :n_ok, :a1,
    :ttft_p50_ms, :ttft_p95_ms, :ttft_p99_ms, :tpot_p50_ms, :tpot_p95_ms, :tpot_p99_ms
)
"""

_UPDATE_STAGE = """
UPDATE stages SET
    n_total=:n_total, n_ok=:n_ok, n_failed=:n_failed, n_aborted=:n_aborted,
    n_warmup=:n_warmup, actual_duration_s=:actual_duration_s,
    a1=:a1, a2=:a2, a3=:a3, slo=:slo, sla=:sla,
    ttft_p50_ms=:ttft_p50_ms, ttft_p90_ms=:ttft_p90_ms, ttft_p99_ms=:ttft_p99_ms,
    tpot_p50_ms=:tpot_p50_ms, tpot_p90_ms=:tpot_p90_ms, tpot_p99_ms=:tpot_p99_ms,
    e2e_p50_ms=:e2e_p50_ms, e2e_p99_ms=:e2e_p99_ms,
    connect_p50_ms=:connect_p50_ms, connect_p99_ms=:connect_p99_ms,
    rps=:rps, output_tps=:output_tps,
    prompt_tokens_avg=:prompt_tokens_avg, completion_tokens_avg=:completion_tokens_avg,
    fail_kinds_json=:fail_kinds_json, ended_at=:ended_at
WHERE id=:id
"""


class RecordStore:
    """SQLite 存储；WAL 模式 + 批量写入，降低对压测进程的干扰。"""

    def __init__(self, db_path: str):
        path = Path(db_path)
        if path.parent and str(path.parent):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()
        self._pending: List[Dict[str, Any]] = []

    def _migrate(self) -> None:
        """轻量迁移：为旧版本数据库补齐新列，保证 --report-only 兼容。"""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(stages)")}
        if "kind" not in cols:
            self._conn.execute(
                "ALTER TABLE stages ADD COLUMN kind TEXT NOT NULL DEFAULT 'stage'"
            )
        if "n_aborted" not in cols:
            self._conn.execute(
                "ALTER TABLE stages ADD COLUMN n_aborted INTEGER DEFAULT 0"
            )
        if "connect_p50_ms" not in cols:
            self._conn.execute(
                "ALTER TABLE stages ADD COLUMN connect_p50_ms REAL"
            )
        if "connect_p99_ms" not in cols:
            self._conn.execute(
                "ALTER TABLE stages ADD COLUMN connect_p99_ms REAL"
            )
        if "actual_duration_s" not in cols:
            self._conn.execute(
                "ALTER TABLE stages ADD COLUMN actual_duration_s REAL"
            )
        # baseline 表在旧版本数据库中可能已存在（无 duration_actual_s 列）
        base_cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(baseline)")
        }
        if base_cols and "duration_actual_s" not in base_cols:
            self._conn.execute(
                "ALTER TABLE baseline ADD COLUMN duration_actual_s REAL"
            )

    # ------------------------------------------------------------------ #
    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def start_stage(
        self,
        concurrency: int,
        mode: str,
        duration_s: float,
        warmup_s: float,
        kind: str = "stage",
    ) -> int:
        cur = self._conn.execute(
            _INSERT_STAGE,
            {
                "concurrency": concurrency,
                "mode": mode,
                "kind": kind,
                "started_at": time.time(),
                "ended_at": None,
                "duration_s": duration_s,
                "warmup_s": warmup_s,
                "n_warmup": 0,
            },
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def add_request(self, row: Dict[str, Any]) -> None:
        self._pending.append(row)
        if len(self._pending) >= 200:
            self.flush_requests()

    def flush_requests(self) -> None:
        if not self._pending:
            return
        self._conn.executemany(_INSERT_REQUEST, self._pending)
        self._conn.commit()
        self._pending.clear()

    def finish_stage(self, stage_id: int, stats: Any) -> None:
        import json as _json

        row = stats.as_row()
        row["id"] = stage_id
        row["ended_at"] = time.time()
        row["fail_kinds_json"] = _json.dumps(stats.fail_kinds, ensure_ascii=False)
        self._conn.execute(_UPDATE_STAGE, row)
        # 按长度桶的拆分另存一张表，避免把字典塞进 stages
        if getattr(stats, "bucket_stats", None):
            self._conn.execute("DELETE FROM stage_buckets WHERE stage_id=?", (stage_id,))
            self._conn.executemany(_INSERT_BUCKET, [
                {
                    "stage_id": stage_id,
                    "bucket": idx,
                    "n": entry.get("n", 0),
                    "a1": entry.get("a1", 0),
                    "a2": entry.get("a2", 0),
                    "a3": entry.get("a3", 0),
                    "a3_ttft_only": entry.get("a3_ttft_only", 0),
                    "a3_tpot_only": entry.get("a3_tpot_only", 0),
                    "a3_both": entry.get("a3_both", 0),
                    "ttft_p99_ms": entry.get("ttft_p99_ms"),
                    "tpot_p99_ms": entry.get("tpot_p99_ms"),
                    "ttft_limit_ms": entry.get("ttft_limit_ms"),
                    "tpot_limit_ms": entry.get("tpot_limit_ms"),
                }
                for idx, entry in sorted(stats.bucket_stats.items())
            ])
        self._conn.commit()
        self.flush_requests()

    def save_baseline(
        self,
        stage_id: int,
        concurrency: int,
        duration_s: float,
        result: Any,
    ) -> None:
        """落库 Phase 0 基线结果（供 --report-only 重建门限）。"""
        self._conn.execute("DELETE FROM baseline WHERE stage_id=?", (stage_id,))
        rows = [
            {
                "stage_id": stage_id,
                "concurrency": concurrency,
                "duration_s": duration_s,
                "duration_actual_s": getattr(result, "duration_actual_s", None),
                "bucket": entry.bucket,
                "n": entry.n,
                "n_ok": entry.n_ok,
                "a1": entry.a1,
                "ttft_p50_ms": entry.ttft_p50_ms,
                "ttft_p95_ms": entry.ttft_p95_ms,
                "ttft_p99_ms": entry.ttft_p99_ms,
                "tpot_p50_ms": entry.tpot_p50_ms,
                "tpot_p95_ms": entry.tpot_p95_ms,
                "tpot_p99_ms": entry.tpot_p99_ms,
            }
            for entry in result.buckets
        ]
        self._conn.executemany(_INSERT_BASELINE, rows)
        self._conn.commit()

    def load_baseline(self) -> Optional[Any]:
        """从数据库重建基线结果；无基线表/无数据时返回 None。"""
        from .baseline import BaselineResult, BucketBaseline

        exists = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='baseline'"
        ).fetchone()
        if not exists:
            return None
        rows = self._conn.execute("SELECT * FROM baseline ORDER BY bucket").fetchall()
        if not rows:
            return None
        has_actual = "duration_actual_s" in rows[0].keys()
        result = BaselineResult(
            concurrency=rows[0]["concurrency"] or 0,
            duration_s=rows[0]["duration_s"] or 0.0,
            duration_actual_s=rows[0]["duration_actual_s"] if has_actual else None,
            buckets=[
                BucketBaseline(
                    bucket=r["bucket"],
                    n=r["n"] or 0,
                    n_ok=r["n_ok"] or 0,
                    a1=r["a1"] or 0,
                    ttft_p50_ms=r["ttft_p50_ms"],
                    ttft_p95_ms=r["ttft_p95_ms"],
                    ttft_p99_ms=r["ttft_p99_ms"],
                    tpot_p50_ms=r["tpot_p50_ms"],
                    tpot_p95_ms=r["tpot_p95_ms"],
                    tpot_p99_ms=r["tpot_p99_ms"],
                )
                for r in rows
            ],
        )
        result.total_n = sum(b.n for b in result.buckets)
        result.total_a1 = sum(b.a1 for b in result.buckets)
        return result

    # ------------------------------------------------------------------ #
    def stage_rows(self, stage_id: int, include_warmup: bool = False) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM requests WHERE stage_id=?"
        if not include_warmup:
            sql += " AND is_warmup=0"
        sql += " ORDER BY id"
        return [dict(r) for r in self._conn.execute(sql, (stage_id,)).fetchall()]

    def warmup_count(self, stage_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM requests WHERE stage_id=? AND is_warmup=1", (stage_id,)
        ).fetchone()
        return int(row["c"])

    def aborted_count(self, stage_id: int) -> int:
        """本级被放弃的在途请求数（含预热段；预热段放弃同样说明 drain 窗口不足）。"""
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM requests WHERE stage_id=? AND status='aborted'", (stage_id,)
        ).fetchone()
        return int(row["c"])

    def all_stages(self, kind: str = "stage") -> List[Dict[str, Any]]:
        """取指定类型的阶段；默认只取正式压力级，排除 Phase 0 基线阶段。"""
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM stages WHERE kind=? ORDER BY id", (kind,)
            ).fetchall()
        ]

    def stage_buckets(self, stage_id: int) -> List[Dict[str, Any]]:
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM stage_buckets WHERE stage_id=? ORDER BY bucket", (stage_id,)
            ).fetchall()
        ]

    def meta_items(self) -> Dict[str, str]:
        return {r["key"]: r["value"] for r in self._conn.execute("SELECT key, value FROM meta").fetchall()}

    def close(self) -> None:
        try:
            self.flush_requests()
        finally:
            self._conn.close()
