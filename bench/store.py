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
    started_at      REAL,
    ended_at        REAL,
    duration_s      REAL,
    warmup_s        REAL,
    n_warmup        INTEGER DEFAULT 0,
    n_total         INTEGER DEFAULT 0,
    n_ok            INTEGER DEFAULT 0,
    n_failed        INTEGER DEFAULT 0,
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
    rps             REAL,
    output_tps      REAL,
    prompt_tokens_avg   REAL,
    completion_tokens_avg REAL,
    fail_kinds_json TEXT
);
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
    concurrency, mode, started_at, ended_at, duration_s, warmup_s, n_warmup
) VALUES (:concurrency, :mode, :started_at, :ended_at, :duration_s, :warmup_s, :n_warmup)
"""

_UPDATE_STAGE = """
UPDATE stages SET
    n_total=:n_total, n_ok=:n_ok, n_failed=:n_failed,
    a1=:a1, a2=:a2, a3=:a3, slo=:slo, sla=:sla,
    ttft_p50_ms=:ttft_p50_ms, ttft_p90_ms=:ttft_p90_ms, ttft_p99_ms=:ttft_p99_ms,
    tpot_p50_ms=:tpot_p50_ms, tpot_p90_ms=:tpot_p90_ms, tpot_p99_ms=:tpot_p99_ms,
    e2e_p50_ms=:e2e_p50_ms, e2e_p99_ms=:e2e_p99_ms,
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
        self._conn.commit()
        self._pending: List[Dict[str, Any]] = []

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

    def start_stage(self, concurrency: int, mode: str, duration_s: float, warmup_s: float) -> int:
        cur = self._conn.execute(
            _INSERT_STAGE,
            {
                "concurrency": concurrency,
                "mode": mode,
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
        self._conn.commit()
        self.flush_requests()

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

    def all_stages(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM stages ORDER BY id").fetchall()]

    def meta_items(self) -> Dict[str, str]:
        return {r["key"]: r["value"] for r in self._conn.execute("SELECT key, value FROM meta").fetchall()}

    def close(self) -> None:
        try:
            self.flush_requests()
        finally:
            self._conn.close()
