"""mock E2E：一次采集、两份报告 + report-only 一致性 + capacity-plan。

用 mock_server.py（soft-capacity 6 / hard-capacity 10）构造"低压全好 →
某级起被拒"的确定形态，验证：
1. kinds=["online","throughput"] 一次运行产出两份报告 + 容量面；
2. 吞吐报告的关键数字（膝点、SLA=RPM）与容量表行数；
3. --report-only 从同一库重建，SLA 数字一致（两条路径口径不分叉）；
4. kinds=["online"]（默认）只产出在线报告，不写容量面；
5. --capacity-plan 对刚生成的容量面反查成功。
"""

from __future__ import annotations

import contextlib
import io
import json
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

from helpers import PROJECT_ROOT, load_cli_module, safe_rmtree, tmp_dir  # noqa: E402

from bench.throughput import ThroughputSurface  # noqa: E402

_cli = load_cli_module()

E2E_CONFIG = """
[service]
base_url = "http://127.0.0.1:{port}"
path = "/v1/chat/completions"
model = "mock-model"
timeout_s = 60.0
connect_timeout_s = 5.0

[slo]
ttft_target_s = 2.0
tpot_target_ms = 50.0
target_mode = "absolute"
min_completion_tokens = 2

[load]
mode = "closed"
concurrency_steps = [2, 8, 16]
stage_duration_s = 4.0
max_stage_duration_s = 8.0
warmup_s = 1.0
max_requests_per_stage = 300
drain_timeout_s = 10.0
baseline_enabled = false

[requests]
max_context_tokens = 4096
prompt_token_targets = [2048]
prompt_weights = [1.0]
output_token_choices = [64]
output_weights = [1.0]
length_buckets = [4096]
stream = true
unique_prefix = true
seed = 20261103

[analysis]
sla_target = 0.999
slo_target = 0.95
min_samples_per_stage = 5
window_requests = 100000
window_hours = 1.0

[storage]
db_path = "{dir}/e2e.db"

[report]
kinds = ["online", "throughput"]
output_path = "{dir}/e2e.md"
throughput_model_path = "{dir}/surface.npz"

[qos_model]
enabled = false
"""


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class MockE2ETest(unittest.TestCase):
    """起真实 mock_server 子进程的全链路测试（约 40 秒）。"""

    @classmethod
    def setUpClass(cls):
        cls.dir = tmp_dir("e2e")
        cls.port = _free_port()
        cls.mock = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "mock_server.py"),
             "--port", str(cls.port), "--ttft-ms", "200", "--tpot-ms", "15",
             "--ttft-per-1k-tokens", "60", "--soft-capacity", "6", "--hard-capacity", "10"],
            cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # 等待 mock 就绪（最多 15s）
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{cls.port}/v1/models", timeout=1
                ) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(0.3)
        else:
            cls.mock.terminate()
            raise unittest.SkipTest("mock_server 未在 15s 内就绪")
        cls.cfg_path = cls.dir / "e2e.toml"
        # Windows 临时路径含反斜杠：TOML 基本字符串会把它当转义符（\U →
        # invalid hex），统一换成正斜杠（Python/open 均兼容）
        cls.cfg_path.write_text(
            E2E_CONFIG.format(port=cls.port, dir=str(cls.dir)).replace("\\", "/"),
            encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.mock.terminate()
        try:
            cls.mock.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.mock.kill()
        safe_rmtree(cls.dir)

    def _run_cli(self, argv):
        buf = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = _cli.main(argv)
        if code != 0:
            raise AssertionError(f"CLI 退出码 {code}\nstdout:\n{buf.getvalue()}\nstderr:\n{err.getvalue()}")
        return code, buf.getvalue()

    def _ensure_base_run(self):
        """完整运行一次（kinds 双报告）；已跑过则复用，保证各测试自给自足
        （unittest 按方法名字母序执行，不能依赖声明顺序）。"""
        db = self.dir / "run1.db"
        if db.exists():
            return
        self._run_cli([
            "--config", str(self.cfg_path),
            "--db", str(db),
            "--report", str(self.dir / "run1.md"),
        ])

    def test_full_run_two_reports_and_surface(self):
        self._ensure_base_run()
        online = self.dir / "run1.md"
        throughput = self.dir / "run1_throughput.md"
        surface = self.dir / "surface.npz"
        self.assertTrue(online.exists(), "在线报告未产出")
        self.assertTrue(throughput.exists(), "吞吐报告未产出")
        self.assertTrue(surface.exists(), "容量面未产出")

        text = throughput.read_text(encoding="utf-8")
        # 边界行 + 范围声明（§7.10.4）
        self.assertIn(f"测量边界：`http://127.0.0.1:{self.port}`", text)
        self.assertIn("范围声明", text)
        # 膝点在并发 16（hard-capacity 10 之上全被拒），容量读数在并发 8
        self.assertIn("A1 拐点 = 负载 16", text)
        self.assertIn("容量读数 @ 负载 8", text)
        # SLA/SLO 以吞吐单位呈现
        self.assertIn("RPM", text)
        self.assertIn("tokens/s", text)
        # 容量表三行（三个压力级）
        self.assertIn("| 2 |", text)
        self.assertIn("| 8 |", text)
        self.assertIn("| 16 |", text)

        s = ThroughputSurface.load(str(surface))
        self.assertEqual(len(s.points), 1)
        self.assertEqual(s.points[0].input_len, 2048)
        self.assertEqual(s.points[0].req_conc, 8)  # 取得峰值的负载水平

    def test_report_only_reproduces_same_sla(self):
        # 同一库重建：实时路径与 --report-only 路径口径不分叉
        self._ensure_base_run()
        code, out = self._run_cli([
            "--config", str(self.cfg_path), "--report-only",
            "--db", str(self.dir / "run1.db"),
            "--report", str(self.dir / "ro.md"),
        ])
        self.assertEqual(code, 0, out)
        self.assertTrue((self.dir / "ro.md").exists())
        self.assertTrue((self.dir / "ro_throughput.md").exists())
        text = (self.dir / "ro_throughput.md").read_text(encoding="utf-8")
        # 与实时路径同一口径：同一库重建出同一膝点与容量读数
        self.assertIn("A1 拐点 = 负载 16", text)
        self.assertIn("容量读数 @ 负载 8", text)

    def test_default_kinds_online_only(self):
        # 默认 kinds（online）：只产出在线报告，不写容量面、不产吞吐报告
        self._ensure_base_run()
        cfg = self.dir / "online_only.toml"
        cfg.write_text(
            E2E_CONFIG.format(port=self.port, dir=str(self.dir))
            .replace("\\", "/")
            .replace('kinds = ["online", "throughput"]', 'kinds = ["online"]')
            .replace("/surface.npz", "/surface_default.npz"),
            encoding="utf-8")
        code, out = self._run_cli([
            "--config", str(cfg), "--report-only",
            "--db", str(self.dir / "run1.db"),
            "--report", str(self.dir / "d.md"),
        ])
        self.assertEqual(code, 0, out)
        self.assertTrue((self.dir / "d.md").exists())
        self.assertFalse((self.dir / "d_throughput.md").exists(),
                         "默认 kinds 不应产出吞吐报告")
        self.assertFalse((self.dir / "surface_default.npz").exists(),
                         "默认 kinds 不应写容量面")

    def test_capacity_plan_query(self):
        self._ensure_base_run()
        surface = self.dir / "surface.npz"
        code, out = self._run_cli([
            "--capacity-plan", str(surface), "--target-rpm", "100", "--input-len", "2048",
        ])
        self.assertEqual(code, 0, out)
        self.assertIn("反查输入", out)
        self.assertIn("预测容量", out)
        self.assertIn("判定", out)
        # 单点容量面：任何不在 2048 的长度都触发外插标注
        code2, out2 = self._run_cli([
            "--capacity-plan", str(surface), "--target-rpm", "100", "--input-len", "99999",
        ])
        self.assertEqual(code2, 0, out2)
        self.assertIn("clamp", out2)


if __name__ == "__main__":
    unittest.main()
