"""report.kinds / 窗口承诺配置的加载与校验（docs/07 §7.9 / §7.10.3）。"""

from __future__ import annotations

import unittest

from helpers import PROJECT_ROOT, make_config, tmp_dir, write_toml  # noqa: E402

from bench.config import ConfigError, load_config  # noqa: E402


class DefaultsTest(unittest.TestCase):
    def test_defaults_are_online_only(self):
        cfg = make_config()
        self.assertEqual(cfg.report.kinds, ["online"])
        self.assertEqual(cfg.analysis.window_requests, 0)
        self.assertEqual(cfg.analysis.window_hours, 0.0)
        self.assertEqual(cfg.analysis.window_safety_factor, 1.1)

    def test_summary_carries_new_keys(self):
        s = make_config().summary()
        for key in ("report_kinds", "window_requests", "window_hours",
                    "window_safety_factor", "throughput_model_path"):
            self.assertIn(key, s)


class TomlParsingTest(unittest.TestCase):
    def setUp(self):
        self.dir = tmp_dir("cfg")
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))

    def _load(self, body: str):
        return load_config(write_toml(self.dir / "c.toml", body))

    BASE = """
[service]
base_url = "http://127.0.0.1:8000"
path = "/v1/chat/completions"
model = "m"
[requests]
prompt_token_targets = [1024]
prompt_weights = [1.0]
"""

    def test_kinds_parsed(self):
        cfg = self._load(self.BASE + '\n[report]\nkinds = ["online", "throughput"]\n')
        self.assertEqual(cfg.report.kinds, ["online", "throughput"])

    def test_kinds_default_when_absent(self):
        cfg = self._load(self.BASE)
        self.assertEqual(cfg.report.kinds, ["online"])

    def test_kinds_invalid_value_rejected(self):
        with self.assertRaises(ConfigError):
            self._load(self.BASE + '\n[report]\nkinds = ["online", "batch"]\n')

    def test_kinds_empty_rejected(self):
        with self.assertRaises(ConfigError):
            self._load(self.BASE + '\n[report]\nkinds = []\n')

    def test_kinds_duplicate_rejected(self):
        with self.assertRaises(ConfigError):
            self._load(self.BASE + '\n[report]\nkinds = ["online", "online"]\n')

    def test_window_both_set_ok(self):
        cfg = self._load(
            self.BASE
            + "\n[analysis]\nwindow_requests = 1000000\nwindow_hours = 24.0\n"
        )
        self.assertEqual(cfg.analysis.window_requests, 1000000)
        self.assertEqual(cfg.analysis.window_hours, 24.0)

    def test_window_half_set_rejected(self):
        with self.assertRaises(ConfigError):
            self._load(self.BASE + "\n[analysis]\nwindow_requests = 1000\n")

    def test_window_negative_rejected(self):
        with self.assertRaises(ConfigError):
            self._load(self.BASE + "\n[analysis]\nwindow_hours = -1.0\n")

    def test_safety_factor_below_one_rejected(self):
        with self.assertRaises(ConfigError):
            self._load(
                self.BASE
                + "\n[analysis]\nwindow_requests = 1000\nwindow_hours = 1.0\n"
                + "window_safety_factor = 0.9\n"
            )

    def test_throughput_model_path_parsed(self):
        cfg = self._load(
            self.BASE
            + '\n[report]\nkinds = ["throughput"]\n'
            + 'throughput_model_path = "results/x.npz"\n'
        )
        self.assertEqual(cfg.report.throughput_model_path, "results/x.npz")

    def test_baseline_off_requires_absolute_mode(self):
        # 容量预设的硬性搭配：baseline_enabled=false 时 target_mode 必须 absolute
        with self.assertRaises(ConfigError):
            self._load(
                self.BASE
                + '\n[slo]\ntarget_mode = "baseline_relative"\n'
                + "\n[load]\nbaseline_enabled = false\n"
            )


class ThroughputPresetTest(unittest.TestCase):
    """config/config.throughput.toml 预设本身必须可加载且口径正确。"""

    def test_preset_loads(self):
        cfg = load_config(str(PROJECT_ROOT / "config" / "config.throughput.toml"))
        self.assertEqual(cfg.report.kinds, ["online", "throughput"])
        self.assertEqual(cfg.requests.prompt_token_targets, [4096])
        self.assertEqual(cfg.requests.prompt_weights, [1.0])
        self.assertEqual(cfg.requests.output_token_choices, [512])
        self.assertEqual(cfg.load.baseline_enabled, False)
        self.assertEqual(cfg.slo.target_mode, "absolute")
        self.assertEqual(cfg.qos_model.enabled, False)
        self.assertEqual(cfg.derived_lengths, [])  # 全部显式配置，无推导
        self.assertEqual(cfg.analysis.window_requests, 100000)


if __name__ == "__main__":
    unittest.main()
