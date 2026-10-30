"""PD 分离推理服务质量（SLO/SLA）基准测试工具的配置加载与校验。

配置格式支持 TOML（Python >= 3.11 内置 tomllib）与 JSON，按扩展名自动识别。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 及以下
    tomllib = None  # type: ignore[assignment]


class ConfigError(Exception):
    """配置缺失或非法时抛出。"""


@dataclass
class ServiceConfig:
    """被测服务（Router 入口）的连接信息。"""

    base_url: str = "http://127.0.0.1:8000"
    api_key: str = ""
    path: str = "/v1/chat/completions"
    model: str = "default"
    timeout_s: float = 600.0          # 单请求整体超时；分钟级请求必须足够大
    connect_timeout_s: float = 10.0
    verify_tls: bool = False
    extra_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class SloConfig:
    """服务质量判定门限。"""

    ttft_target_s: float = 2.0        # 首 token 时间目标
    tpot_target_ms: float = 50.0      # 每输出 token 时间目标
    min_completion_tokens: int = 2    # 少于此值时 TPOT 不参与判定


@dataclass
class LoadConfig:
    """压力施加方式。"""

    mode: str = "closed"              # closed=闭环并发 | open=开环固定 RPS
    concurrency_steps: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
    )
    stage_duration_s: float = 60.0    # 每级稳压时间
    warmup_s: float = 10.0            # 每级预热时间（不计入统计）
    think_time_s: float = 0.0         # 闭环模式下每次请求前的思考时间
    max_requests_per_stage: int = 5000        # 测量段请求数上限
    max_warmup_requests_per_stage: int = 5000 # 预热段请求数上限（与测量段分开，防止预热耗尽配额）
    drain_timeout_s: float = 60.0     # 到点后在途请求的最大等待
    target_rps: float = 0.0           # 仅 open 模式
    max_inflight: int = 4096          # 仅 open 模式


@dataclass
class RequestConfig:
    """请求特征生成规则。"""

    prompt_token_targets: List[int] = field(default_factory=lambda: [512, 2048, 8192])
    prompt_weights: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.3])
    max_context_tokens: int = 32768   # 服务支持的上下文上限（prompt+output 必须不超过）
    output_token_choices: List[int] = field(default_factory=lambda: [128, 256])
    output_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])
    temperature: float = 0.7
    stream: bool = True
    unique_prefix: bool = True        # 每请求加唯一前缀，规避前缀缓存干扰
    seed: int = 20241030
    chars_per_token: float = 3.5      # 无 tokenizer 时的字符/token 估算比


@dataclass
class AnalysisConfig:
    """结论判定参数。"""

    sla_target: float = 0.999         # 3 个 9
    slo_target: float = 0.95
    min_samples_per_stage: int = 100  # 样本少于此值的级别不参与达标判定


@dataclass
class StorageConfig:
    db_path: str = ""


@dataclass
class ReportConfig:
    output_path: str = ""


@dataclass
class Config:
    service: ServiceConfig
    slo: SloConfig
    load: LoadConfig
    requests: RequestConfig
    analysis: AnalysisConfig
    storage: StorageConfig
    report: ReportConfig
    source_path: str = ""

    # ------------------------------------------------------------------ #
    def apply_overrides(self, **kw: Any) -> None:
        """应用命令行覆盖（仅覆盖显式传入的非 None 值）。"""
        for key, value in kw.items():
            if value is None:
                continue
            if not hasattr(self, key):
                raise ConfigError(f"未知的配置覆盖项: {key}")
            setattr(self, key, value)

    def summary(self) -> Dict[str, Any]:
        return {
            "base_url": self.service.base_url,
            "path": self.service.path,
            "model": self.service.model,
            "stream": self.requests.stream,
            "concurrency_steps": list(self.load.concurrency_steps),
            "stage_duration_s": self.load.stage_duration_s,
            "warmup_s": self.load.warmup_s,
            "ttft_target_s": self.slo.ttft_target_s,
            "tpot_target_ms": self.slo.tpot_target_ms,
            "sla_target": self.analysis.sla_target,
            "slo_target": self.analysis.slo_target,
            "prompt_token_targets": list(self.requests.prompt_token_targets),
            "output_token_choices": list(self.requests.output_token_choices),
            "max_context_tokens": self.requests.max_context_tokens,
        }


# ---------------------------------------------------------------------- #
# 加载
# ---------------------------------------------------------------------- #

def _require(mapping: Dict[str, Any], key: str, typ: type, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"[{where}] 缺少必填字段: {key}")
    value = mapping[key]
    if not isinstance(value, typ):
        raise ConfigError(f"[{where}] 字段 {key} 类型应为 {typ.__name__}，实际为 {type(value).__name__}")
    return value


def _optional(mapping: Dict[str, Any], key: str, default: Any, typ: type, where: str) -> Any:
    if key not in mapping or mapping[key] is None:
        return default
    value = mapping[key]
    if isinstance(value, bool) and typ is float:
        value = float(value)
    if not isinstance(value, typ):
        raise ConfigError(f"[{where}] 字段 {key} 类型应为 {typ.__name__}，实际为 {type(value).__name__}")
    return value


def _num_list(mapping: Dict[str, Any], key: str, where: str) -> Optional[List[float]]:
    raw = mapping.get(key)
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw):
        raise ConfigError(f"[{where}] 字段 {key} 必须为数字数组")
    return [float(v) for v in raw]


def _int_list(mapping: Dict[str, Any], key: str, where: str) -> Optional[List[int]]:
    raw = mapping.get(key)
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in raw):
        raise ConfigError(f"[{where}] 字段 {key} 必须为整数数组")
    return list(raw)


def load_config(path: str) -> Config:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    if p.suffix.lower() == ".toml":
        if tomllib is None:
            raise ConfigError("当前 Python 无 tomllib（需要 >= 3.11），请改用 JSON 配置或升级 Python")
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ConfigError(f"不支持的配置格式: {p.suffix}（支持 .toml / .json）")
    if not isinstance(raw, dict):
        raise ConfigError("配置文件根节点必须是表/对象")

    svc_raw = raw.get("service", {})
    slo_raw = raw.get("slo", {})
    load_raw = raw.get("load", {})
    req_raw = raw.get("requests", {})
    ana_raw = raw.get("analysis", {})
    sto_raw = raw.get("storage", {})
    rep_raw = raw.get("report", {})

    service = ServiceConfig(
        base_url=_optional(svc_raw, "base_url", ServiceConfig.base_url, str, "service"),
        api_key=_optional(svc_raw, "api_key", "", str, "service"),
        path=_optional(svc_raw, "path", ServiceConfig.path, str, "service"),
        model=_optional(svc_raw, "model", ServiceConfig.model, str, "service"),
        timeout_s=_optional(svc_raw, "timeout_s", ServiceConfig.timeout_s, float, "service"),
        connect_timeout_s=_optional(svc_raw, "connect_timeout_s", ServiceConfig.connect_timeout_s, float, "service"),
        verify_tls=_optional(svc_raw, "verify_tls", False, bool, "service"),
        extra_headers=dict(svc_raw.get("extra_headers", {})),
    )

    slo = SloConfig(
        ttft_target_s=_optional(slo_raw, "ttft_target_s", SloConfig.ttft_target_s, float, "slo"),
        tpot_target_ms=_optional(slo_raw, "tpot_target_ms", SloConfig.tpot_target_ms, float, "slo"),
        min_completion_tokens=_optional(slo_raw, "min_completion_tokens", SloConfig.min_completion_tokens, int, "slo"),
    )

    steps = _int_list(load_raw, "concurrency_steps", "load") or list(LoadConfig.concurrency_steps)
    load = LoadConfig(
        mode=_optional(load_raw, "mode", "closed", str, "load"),
        concurrency_steps=sorted(set(steps)),
        stage_duration_s=_optional(load_raw, "stage_duration_s", LoadConfig.stage_duration_s, float, "load"),
        warmup_s=_optional(load_raw, "warmup_s", LoadConfig.warmup_s, float, "load"),
        think_time_s=_optional(load_raw, "think_time_s", LoadConfig.think_time_s, float, "load"),
        max_requests_per_stage=_optional(load_raw, "max_requests_per_stage", LoadConfig.max_requests_per_stage, int, "load"),
        max_warmup_requests_per_stage=_optional(
            load_raw, "max_warmup_requests_per_stage", LoadConfig.max_warmup_requests_per_stage, int, "load"
        ),
        drain_timeout_s=_optional(load_raw, "drain_timeout_s", LoadConfig.drain_timeout_s, float, "load"),
        target_rps=_optional(load_raw, "target_rps", 0.0, float, "load"),
        max_inflight=_optional(load_raw, "max_inflight", LoadConfig.max_inflight, int, "load"),
    )

    requests = RequestConfig(
        prompt_token_targets=_int_list(req_raw, "prompt_token_targets", "requests")
        or list(RequestConfig.prompt_token_targets),
        prompt_weights=_num_list(req_raw, "prompt_weights", "requests") or list(RequestConfig.prompt_weights),
        max_context_tokens=_optional(req_raw, "max_context_tokens", RequestConfig.max_context_tokens, int, "requests"),
        output_token_choices=_int_list(req_raw, "output_token_choices", "requests")
        or list(RequestConfig.output_token_choices),
        output_weights=_num_list(req_raw, "output_weights", "requests") or list(RequestConfig.output_weights),
        temperature=_optional(req_raw, "temperature", RequestConfig.temperature, float, "requests"),
        stream=_optional(req_raw, "stream", RequestConfig.stream, bool, "requests"),
        unique_prefix=_optional(req_raw, "unique_prefix", RequestConfig.unique_prefix, bool, "requests"),
        seed=_optional(req_raw, "seed", RequestConfig.seed, int, "requests"),
        chars_per_token=_optional(req_raw, "chars_per_token", RequestConfig.chars_per_token, float, "requests"),
    )

    analysis = AnalysisConfig(
        sla_target=_optional(ana_raw, "sla_target", AnalysisConfig.sla_target, float, "analysis"),
        slo_target=_optional(ana_raw, "slo_target", AnalysisConfig.slo_target, float, "analysis"),
        min_samples_per_stage=_optional(ana_raw, "min_samples_per_stage", AnalysisConfig.min_samples_per_stage, int, "analysis"),
    )

    storage = StorageConfig(db_path=str(sto_raw.get("db_path", "")))
    report = ReportConfig(output_path=str(rep_raw.get("output_path", "")))

    cfg = Config(
        service=service,
        slo=slo,
        load=load,
        requests=requests,
        analysis=analysis,
        storage=storage,
        report=report,
        source_path=str(p),
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """语义校验：任何一项不合法都直接 fail-fast，避免跑几十分钟后才发现配置错误。"""
    s, l, r, a = cfg.service, cfg.load, cfg.requests, cfg.analysis

    if not s.base_url.startswith(("http://", "https://")):
        raise ConfigError(f"service.base_url 必须以 http:// 或 https:// 开头: {s.base_url}")
    if not s.path.startswith("/"):
        raise ConfigError(f"service.path 必须以 / 开头: {s.path}")
    if s.timeout_s <= 0 or s.connect_timeout_s <= 0:
        raise ConfigError("service.timeout_s / connect_timeout_s 必须为正数")

    if l.mode not in ("closed", "open"):
        raise ConfigError(f"load.mode 仅支持 closed / open: {l.mode}")
    if not l.concurrency_steps:
        raise ConfigError("load.concurrency_steps 不能为空")
    if any(c <= 0 for c in l.concurrency_steps):
        raise ConfigError("load.concurrency_steps 必须全为正整数")
    if l.stage_duration_s <= 0:
        raise ConfigError("load.stage_duration_s 必须为正数")
    if l.warmup_s < 0:
        raise ConfigError("load.warmup_s 不能为负")
    if l.warmup_s >= l.stage_duration_s:
        raise ConfigError("load.warmup_s 必须小于 load.stage_duration_s")
    if l.max_requests_per_stage <= 0:
        raise ConfigError("load.max_requests_per_stage 必须为正数")
    if l.max_warmup_requests_per_stage <= 0:
        raise ConfigError("load.max_warmup_requests_per_stage 必须为正数")
    if l.mode == "open" and l.target_rps <= 0:
        raise ConfigError("open 模式下必须配置 load.target_rps > 0")

    if not r.prompt_token_targets or any(t <= 0 for t in r.prompt_token_targets):
        raise ConfigError("requests.prompt_token_targets 必须为非空正整数数组")
    if not r.output_token_choices or any(t <= 0 for t in r.output_token_choices):
        raise ConfigError("requests.output_token_choices 必须为非空正整数数组")
    if len(r.prompt_weights) != len(r.prompt_token_targets):
        raise ConfigError("requests.prompt_weights 与 prompt_token_targets 长度不一致")
    if len(r.output_weights) != len(r.output_token_choices):
        raise ConfigError("requests.output_weights 与 output_token_choices 长度不一致")
    if abs(sum(r.prompt_weights)) < 1e-9 or abs(sum(r.output_weights)) < 1e-9:
        raise ConfigError("权重数组之和不能为 0")
    if r.chars_per_token <= 0:
        raise ConfigError("requests.chars_per_token 必须为正数")

    longest = max(r.prompt_token_targets) + max(r.output_token_choices)
    if longest > r.max_context_tokens:
        raise ConfigError(
            f"最长请求（prompt {max(r.prompt_token_targets)} + output {max(r.output_token_choices)} "
            f"= {longest}）超过 max_context_tokens={r.max_context_tokens}，请下调长度或修正上下文上限"
        )

    if not (0.0 < a.sla_target <= 1.0):
        raise ConfigError("analysis.sla_target 必须在 (0, 1] 区间")
    if not (0.0 < a.slo_target <= 1.0):
        raise ConfigError("analysis.slo_target 必须在 (0, 1] 区间")
    if a.min_samples_per_stage <= 0:
        raise ConfigError("analysis.min_samples_per_stage 必须为正数")

    if cfg.slo.ttft_target_s <= 0 or cfg.slo.tpot_target_ms <= 0:
        raise ConfigError("slo.ttft_target_s / tpot_target_ms 必须为正数")
