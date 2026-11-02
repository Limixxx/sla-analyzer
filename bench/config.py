"""PD 分离推理服务质量（SLO/SLA）基准测试工具的配置加载与校验。

配置格式支持 TOML（Python >= 3.11 内置 tomllib）与 JSON，按扩展名自动识别。
"""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, field
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
    """服务质量判定门限——人工只提"质量价值目标"，尺子默认由实测产出。

    target_mode 决定门限从哪来：
    - ``baseline_relative``（默认）：门限 = Phase 0 低压基线各长度桶 p99 ×
      系数。人工唯一需要提的质量价值目标是**退化预算**（"比无压力时慢
      多少倍算不达标"），绝对时延水平交给实测；换引擎/模型只需重测基线，
      整条曲线自动平移。此时 A3 拐点的含义是"相对基线的退化起点"。
    - ``absolute``：用下方固定的 ttft_target_s / tpot_target_ms，即业务
      契约口径（"p99 TTFT ≤ 2s"这类对外承诺），跨次运行可比，适合验收；
      此时 A3 拐点的含义是"契约违约起点"。

    两种模式都不支持"用本次运行观测到的分位数定标"：那会让门限随压力同步
    抬升，SLO 恒约等于目标值，指标失去分辨力、无法用于验收。
    """

    ttft_target_s: float = 2.0        # absolute 模式：首 token 时间目标
    tpot_target_ms: float = 50.0      # absolute 模式：每输出 token 时间目标
    target_mode: str = "baseline_relative"  # baseline_relative | absolute
    ttft_factor: float = 3.0          # baseline_relative：TTFT 门限 = 基线 p99 × 本系数（质量退化预算）
    tpot_factor: float = 2.0          # baseline_relative：TPOT 门限 = 基线 p99 × 本系数（质量退化预算）
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
    max_stage_duration_s: float = 300.0  # 单级实际上限：样本不足时自适应延长至此为止
    max_requests_per_stage: int = 5000        # 测量段请求数上限
    max_warmup_requests_per_stage: int = 5000 # 预热段请求数上限（与测量段分开，防止预热耗尽配额）
    drain_timeout_s: float = 120.0    # 到点后在途请求的最大等待（分钟级请求需大于最长 E2E）
    target_rps: float = 0.0           # 仅 open 模式
    max_inflight: int = 4096          # 仅 open 模式
    # Phase 0 基线剖析：在正式 ramp 之前用低压并发测一次"无压力时的服务时延"，
    # 仅当 slo.target_mode=baseline_relative 时其分位数才参与门限计算。
    baseline_enabled: bool = True
    baseline_concurrency: int = 1     # 基线并发，必须低到不触发排队/拒绝
    baseline_duration_s: float = 30.0
    baseline_warmup_s: float = 5.0


@dataclass
class RequestConfig:
    """请求特征生成规则。"""

    # 以下三个长度参数【留空】时按 max_context_tokens 自动推导（见 _resolve_length_defaults）：
    #   prompt_token_targets ← [ctx/64, ctx/16, ctx/4]，权重 [0.2, 0.5, 0.3]
    #   length_buckets       ← [ctx/32, ctx/8]
    # 对 32768 上下文精确复现历史默认值，对 8K/128K 等模型自动缩放。
    # 显式配置永远优先——长度分布应镜像生产流量画像，而非模型能力上限。
    prompt_token_targets: List[int] = field(default_factory=list)
    prompt_weights: List[float] = field(default_factory=list)
    max_context_tokens: int = 32768   # 服务支持的上下文上限（prompt+output 必须不超过）
    output_token_choices: List[int] = field(default_factory=lambda: [128, 256])
    output_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])
    temperature: float = 0.7
    stream: bool = True
    unique_prefix: bool = True        # 每请求加唯一前缀，规避前缀缓存干扰
    seed: int = 20241030
    chars_per_token: float = 3.5      # 无 tokenizer 时的字符/token 估算比
    # prompt 长度分桶上界（token，左开右闭）；桶数 = len+1，最后一个桶无上界。
    # 例 [1024, 4096] → ≤1024 / 1025-4096 / >4096 三桶。
    # 分桶的意义：TTFT 与 prompt 长度强相关，全局单门限会让长 prompt 请求
    # 天然落入 A3，掩盖真正的排队信号。
    length_buckets: List[int] = field(default_factory=list)


@dataclass
class WatermarkConfig:
    """水位判定参数（服务端水位的黑盒可观测投影，见 docs/05-watermark.md）。

    推理水位分三层：软水位（BUSY，转入排队不拒绝）、硬水位（OVERLOAD，
    直接拒绝）、SLA 水位（业务承诺）。压测工具是黑盒客户端，看不到
    KV Cache / 批 slot / GPU 等服务端资源指标，因此这里判定的是
    **可观测投影**：服务端排队 → TTFT 抬升、A3 增多；服务端拒绝 → A1
    （拒绝族）增多；服务端配无限队列时不拒绝，过载投影为客户端超时 →
    A1（超时族）增多。两种 A1 都算"接不住"，均触发硬水位，归因见
    metrics.split_fail_kinds。
    OR 逻辑（任一硬条件触发即 OVERLOAD），优先级 OVERLOAD > BUSY > NORMAL。
    """

    enabled: bool = True
    soft_slo_violation_ratio: float = 0.05   # A3 占比 ≥ 此值 → 软水位 BUSY（排队预警）
    hard_slo_violation_ratio: float = 0.50   # A3 占比 ≥ 此值 → 硬水位 OVERLOAD（质量崩塌）
    hard_reject_ratio: float = 0.01          # A1 占比 ≥ 此值 → 硬水位 OVERLOAD
                                              # （服务拒绝，或无限队列下排队耗尽投影为客户端超时）
    soft_ttft_p99_ms: float = 0.0            # >0 时 TTFT p99 超此值也判 BUSY；0=关闭


@dataclass
class QosModelConfig:
    """服务质量预测模型（MLP）参数，见 docs/06-qos-model.md。

    压测采样（并发 × prompt × completion → TTFT/TPOT）训练一个小型 MLP，
    把离散压力级间的空白补成连续预测。三条统计纪律不可协商：
    按 stage 分组交叉验证（防同级泄漏）、残差 ECDF 校准（不假设分布族）、
    包络守卫（外推不参与结论）。
    """

    enabled: bool = True
    min_samples: int = 200            # 低于此样本量不训练（宁缺勿滥）
    hidden: List[int] = field(default_factory=lambda: [12, 12])
    epochs: int = 120                 # 最大轮数（全批量 Adam）
    learning_rate: float = 0.01
    val_folds: int = 4                # 按 stage 分组的折数；stage 不足自动降级
    min_r2: float = 0.5               # 验证 R² 下限，低于此值报告不出预测结论
    max_samples: int = 4000           # 单级样本封顶（按 stage 分层抽样）
    max_train_seconds: float = 150.0  # 训练时长硬预算（纯 Python，见 docs/06 §2 基准）
    save_path: str = "results/qos_model.npz"


@dataclass
class AnalysisConfig:
    """结论判定参数——人工价值目标（只作约束，边界由实测产出）。

    sla_target / slo_target 是**价值目标**，不是门限的来源：
    - sla_target：可用性目标（几个 9，0.999 = 3 个 9），用于筛选 SLA 达标
      前缀；SLA 的边界值（A1 拐点）由压测实测产出，见
      ``metrics.find_service_levels`` 与报告"实测服务等级"结论；
    - slo_target：质量达标率目标，用于筛选 SLO 达标前缀；SLO 的边界值
      （A3 拐点）同样由实测产出。

    A2/A3 的判定尺子（时延门限）来自 ``slo.target_mode``：默认
    baseline_relative（基线实测 p99 × 退化预算系数），absolute 为业务
    契约验收模式。人工因此只需提两类价值目标：可用性几个 9、质量退化
    预算——不提任何绝对时延毫秒数。
    """

    sla_target: float = 0.999         # 3 个 9
    slo_target: float = 0.95
    min_samples_per_stage: int = 60   # 样本少于此值的级别不参与"达标/不达标"判定
    # 注意：闭环模式下低并发的样本产出速率受单请求 E2E 限制
    # （并发 1、E2E≈3.5s 时 60s 仅约 17 个样本），因此 runner 会在
    # 测量段样本不足时自动延长本级（上限 load.max_stage_duration_s）；
    # 仍不足的级别以"低置信度参考"进入结论，不直接丢弃。
    #
    # baseline_relative 模式下每个【长度桶】都要单独达到样本量才不回退
    # absolute 门限，因此基线阶段用独立的（通常更小的）门限：按权重
    # [0.2, 0.5, 0.3] 反推，最短桶要 floor 个样本共需 floor/min(权重)
    # 个总样本（默认 30 → 150 个）。基线阶段的自适应延长以此判据驱动，
    # 结束时会给出"还需多少样本/多少秒"的量化提示。
    baseline_min_samples: int = 30


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
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    qos_model: QosModelConfig = field(default_factory=QosModelConfig)
    source_path: str = ""
    # 本次运行中由 max_context_tokens 自动推导的字段名（如 ["prompt_token_targets"]）；
    # 空列表表示长度参数全部为显式配置。仅用于日志/报告标注，可审计。
    derived_lengths: List[str] = field(default_factory=list)

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
            "max_stage_duration_s": self.load.max_stage_duration_s,
            "warmup_s": self.load.warmup_s,
            "ttft_target_s": self.slo.ttft_target_s,
            "tpot_target_ms": self.slo.tpot_target_ms,
            "target_mode": self.slo.target_mode,
            "ttft_factor": self.slo.ttft_factor,
            "tpot_factor": self.slo.tpot_factor,
            "length_buckets": list(self.requests.length_buckets),
            "baseline_enabled": self.load.baseline_enabled,
            "baseline_concurrency": self.load.baseline_concurrency,
            "baseline_duration_s": self.load.baseline_duration_s,
            "sla_target": self.analysis.sla_target,
            "slo_target": self.analysis.slo_target,
            "min_samples_per_stage": self.analysis.min_samples_per_stage,
            "baseline_min_samples": self.analysis.baseline_min_samples,
            "watermark_enabled": self.watermark.enabled,
            "watermark_soft_slo_violation_ratio": self.watermark.soft_slo_violation_ratio,
            "watermark_hard_slo_violation_ratio": self.watermark.hard_slo_violation_ratio,
            "watermark_hard_reject_ratio": self.watermark.hard_reject_ratio,
            "watermark_soft_ttft_p99_ms": self.watermark.soft_ttft_p99_ms,
            "qos_model_enabled": self.qos_model.enabled,
            "qos_model_min_samples": self.qos_model.min_samples,
            "qos_model_hidden": list(self.qos_model.hidden),
            "qos_model_save_path": self.qos_model.save_path,
            "prompt_token_targets": list(self.requests.prompt_token_targets),
            "output_token_choices": list(self.requests.output_token_choices),
            "max_context_tokens": self.requests.max_context_tokens,
            "derived_lengths": list(self.derived_lengths),
        }


# ---------------------------------------------------------------------- #
# 加载
# ---------------------------------------------------------------------- #

def _require(mapping: Dict[str, Any], key: str, typ: type, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"[{where}] 缺少必填字段: {key}")
    return _coerce(mapping[key], typ, key, where)


def _coerce(value: Any, typ: type, key: str, where: str) -> Any:
    """字段类型校验与宽容策略。

    TOML 区分整数字面量与浮点字面量，但数值语义上 150 与 150.0 等价，
    因此 float 字段接受 int 并显式转 float（否则用户写
    max_train_seconds = 150 会被误判"类型应为 float，实际为 int"，
    而 config.example.toml 自己就曾这样写）。int 字段保持严格：
    epochs = 120.5 之类应当报错而不是静默截断。

    bool 必须显式排除：isinstance(True, int) 为 True，不拦住的话
    min_samples = true 会静默变成 1。
    """
    if typ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"[{where}] 字段 {key} 类型应为 float，实际为 {type(value).__name__}")
        return float(value)
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"[{where}] 字段 {key} 类型应为 int，实际为 {type(value).__name__}")
        return value
    if not isinstance(value, typ):
        raise ConfigError(f"[{where}] 字段 {key} 类型应为 {typ.__name__}，实际为 {type(value).__name__}")
    return value


def _optional(mapping: Dict[str, Any], key: str, default: Any, typ: type, where: str) -> Any:
    if key not in mapping or mapping[key] is None:
        return default
    return _coerce(mapping[key], typ, key, where)


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


def _field_default(cls: type, name: str) -> Any:
    """取 dataclass 字段的默认值。

    不能写成 `cls.name`：带 default_factory 的字段在类上没有该属性
    （只有配置里显式写了对应键、走 `or` 短路时才会掩盖这个错误）。
    """
    f = cls.__dataclass_fields__[name]
    if f.default_factory is not MISSING:
        return f.default_factory()
    return f.default


# ---------------------------------------------------------------------- #
# 长度参数自动推导（未显式配置时按 max_context_tokens 缩放）
# ---------------------------------------------------------------------- #

def _derive_prompt_mix(max_context: int) -> tuple:
    """按上下文上限推导默认 prompt 长度分布：(targets, weights)。

    取 [ctx/64, ctx/16, ctx/4] 三档 + "短多长少"权重 [0.2, 0.5, 0.3]，
    各档设下限（32/64/128）防止极小上下文退化出无意义长度。
    32768 → ([512, 2048, 8192], [0.2, 0.5, 0.3])，与历史默认一致。
    """
    plan = ((64, 0.2, 32), (16, 0.5, 64), (4, 0.3, 128))  # (分母, 权重, 下限)
    targets = [max(floor, max_context // divisor) for divisor, _, floor in plan]
    weights = [weight for _, weight, _ in plan]
    return targets, weights


def _derive_length_buckets(max_context: int) -> List[int]:
    """按上下文上限推导默认分桶上界 [ctx/32, ctx/8]。

    32768 → [1024, 4096]，与历史默认一致；8K → [256, 1024]。
    两档设下限（64/256）；由于 ctx/8 > ctx/32 恒成立，推导结果必然严格递增。
    """
    return [max(64, max_context // 32), max(256, max_context // 8)]


def _resolve_length_defaults(cfg: Config) -> None:
    """长度参数缺省值解析：留空的 prompt 分布/分桶按 max_context_tokens 推导。

    放在加载后、校验前，保证下游（PromptFactory / baseline / report）读到的
    永远是具体值。显式配置永远优先——长度分布应镜像生产流量画像，
    max_context_tokens 只是能力上限，不能代替流量假设。
    """
    r = cfg.requests
    derived: List[str] = []
    if not r.prompt_token_targets:
        r.prompt_token_targets, r.prompt_weights = _derive_prompt_mix(r.max_context_tokens)
        derived.append("prompt_token_targets")
    elif not r.prompt_weights:
        # 显式配了长度没配权重：等权，避免长度不一致直接报错
        r.prompt_weights = [1.0 / len(r.prompt_token_targets)] * len(r.prompt_token_targets)
        derived.append("prompt_weights")
    if not r.length_buckets:
        r.length_buckets = _derive_length_buckets(r.max_context_tokens)
        derived.append("length_buckets")
    cfg.derived_lengths = derived


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
    wm_raw = raw.get("watermark", {})
    qm_raw = raw.get("qos_model", {})

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
        target_mode=_optional(slo_raw, "target_mode", SloConfig.target_mode, str, "slo"),
        ttft_factor=_optional(slo_raw, "ttft_factor", SloConfig.ttft_factor, float, "slo"),
        tpot_factor=_optional(slo_raw, "tpot_factor", SloConfig.tpot_factor, float, "slo"),
        min_completion_tokens=_optional(slo_raw, "min_completion_tokens", SloConfig.min_completion_tokens, int, "slo"),
    )

    steps = _int_list(load_raw, "concurrency_steps", "load") or _field_default(LoadConfig, "concurrency_steps")
    load = LoadConfig(
        mode=_optional(load_raw, "mode", "closed", str, "load"),
        concurrency_steps=sorted(set(steps)),
        stage_duration_s=_optional(load_raw, "stage_duration_s", LoadConfig.stage_duration_s, float, "load"),
        warmup_s=_optional(load_raw, "warmup_s", LoadConfig.warmup_s, float, "load"),
        think_time_s=_optional(load_raw, "think_time_s", LoadConfig.think_time_s, float, "load"),
        max_stage_duration_s=_optional(
            load_raw, "max_stage_duration_s", LoadConfig.max_stage_duration_s, float, "load"
        ),
        max_requests_per_stage=_optional(load_raw, "max_requests_per_stage", LoadConfig.max_requests_per_stage, int, "load"),
        max_warmup_requests_per_stage=_optional(
            load_raw, "max_warmup_requests_per_stage", LoadConfig.max_warmup_requests_per_stage, int, "load"
        ),
        drain_timeout_s=_optional(load_raw, "drain_timeout_s", LoadConfig.drain_timeout_s, float, "load"),
        target_rps=_optional(load_raw, "target_rps", 0.0, float, "load"),
        max_inflight=_optional(load_raw, "max_inflight", LoadConfig.max_inflight, int, "load"),
        baseline_enabled=_optional(load_raw, "baseline_enabled", LoadConfig.baseline_enabled, bool, "load"),
        baseline_concurrency=_optional(
            load_raw, "baseline_concurrency", LoadConfig.baseline_concurrency, int, "load"
        ),
        baseline_duration_s=_optional(
            load_raw, "baseline_duration_s", LoadConfig.baseline_duration_s, float, "load"
        ),
        baseline_warmup_s=_optional(
            load_raw, "baseline_warmup_s", LoadConfig.baseline_warmup_s, float, "load"
        ),
    )

    requests = RequestConfig(
        prompt_token_targets=_int_list(req_raw, "prompt_token_targets", "requests") or [],
        prompt_weights=_num_list(req_raw, "prompt_weights", "requests") or [],
        max_context_tokens=_optional(req_raw, "max_context_tokens", RequestConfig.max_context_tokens, int, "requests"),
        output_token_choices=_int_list(req_raw, "output_token_choices", "requests")
        or _field_default(RequestConfig, "output_token_choices"),
        output_weights=_num_list(req_raw, "output_weights", "requests")
        or _field_default(RequestConfig, "output_weights"),
        temperature=_optional(req_raw, "temperature", RequestConfig.temperature, float, "requests"),
        stream=_optional(req_raw, "stream", RequestConfig.stream, bool, "requests"),
        unique_prefix=_optional(req_raw, "unique_prefix", RequestConfig.unique_prefix, bool, "requests"),
        seed=_optional(req_raw, "seed", RequestConfig.seed, int, "requests"),
        chars_per_token=_optional(req_raw, "chars_per_token", RequestConfig.chars_per_token, float, "requests"),
        length_buckets=_int_list(req_raw, "length_buckets", "requests") or [],
    )

    analysis = AnalysisConfig(
        sla_target=_optional(ana_raw, "sla_target", AnalysisConfig.sla_target, float, "analysis"),
        slo_target=_optional(ana_raw, "slo_target", AnalysisConfig.slo_target, float, "analysis"),
        min_samples_per_stage=_optional(ana_raw, "min_samples_per_stage", AnalysisConfig.min_samples_per_stage, int, "analysis"),
        baseline_min_samples=_optional(ana_raw, "baseline_min_samples", AnalysisConfig.baseline_min_samples, int, "analysis"),
    )

    storage = StorageConfig(db_path=str(sto_raw.get("db_path", "")))
    report = ReportConfig(output_path=str(rep_raw.get("output_path", "")))

    watermark = WatermarkConfig(
        enabled=_optional(wm_raw, "enabled", WatermarkConfig.enabled, bool, "watermark"),
        soft_slo_violation_ratio=_optional(
            wm_raw, "soft_slo_violation_ratio", WatermarkConfig.soft_slo_violation_ratio, float, "watermark"
        ),
        hard_slo_violation_ratio=_optional(
            wm_raw, "hard_slo_violation_ratio", WatermarkConfig.hard_slo_violation_ratio, float, "watermark"
        ),
        hard_reject_ratio=_optional(
            wm_raw, "hard_reject_ratio", WatermarkConfig.hard_reject_ratio, float, "watermark"
        ),
        soft_ttft_p99_ms=_optional(
            wm_raw, "soft_ttft_p99_ms", WatermarkConfig.soft_ttft_p99_ms, float, "watermark"
        ),
    )

    qos_model = QosModelConfig(
        enabled=_optional(qm_raw, "enabled", QosModelConfig.enabled, bool, "qos_model"),
        min_samples=_optional(qm_raw, "min_samples", QosModelConfig.min_samples, int, "qos_model"),
        hidden=_int_list(qm_raw, "hidden", "qos_model")
        or _field_default(QosModelConfig, "hidden"),
        epochs=_optional(qm_raw, "epochs", QosModelConfig.epochs, int, "qos_model"),
        learning_rate=_optional(qm_raw, "learning_rate", QosModelConfig.learning_rate, float, "qos_model"),
        val_folds=_optional(qm_raw, "val_folds", QosModelConfig.val_folds, int, "qos_model"),
        min_r2=_optional(qm_raw, "min_r2", QosModelConfig.min_r2, float, "qos_model"),
        max_samples=_optional(qm_raw, "max_samples", QosModelConfig.max_samples, int, "qos_model"),
        max_train_seconds=_optional(
            qm_raw, "max_train_seconds", QosModelConfig.max_train_seconds, float, "qos_model"
        ),
        save_path=_optional(qm_raw, "save_path", QosModelConfig.save_path, str, "qos_model"),
    )

    cfg = Config(
        service=service,
        slo=slo,
        load=load,
        requests=requests,
        analysis=analysis,
        storage=storage,
        report=report,
        watermark=watermark,
        qos_model=qos_model,
        source_path=str(p),
    )
    _resolve_length_defaults(cfg)
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
    if l.max_stage_duration_s < l.stage_duration_s:
        raise ConfigError("load.max_stage_duration_s 不能小于 load.stage_duration_s")
    if l.max_requests_per_stage <= 0:
        raise ConfigError("load.max_requests_per_stage 必须为正数")
    if l.max_warmup_requests_per_stage <= 0:
        raise ConfigError("load.max_warmup_requests_per_stage 必须为正数")
    if l.mode == "open" and l.target_rps <= 0:
        raise ConfigError("open 模式下必须配置 load.target_rps > 0")

    if r.max_context_tokens <= 0:
        raise ConfigError("requests.max_context_tokens 必须为正整数（它是长度自动推导的基准）")
    if not r.prompt_token_targets or any(t <= 0 for t in r.prompt_token_targets):
        raise ConfigError(
            "requests.prompt_token_targets 必须为非空正整数数组"
            "（留空则按 max_context_tokens 自动推导 [ctx/64, ctx/16, ctx/4]）"
        )
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
        hint = (
            "（当前 prompt 长度为按 max_context_tokens 自动推导的默认值，"
            "可显式配置 requests.prompt_token_targets 覆盖）"
            if "prompt_token_targets" in cfg.derived_lengths
            else "请下调长度或修正上下文上限"
        )
        raise ConfigError(
            f"最长请求（prompt {max(r.prompt_token_targets)} + output {max(r.output_token_choices)} "
            f"= {longest}）超过 max_context_tokens={r.max_context_tokens}，{hint}"
        )

    if not (0.0 < a.sla_target <= 1.0):
        raise ConfigError("analysis.sla_target 必须在 (0, 1] 区间")
    if not (0.0 < a.slo_target <= 1.0):
        raise ConfigError("analysis.slo_target 必须在 (0, 1] 区间")
    if a.min_samples_per_stage <= 0:
        raise ConfigError("analysis.min_samples_per_stage 必须为正数")
    if a.baseline_min_samples <= 0:
        raise ConfigError("analysis.baseline_min_samples 必须为正数")

    # --- 水位判定参数 ---
    w = cfg.watermark
    for name in ("soft_slo_violation_ratio", "hard_slo_violation_ratio", "hard_reject_ratio"):
        if not 0.0 < getattr(w, name) <= 1.0:
            raise ConfigError(f"watermark.{name} 必须在 (0, 1] 区间")
    if w.soft_slo_violation_ratio >= w.hard_slo_violation_ratio:
        raise ConfigError(
            "watermark.soft_slo_violation_ratio 必须小于 hard_slo_violation_ratio"
            "（软门限 ≥ 硬门限时 BUSY 永不可能出现，状态机退化为两档）"
        )
    if w.soft_ttft_p99_ms < 0:
        raise ConfigError("watermark.soft_ttft_p99_ms 不能为负（0 表示关闭该判据）")

    # --- 预测模型参数 ---
    q = cfg.qos_model
    if q.min_samples <= 0:
        raise ConfigError("qos_model.min_samples 必须为正数")
    if not q.hidden or any(h <= 0 for h in q.hidden):
        raise ConfigError("qos_model.hidden 必须为非空正整数数组（隐藏层宽度）")
    if len(q.hidden) > 4:
        raise ConfigError("qos_model.hidden 超过 4 个隐藏层时纯 Python 训练代价过高")
    if q.epochs <= 0:
        raise ConfigError("qos_model.epochs 必须为正数")
    if q.learning_rate <= 0:
        raise ConfigError("qos_model.learning_rate 必须为正数")
    if q.val_folds <= 0:
        raise ConfigError("qos_model.val_folds 必须为正数")
    if not 0.0 <= q.min_r2 <= 1.0:
        raise ConfigError("qos_model.min_r2 必须在 [0, 1] 区间")
    if q.max_samples < q.min_samples:
        raise ConfigError("qos_model.max_samples 不能小于 min_samples")
    if q.max_train_seconds <= 0:
        raise ConfigError("qos_model.max_train_seconds 必须为正数")
    if q.enabled and not str(q.save_path).strip():
        raise ConfigError("qos_model.enabled=true 时 qos_model.save_path 不能为空")

    # --- SLO 门限来源 ---
    if cfg.slo.target_mode not in ("absolute", "baseline_relative"):
        raise ConfigError(
            f"slo.target_mode 仅支持 absolute / baseline_relative: {cfg.slo.target_mode}"
        )
    if cfg.slo.ttft_factor < 1.0 or cfg.slo.tpot_factor < 1.0:
        raise ConfigError("slo.ttft_factor / tpot_factor 必须 >= 1.0（小于 1 会把正常请求判为不达标）")
    if cfg.slo.target_mode == "baseline_relative" and not l.baseline_enabled:
        raise ConfigError("slo.target_mode=baseline_relative 时 load.baseline_enabled 不能为 false")

    # --- Phase 0 基线参数 ---
    if l.baseline_concurrency <= 0:
        raise ConfigError("load.baseline_concurrency 必须为正整数")
    if l.baseline_duration_s <= 0:
        raise ConfigError("load.baseline_duration_s 必须为正数")
    if l.baseline_warmup_s < 0:
        raise ConfigError("load.baseline_warmup_s 不能为负")
    if l.baseline_warmup_s >= l.baseline_duration_s:
        raise ConfigError("load.baseline_warmup_s 必须小于 load.baseline_duration_s")
    if l.baseline_concurrency >= min(l.concurrency_steps) and l.baseline_enabled:
        # 不阻断：用户可能刻意如此；但基线会含排队，p99 虚高导致门限虚松
        pass

    # --- 长度分桶 ---
    if not r.length_buckets:
        raise ConfigError(
            "requests.length_buckets 不能为空（留空则按 max_context_tokens 自动推导 [ctx/32, ctx/8]）"
        )
    if any(b <= 0 for b in r.length_buckets):
        raise ConfigError("requests.length_buckets 必须全为正整数")
    if any(b2 <= b1 for b1, b2 in zip(r.length_buckets, r.length_buckets[1:])):
        raise ConfigError("requests.length_buckets 必须严格递增")

    if cfg.slo.ttft_target_s <= 0 or cfg.slo.tpot_target_ms <= 0:
        raise ConfigError("slo.ttft_target_s / tpot_target_ms 必须为正数")
