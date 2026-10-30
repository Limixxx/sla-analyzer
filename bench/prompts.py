"""按 token 预算合成压测 prompt。

不依赖外部 tokenizer：用可配置的 chars_per_token 做估算来构造确定性的 prompt，
真实 token 数以服务端 usage 字段为准并落库（prompt_tokens vs prompt_planned_tokens），
两者偏差可在报告中复盘，必要时再校准 chars_per_token。
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import List, Tuple

from .config import RequestConfig

_FILLER_SENTENCES: Tuple[str, ...] = (
    "The inference gateway routes each request through a prefill worker and a decode worker.",
    "Load balancing decisions depend on the observed queue depth and the kv cache utilization.",
    "A disaggregated architecture separates prefill compute from decode compute for better efficiency.",
    "Time to first token measures how quickly the system starts responding to the user request.",
    "Time per output token captures the steady state decoding speed of the serving cluster.",
    "Service level objectives are defined over the fraction of requests meeting both targets.",
    "Availability includes slow but successful responses, while quality excludes them entirely.",
    "The scheduler places replicas according to node affinity and remaining accelerator memory.",
    "Bootstrap ports carry the kv transfer endpoints between prefill and decode instances.",
    "Each streaming chunk arrives as a server sent event with an incremental content delta.",
    "Prompt length distributions should mirror the traffic mix of the production workload.",
    "Repeated measurements at increasing concurrency reveal the saturation knee of the system.",
    "Rejections under overload are counted as failures in the availability metric.",
    "The router buffers requests briefly and rejects them once its internal queue is full.",
    "Percentile latency is often more informative than the average for tail sensitive services.",
)


@dataclass
class RequestSpec:
    """一次请求的特征规格（落库时只记录这些特征，不记录原文）。"""

    prompt_tokens_target: int
    max_tokens: int


class PromptFactory:
    """确定性 prompt 生成器：相同 seed 与序号产生相同 prompt。"""

    def __init__(self, cfg: RequestConfig):
        self._cfg = cfg
        self._rng = random.Random(cfg.seed)
        self._seq = 0

    def sample_spec(self) -> RequestSpec:
        prompt_tokens = self._weighted_choice(self._cfg.prompt_token_targets, self._cfg.prompt_weights)
        max_tokens = self._weighted_choice(self._cfg.output_token_choices, self._cfg.output_weights)
        return RequestSpec(prompt_tokens_target=int(prompt_tokens), max_tokens=int(max_tokens))

    def build(self, spec: RequestSpec) -> str:
        """构造约等于 spec.prompt_tokens_target 的 prompt 文本。"""
        self._seq += 1
        cfg = self._cfg
        target_chars = max(1, int(spec.prompt_tokens_target * cfg.chars_per_token))

        header = f"benchmark-request seq={self._seq} nonce={uuid.uuid4().hex[:12]}\n\n"
        if not cfg.unique_prefix:
            header = f"benchmark-request seq={self._seq}\n\n"

        parts: List[str] = [header]
        size = len(header)
        idx = 0
        while size < target_chars:
            sentence = _FILLER_SENTENCES[idx % len(_FILLER_SENTENCES)]
            idx += 1
            # 每轮附加一个可变编号，避免完全重复的文本被上游去重/缓存
            chunk = f"{sentence} (case {self._rng.randrange(100000):05d})\n"
            parts.append(chunk)
            size += len(chunk)
        text = "".join(parts)
        return text[:target_chars]

    # ------------------------------------------------------------------ #
    def _weighted_choice(self, values: List[int], weights: List[float]) -> int:
        total = float(sum(weights))
        if total <= 0:
            return values[0]
        pick = self._rng.random() * total
        acc = 0.0
        for value, weight in zip(values, weights):
            acc += weight
            if pick <= acc:
                return value
        return values[-1]
