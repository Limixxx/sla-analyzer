"""基础统计工具：分位数与均值。

单独成模块以避免 metrics 与 baseline 之间的循环依赖。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """最近秩（nearest-rank）分位数；q 取值 (0, 1]。

    定义：第 q 分位数 = 排序后第 ceil(q × N) 小的值（N 为样本数）。
    该定义不插值，样本量变化时分位数只会在新样本跨过秩边界时跳变，
    适合作为门限来源——不会因个别样本插入而被"平均"拉偏。
    """
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    if q <= 0:
        return float(ordered[0])
    if q >= 1:
        return float(ordered[-1])
    # epsilon 抵御 0.99*100 这类浮点误差（真值恰为整数时略微下抬）
    rank = max(1, min(n, math.ceil(q * n - 1e-9)))
    return float(ordered[rank - 1])


def mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None
