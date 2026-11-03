"""报告格式化原语（百分比 / 毫秒 / 数值 / Markdown 表格 / 比率）。

从 ``report.py`` 抽出的小模块（docs/07 §7.10.6）：在线报告与吞吐报告
（``report_batch.py``）共享同一套格式化，避免两套渲染器各写一份、
漂移后对不上。本模块只做纯字符串变换，不 import 任何 bench 内部模块，
处于依赖图最底层（report.py 与 report_batch.py 都向下依赖它）。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence


def pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def ms(value: Optional[float]) -> str:
    return f"{value:.0f}" if value is not None else "n/a"


def num(value: Optional[float], digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def nines_suffix(target: float) -> str:
    """0.999 → "（3 个 9）"；不是整数个 9 时返回空串（不硬凑）。"""
    if target >= 1.0:
        return ""
    nines = -math.log10(1.0 - target)
    nearest = round(nines)
    if nearest >= 1 and abs(nines - nearest) < 0.05:
        return f"（{nearest} 个 9）"
    return ""


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def ratio(current: Optional[float], base: Optional[float]) -> str:
    if current is None or base is None or base <= 0:
        return "n/a"
    return f"×{current / base:.2f}"
