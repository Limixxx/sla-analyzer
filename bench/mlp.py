"""纯 Python 多层感知机（全批量 Adam，ReLU 隐藏层 + 线性输出）。

规模假设：样本 ≤ 1e4、隐藏层 ≤ 32、层数 ≤ 4——此规模下纯 Python 训练在
分钟级可完成（见 docs/06-qos-model.md §2）。接口保持最小：训练、预测、
权重导入导出；不做自动微分，反向传播手工推导。
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from operator import mul as _mul
from typing import Callable, List, Optional, Sequence, Tuple


class MLP:
    """sizes 如 [3, 12, 12, 2]：输入 3 维、两个隐藏层、输出 2 维。"""

    def __init__(self, sizes: Sequence[int], seed: int = 0):
        if len(sizes) < 3:
            raise ValueError("MLP 至少需要输入层 + 1 个隐藏层 + 输出层")
        if any(s <= 0 for s in sizes):
            raise ValueError("层宽必须为正")
        self.sizes = [int(s) for s in sizes]
        rng = random.Random(seed)
        self.W: List[List[List[float]]] = []
        self.b: List[List[float]] = []
        for fan_in, fan_out in zip(self.sizes, self.sizes[1:]):
            # Xavier 均匀初始化
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            self.W.append(
                [[rng.uniform(-limit, limit) for _ in range(fan_in)] for _ in range(fan_out)]
            )
            self.b.append([0.0] * fan_out)

    # ------------------------------------------------------------------ #
    def predict(self, x: Sequence[float]) -> List[float]:
        """单样本前向推理。"""
        acts = list(x)
        n_layers = len(self.W)
        for li in range(n_layers):
            W, b = self.W[li], self.b[li]
            z = [sum(w * a for w, a in zip(row, acts)) + bias for row, bias in zip(W, b)]
            acts = z if li == n_layers - 1 else [v if v > 0.0 else 0.0 for v in z]
        return acts

    def train(
        self,
        X: Sequence[Sequence[float]],
        Y: Sequence[Sequence[float]],
        epochs: int = 200,
        lr: float = 0.01,
        deadline: Optional[float] = None,
        on_epoch: Optional[Callable[[int, float], None]] = None,
    ) -> List[float]:
        """全批量 Adam 最小化 MSE。deadline 为 time.monotonic() 绝对时刻。

        返回每轮损失；超 deadline 时停止并保留已训权重。
        """
        n = len(X)
        if n == 0:
            raise ValueError("训练集为空")
        dim_in, dim_out = len(X[0]), len(Y[0])
        if dim_in != self.sizes[0] or dim_out != self.sizes[-1]:
            raise ValueError("数据维度与网络结构不匹配")

        # Adam 状态
        mW = [[[0.0] * len(row) for row in W] for W in self.W]
        vW = [[[0.0] * len(row) for row in W] for W in self.W]
        mb = [[0.0] * len(b) for b in self.b]
        vb = [[0.0] * len(b) for b in self.b]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        history: List[float] = []

        for epoch in range(1, epochs + 1):
            t += 1
            # ---- 前向：累积激活（用于反向）与损失 ----
            acts_all: List[List[List[float]]] = []  # 每层激活（含输入层）
            a_in = [list(x) for x in X]
            acts_all.append(a_in)
            total_loss = 0.0
            n_layers = len(self.W)
            for li in range(n_layers):
                W, b = self.W[li], self.b[li]
                prev = acts_all[li]
                z = [
                    [sum(map(_mul, row, sample)) + bias for row, bias in zip(W, b)]
                    for sample in prev
                ]
                if li == n_layers - 1:
                    a = z
                    for zi, yi in zip(z, Y):
                        for zo, yo in zip(zi, yi):
                            d = zo - yo
                            total_loss += d * d
                else:
                    a = [[v if v > 0.0 else 0.0 for v in row] for row in z]
                acts_all.append(a)
            history.append(total_loss / (n * dim_out))

            # ---- 反向：输出层 delta = 2*(a - y)/N ----
            last = acts_all[-1]
            delta = [
                [2.0 * (a - y) / (n * dim_out) for a, y in zip(ai, yi)] for ai, yi in zip(last, Y)
            ]
            # 逐层回传（从后往前）
            for li in range(n_layers - 1, -1, -1):
                prev = acts_all[li]  # 本层输入
                W, b = self.W[li], self.b[li]
                d = delta
                # 列预取：d 的每列 = 各样本在该输出单元的 delta
                d_cols = [[row[o] for row in d] for o in range(len(b))]
                p_cols = [[row[i] for row in prev] for i in range(len(W[0]))]
                # 梯度 gW[o][i] = Σ_n delta[n][o]·prev[n][i]
                gW = [
                    [sum(map(_mul, d_col, p_col)) for p_col in p_cols]
                    for d_col in d_cols
                ]
                gb = [sum(d_col) for d_col in d_cols]
                # 回传到上一层的 delta（隐藏层，过 ReLU 导数）
                if li > 0:
                    w_cols = [[W[o][i] for o in range(len(W))] for i in range(len(W[0]))]
                    new_delta = [
                        [
                            (sum(map(_mul, w_col, d_n)) if prev_act[i] > 0.0 else 0.0)
                            for i, w_col in enumerate(w_cols)
                        ]
                        for d_n, prev_act in zip(d, prev)
                    ]
                    delta = new_delta
                # Adam 更新
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                for o in range(len(W)):
                    row, grow, mrow, vrow = W[o], gW[o], mW[li][o], vW[li][o]
                    for i in range(len(row)):
                        mrow[i] = beta1 * mrow[i] + (1 - beta1) * grow[i]
                        vrow[i] = beta2 * vrow[i] + (1 - beta2) * grow[i] * grow[i]
                        mh = mrow[i] / bc1
                        vh = vrow[i] / bc2
                        row[i] -= lr * mh / (math.sqrt(vh) + eps)
                for o in range(len(b)):
                    mbo, vbo, gbo = mb[li][o], vb[li][o], gb[o]
                    mbo = beta1 * mbo + (1 - beta1) * gbo
                    vbo = beta2 * vbo + (1 - beta2) * gbo * gbo
                    mb[li][o], vb[li][o] = mbo, vbo
                    b[o] -= lr * (mbo / bc1) / (math.sqrt(vbo / bc2) + eps)

            if on_epoch is not None and (epoch % 20 == 0 or epoch == 1):
                on_epoch(epoch, history[-1])
            if deadline is not None and time.monotonic() > deadline:
                break
        return history

    # ------------------------------------------------------------------ #
    def params_flat(self) -> Tuple[List[List[List[float]]], List[List[float]]]:
        return self.W, self.b

    def load_params_flat(self, W: List[List[List[float]]], b: List[List[float]]) -> None:
        if len(W) != len(self.W) or any(
            len(a) != len(c) or len(a[0]) != len(c[0])
            for a, c in zip(W, self.W)
        ):
            raise ValueError("权重形状与网络结构不匹配")
        self.W = [[list(map(float, row)) for row in layer] for layer in W]
        self.b = [list(map(float, bias)) for bias in b]


@dataclass
class TrainTrace:
    """一次训练的过程记录（供报告/调试）。"""

    epochs_run: int = 0
    final_loss: float = 0.0
    stopped_by_deadline: bool = False
    seconds: float = 0.0
    history: List[float] = field(default_factory=list)
