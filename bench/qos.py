"""服务质量预测模型（MLP）：从压测采样学习 (并发, prompt, completion) → (TTFT, TPOT)。

设计纪律（docs/06-qos-model.md）：
1. A3 样本必须入训——容量膝点区只有 A3；warmup/aborted/A1 剔除；
2. 按 stage 分组 K 折交叉验证——同级样本共享并发特征，随机划分会泄漏；
3. 残差 ECDF 校准——折外残差汇成经验分布，分位数与通过率不假设分布族；
4. 包络守卫——超出训练覆盖的（并发 × 长度）一律标记外推，不参与结论。

预测的是"请求被服务时的质量"（TTFT/TPOT 分布），**不预测失败/拒绝概率**：
A1 样本已被剔除，可用性（SLA）侧不可由本模型给出。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import QosModelConfig
from .mlp import MLP
from .npz import load_npz, save_npz

FEATURES: Tuple[str, ...] = ("concurrency", "prompt_tokens", "completion_tokens")
TARGETS: Tuple[str, ...] = ("ttft", "tpot")
_RESID_CAP = 4000          # 折外残差池封顶（系统抽样，保证枚举成本可控）
_ENVELOPE_TOL = 0.05       # 包络容差：超出训练极值 5% 即判外推
_MODEL_VERSION = 1


# ---------------------------------------------------------------------- #
# 数据
# ---------------------------------------------------------------------- #
@dataclass
class Sample:
    stage_id: int
    concurrency: int
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    tpot_ms: float


@dataclass
class MixPoint:
    """流量构成中的一个点（长度桶代表值 + 权重 + 该桶生效门限）。"""

    prompt_tokens: float
    completion_tokens: float
    weight: float
    bucket: int = -1
    ttft_limit_ms: Optional[float] = None   # None = 用调用方默认门限
    tpot_limit_ms: Optional[float] = None


@dataclass
class PassRates:
    ttft: float
    tpot: float
    both: float          # 两者同时达标 = 预测 A2 占比（SLO 口径）


@dataclass
class PointPrediction:
    concurrency: int
    prompt_tokens: float
    completion_tokens: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p90_ms: float
    tpot_p99_ms: float
    extrapolated: bool
    outside: List[str] = field(default_factory=list)


@dataclass
class FoldResult:
    fold: int
    val_stage_ids: List[int]
    n_train: int
    n_val: int
    best_epoch: int
    val_loss: float


@dataclass
class TrainReport:
    """训练过程与验证指标（渲染进报告；usable=False 时报告只给数据口径）。"""

    n_samples: int = 0
    n_stages: int = 0
    folds: List[FoldResult] = field(default_factory=list)
    cross_validated: bool = False
    best_epochs_median: int = 0
    r2_ttft: float = 0.0
    r2_tpot: float = 0.0
    mae_ttft_ms: float = 0.0
    mae_tpot_ms: float = 0.0
    mape_ttft: float = 0.0
    mape_tpot: float = 0.0
    seconds: float = 0.0
    stopped_by_deadline: bool = False
    usable: bool = False
    skip_reason: str = ""
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------- #
# 变换工具
# ---------------------------------------------------------------------- #
def _log(x: float) -> float:
    return math.log(x) if x > 0 else 0.0


def _from_log(y: float) -> float:
    """log 空间还原为毫秒（+1 平滑的逆），负值截断到 0。"""
    return max(0.0, math.exp(y) - 1.0)


def _nearest_rank(sorted_vals: Sequence[float], q: float) -> float:
    """最近秩分位数（与本项目分位数口径一致，不插值）。"""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = max(0, math.ceil(q * n) - 1)
    return sorted_vals[idx]


# ---------------------------------------------------------------------- #
# 模型
# ---------------------------------------------------------------------- #
class QoSModel:
    """训练完成的预测模型：MLP + 归一化 + 折外残差 ECDF + 包络。"""

    def __init__(
        self,
        net: MLP,
        feat_mean: List[float],
        feat_std: List[float],
        tgt_mean: List[float],
        tgt_std: List[float],
        resid_ttft: List[float],
        resid_tpot: List[float],
        feat_min: List[float],
        feat_max: List[float],
        meta: Dict[str, Any],
    ):
        self.net = net
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.tgt_mean = tgt_mean
        self.tgt_std = tgt_std
        self.resid_ttft = resid_ttft        # 与 resid_tpot 按索引配对（联合结构）
        self.resid_tpot = resid_tpot
        self.feat_min = feat_min
        self.feat_max = feat_max
        self.meta = meta
        # 残差的对数因子缓存：时延 = (典型值+1) * exp(r) - 1
        self._resid_factor_ttft = [math.exp(r) for r in resid_ttft]
        self._resid_factor_tpot = [math.exp(r) for r in resid_tpot]
        self._sorted_ttft = sorted(resid_ttft)
        self._sorted_tpot = sorted(resid_tpot)

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #
    @classmethod
    def train(
        cls,
        samples: List[Sample],
        cfg: QosModelConfig,
        service_fingerprint: str = "",
    ) -> Tuple["QoSModel", TrainReport]:
        rep = TrainReport(n_samples=len(samples), n_stages=len({s.stage_id for s in samples}))
        if len(samples) < cfg.min_samples:
            rep.skip_reason = (
                f"有效训练样本 {len(samples)} 个 < min_samples={cfg.min_samples}，不训练"
            )
            return cls._empty(), rep

        deadline = time.monotonic() + cfg.max_train_seconds
        stage_ids = sorted({s.stage_id for s in samples})
        k = min(cfg.val_folds, len(stage_ids)) if len(stage_ids) >= 2 else 1

        # 按 stage 分组：stage 按并发排序后轮流分配，保证每折都覆盖低/高并发
        ordered_stages = sorted(
            stage_ids,
            key=lambda sid: min(s.concurrency for s in samples if s.stage_id == sid),
        )
        fold_of_stage = {sid: i % k for i, sid in enumerate(ordered_stages)}

        # ---- 交叉验证：每折训练 + 记录折外预测 ----
        oof_pred: Dict[int, List[float]] = {}   # sample idx -> [y_ttft_log, y_tpot_log]
        oof_true: Dict[int, List[float]] = {}
        best_epochs: List[int] = []
        if k >= 2:
            rep.cross_validated = True
            for fold in range(k):
                val_stage_ids = [sid for sid in stage_ids if fold_of_stage[sid] == fold]
                train_s = [s for s in samples if fold_of_stage[s.stage_id] != fold]
                val_s = [s for s in samples if fold_of_stage[s.stage_id] == fold]
                if not train_s or not val_s:
                    continue
                norm = _Norm.fit(train_s)
                Xtr = [norm.features(s) for s in train_s]
                Ytr = [norm.targets(s) for s in train_s]
                Xva = [norm.features(s) for s in val_s]
                Yva = [norm.targets(s) for s in val_s]
                net = MLP([len(FEATURES)] + list(cfg.hidden) + [len(TARGETS)], seed=0)
                best_epoch, best_loss, best_snapshot = _train_track_best(
                    net, Xtr, Ytr, Xva, Yva, cfg, deadline
                )
                for s, xv, yv in zip(val_s, Xva, Yva):
                    pred = net.predict(xv)
                    oof_pred[id(s)] = [
                        pred[0] * norm.tgt_std[0] + norm.tgt_mean[0],
                        pred[1] * norm.tgt_std[1] + norm.tgt_mean[1],
                    ]
                    oof_true[id(s)] = [
                        yv[0] * norm.tgt_std[0] + norm.tgt_mean[0],
                        yv[1] * norm.tgt_std[1] + norm.tgt_mean[1],
                    ]
                best_epochs.append(best_epoch)
                rep.folds.append(
                    FoldResult(
                        fold=fold,
                        val_stage_ids=val_stage_ids,
                        n_train=len(train_s),
                        n_val=len(val_s),
                        best_epoch=best_epoch,
                        val_loss=best_loss,
                    )
                )
                if time.monotonic() > deadline:
                    rep.stopped_by_deadline = True
                    break
        else:
            rep.warnings.append(
                "压力级少于 2 个，无法按 stage 分组交叉验证；指标为训练集指标（乐观，仅供参考）"
            )

        # ---- 指标（折外池化；无 CV 时退化为训练集指标） ----
        if oof_pred:
            preds = [oof_pred[i] for i in sorted(oof_pred)]
            trues = [oof_true[i] for i in sorted(oof_pred)]
            _fill_metrics(rep, trues, preds)
            resid_ttft = [t[0] - p[0] for t, p in zip(trues, preds)]
            resid_tpot = [t[1] - p[1] for t, p in zip(trues, preds)]
        else:
            norm_all = _Norm.fit(samples)
            X = [norm_all.features(s) for s in samples]
            Y = [norm_all.targets(s) for s in samples]
            net = MLP([len(FEATURES)] + list(cfg.hidden) + [len(TARGETS)], seed=0)
            net.train(X, Y, epochs=cfg.epochs, lr=cfg.learning_rate, deadline=deadline)
            preds = [net.predict(x) for x in X]
            preds_log = [
                [p[0] * norm_all.tgt_std[0] + norm_all.tgt_mean[0],
                 p[1] * norm_all.tgt_std[1] + norm_all.tgt_mean[1]]
                for p in preds
            ]
            trues_log = [
                [y[0] * norm_all.tgt_std[0] + norm_all.tgt_mean[0],
                 y[1] * norm_all.tgt_std[1] + norm_all.tgt_mean[1]]
                for y in Y
            ]
            _fill_metrics(rep, trues_log, preds_log)
            resid_ttft = [t[0] - p[0] for t, p in zip(trues_log, preds_log)]
            resid_tpot = [t[1] - p[1] for t, p in zip(trues_log, preds_log)]
            best_epochs = [cfg.epochs]

        # ---- 最终模型：全量数据 + 折间最优轮数中位数 ----
        final_epochs = int(sorted(best_epochs)[len(best_epochs) // 2]) if best_epochs else cfg.epochs
        final_epochs = max(1, min(final_epochs, cfg.epochs))
        rep.best_epochs_median = final_epochs
        norm = _Norm.fit(samples)
        X = [norm.features(s) for s in samples]
        Y = [norm.targets(s) for s in samples]
        net = MLP([len(FEATURES)] + list(cfg.hidden) + [len(TARGETS)], seed=0)
        t0 = time.monotonic()
        net.train(X, Y, epochs=final_epochs, lr=cfg.learning_rate, deadline=deadline)
        rep.seconds = time.monotonic() - t0

        # 残差池封顶（系统抽样，保配对）
        if len(resid_ttft) > _RESID_CAP:
            step = len(resid_ttft) / _RESID_CAP
            idx = [int(i * step) for i in range(_RESID_CAP)]
            resid_ttft = [resid_ttft[i] for i in idx]
            resid_tpot = [resid_tpot[i] for i in idx]

        rep.usable = (rep.r2_ttft >= cfg.min_r2 and rep.r2_tpot >= cfg.min_r2)
        if not rep.usable and not rep.skip_reason:
            rep.skip_reason = (
                f"验证 R²（TTFT={rep.r2_ttft:.3f}, TPOT={rep.r2_tpot:.3f}）低于 "
                f"min_r2={cfg.min_r2}"
            )

        model = cls(
            net=net,
            feat_mean=norm.feat_mean,
            feat_std=norm.feat_std,
            tgt_mean=norm.tgt_mean,
            tgt_std=norm.tgt_std,
            resid_ttft=resid_ttft,
            resid_tpot=resid_tpot,
            feat_min=norm.feat_min,
            feat_max=norm.feat_max,
            meta={
                "version": _MODEL_VERSION,
                "features": list(FEATURES),
                "targets": list(TARGETS),
                "hidden": list(cfg.hidden),
                "n_samples": len(samples),
                "n_stages": rep.n_stages,
                "cross_validated": rep.cross_validated,
                "r2_ttft": rep.r2_ttft,
                "r2_tpot": rep.r2_tpot,
                "mae_ttft_ms": rep.mae_ttft_ms,
                "mae_tpot_ms": rep.mae_tpot_ms,
                "mape_ttft": rep.mape_ttft,
                "mape_tpot": rep.mape_tpot,
                "best_epochs_median": rep.best_epochs_median,
                "folds": [
                    {
                        "fold": f.fold,
                        "val_stage_ids": list(f.val_stage_ids),
                        "n_train": f.n_train,
                        "n_val": f.n_val,
                        "best_epoch": f.best_epoch,
                        "val_loss": f.val_loss,
                    }
                    for f in rep.folds
                ],
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "train_seconds": rep.seconds,
                "service": service_fingerprint,
            },
        )
        return model, rep

    @classmethod
    def _empty(cls) -> "QoSModel":
        net = MLP([len(FEATURES)] + [1] + [len(TARGETS)], seed=0)
        return cls(net, [0.0] * 3, [1.0] * 3, [0.0] * 2, [1.0] * 2, [], [],
                   [0.0] * 3, [0.0] * 3, {"version": _MODEL_VERSION, "empty": True})

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    def _features(self, conc: float, prompt: float, completion: float) -> List[float]:
        raw = [_log(conc), _log(prompt), _log(completion)]
        return [(r - m) / s for r, m, s in zip(raw, self.feat_mean, self.feat_std)]

    def _predict_log(self, conc: float, prompt: float, completion: float) -> Tuple[float, float]:
        z = self.net.predict(self._features(conc, prompt, completion))
        return (
            z[0] * self.tgt_std[0] + self.tgt_mean[0],
            z[1] * self.tgt_std[1] + self.tgt_mean[1],
        )

    def _envelope_check(self, conc: float, prompt: float, completion: float) -> List[str]:
        outside: List[str] = []
        for name, value, lo, hi in zip(
            FEATURES, (conc, prompt, completion), self.feat_min, self.feat_max
        ):
            tol_lo = lo * (1.0 - _ENVELOPE_TOL)
            tol_hi = hi * (1.0 + _ENVELOPE_TOL)
            if value < tol_lo or value > tol_hi:
                outside.append(f"{name}={value:g}（训练范围 {lo:g}~{hi:g}）")
        return outside

    def predict(self, conc: float, prompt: float, completion: float) -> PointPrediction:
        """预测该负载点的 TTFT/TPOT 分位数（log 空间预测 + 残差最近秩分位数）。

        分位数 = 典型值 × exp(残差分位数) − 1，其中典型值 = exp(ŷ) − 1（残差 0 处）。
        """
        y_ttft, y_tpot = self._predict_log(conc, prompt, completion)
        base_ttft, base_tpot = math.exp(y_ttft), math.exp(y_tpot)
        outside = self._envelope_check(conc, prompt, completion)

        def _quant(base: float, sorted_resid: Sequence[float], q: float) -> float:
            return max(0.0, base * math.exp(_nearest_rank(sorted_resid, q)) - 1.0)

        return PointPrediction(
            concurrency=int(conc),
            prompt_tokens=prompt,
            completion_tokens=completion,
            ttft_p50_ms=_quant(base_ttft, self._sorted_ttft, 0.50),
            ttft_p90_ms=_quant(base_ttft, self._sorted_ttft, 0.90),
            ttft_p99_ms=_quant(base_ttft, self._sorted_ttft, 0.99),
            tpot_p50_ms=_quant(base_tpot, self._sorted_tpot, 0.50),
            tpot_p90_ms=_quant(base_tpot, self._sorted_tpot, 0.90),
            tpot_p99_ms=_quant(base_tpot, self._sorted_tpot, 0.99),
            extrapolated=bool(outside),
            outside=outside,
        )

    def pass_rates(
        self,
        conc: float,
        prompt: float,
        completion: float,
        ttft_limit_ms: float,
        tpot_limit_ms: float,
    ) -> PassRates:
        """预测 TTFT/TPOT/双达标的通过率。

        用配对残差枚举（保 TTFT/TPOT 残差的联合结构）：
        时延 = (典型值+1)·exp(r) - 1，逐残差对判定是否 ≤ 门限。
        """
        y_ttft, y_tpot = self._predict_log(conc, prompt, completion)
        base_ttft, base_tpot = math.exp(y_ttft) - 1.0, math.exp(y_tpot) - 1.0
        n = len(self.resid_ttft)
        if n == 0:
            return PassRates(0.0, 0.0, 0.0)
        ok_ttft = ok_tpot = ok_both = 0
        for f_ttft, f_tpot in zip(self._resid_factor_ttft, self._resid_factor_tpot):
            ttft = (base_ttft + 1.0) * f_ttft - 1.0
            tpot = (base_tpot + 1.0) * f_tpot - 1.0
            t_ok = ttft <= ttft_limit_ms
            p_ok = tpot <= tpot_limit_ms
            ok_ttft += t_ok
            ok_tpot += p_ok
            ok_both += t_ok and p_ok
        return PassRates(ok_ttft / n, ok_tpot / n, ok_both / n)

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        arrays: Dict[str, Any] = {}
        for li, (W, b) in enumerate(zip(self.net.W, self.net.b), start=1):
            arrays[f"w{li}"] = W
            arrays[f"b{li}"] = b
        arrays["feat_mean"] = self.feat_mean
        arrays["feat_std"] = self.feat_std
        arrays["tgt_mean"] = self.tgt_mean
        arrays["tgt_std"] = self.tgt_std
        arrays["resid_ttft"] = self.resid_ttft
        arrays["resid_tpot"] = self.resid_tpot
        arrays["feat_min"] = self.feat_min
        arrays["feat_max"] = self.feat_max
        meta_bytes = json.dumps(self.meta, ensure_ascii=False).encode("utf-8")
        arrays["meta_json"] = meta_bytes
        save_npz(path, arrays)

    @classmethod
    def load(cls, path: str) -> "QoSModel":
        data = load_npz(path)
        meta_raw = data.get("meta_json")
        if isinstance(meta_raw, (bytes, bytearray)):
            meta = json.loads(bytes(meta_raw).decode("utf-8"))
        else:
            meta = {}
        hidden = list(meta.get("hidden") or [12, 12])
        net = MLP([len(FEATURES)] + hidden + [len(TARGETS)], seed=0)
        W, b = [], []
        li = 1
        while f"w{li}" in data:
            W.append([[float(v) for v in row] for row in data[f"w{li}"]])
            b.append([float(v) for v in data[f"b{li}"]])
            li += 1
        if len(W) != len(net.W):
            raise ValueError(f"模型层数({len(W)})与 meta.hidden({hidden}) 不一致")
        net.load_params_flat(W, b)
        resid_ttft = [float(v) for v in data.get("resid_ttft", [])]
        resid_tpot = [float(v) for v in data.get("resid_tpot", [])]
        if len(resid_ttft) != len(resid_tpot):
            raise ValueError("模型文件损坏：resid_ttft 与 resid_tpot 长度不一致")
        return cls(
            net=net,
            feat_mean=[float(v) for v in data["feat_mean"]],
            feat_std=[float(v) for v in data["feat_std"]],
            tgt_mean=[float(v) for v in data["tgt_mean"]],
            tgt_std=[float(v) for v in data["tgt_std"]],
            resid_ttft=resid_ttft,
            resid_tpot=resid_tpot,
            feat_min=[float(v) for v in data["feat_min"]],
            feat_max=[float(v) for v in data["feat_max"]],
            meta=meta,
        )


# ---------------------------------------------------------------------- #
# 训练辅助
# ---------------------------------------------------------------------- #
class _Norm:
    """特征/目标标准化参数（log 空间）。"""

    def __init__(self, samples: List[Sample]):
        n = len(samples)
        self.feat_mean, self.feat_std = self._stats(
            [[_log(s.concurrency), _log(s.prompt_tokens), _log(s.completion_tokens)] for s in samples]
        )
        self.tgt_mean, self.tgt_std = self._stats(
            [[_log(s.ttft_ms + 1.0), _log(s.tpot_ms + 1.0)] for s in samples]
        )
        raw = [
            [float(s.concurrency), float(s.prompt_tokens), float(s.completion_tokens)]
            for s in samples
        ]
        self.feat_min = [min(col) for col in zip(*raw)]
        self.feat_max = [max(col) for col in zip(*raw)]
        self._n = n

    @staticmethod
    def _stats(rows: List[List[float]]) -> Tuple[List[float], List[float]]:
        cols = len(rows[0])
        mean = [sum(r[c] for r in rows) / len(rows) for c in range(cols)]
        var = [sum((r[c] - mean[c]) ** 2 for r in rows) / len(rows) for c in range(cols)]
        std = [math.sqrt(v) if v > 1e-12 else 1.0 for v in var]
        return mean, std

    @classmethod
    def fit(cls, samples: List[Sample]) -> "_Norm":
        return cls(samples)

    def features(self, s: Sample) -> List[float]:
        raw = [_log(s.concurrency), _log(s.prompt_tokens), _log(s.completion_tokens)]
        return [(r - m) / sd for r, m, sd in zip(raw, self.feat_mean, self.feat_std)]

    def targets(self, s: Sample) -> List[float]:
        raw = [_log(s.ttft_ms + 1.0), _log(s.tpot_ms + 1.0)]
        return [(r - m) / sd for r, m, sd in zip(raw, self.tgt_mean, self.tgt_std)]


def _train_track_best(
    net: MLP,
    X: List[List[float]],
    Y: List[List[float]],
    Xva: List[List[float]],
    Yva: List[List[float]],
    cfg: QosModelConfig,
    deadline: float,
) -> Tuple[int, float, List[Any]]:
    """训练并跟踪验证损失最优轮（快照权重）；用于折间轮数选择。"""

    best_epoch, best_loss = 0, float("inf")
    best_snapshot: List[Any] = []
    state = {"epoch": 0}

    def _val_loss() -> float:
        total = 0.0
        for xv, yv in zip(Xva, Yva):
            pred = net.predict(xv)
            for p, y in zip(pred, yv):
                d = p - y
                total += d * d
        return total / max(1, len(Xva) * len(Yva[0]))

    def _on_epoch(epoch: int, _loss: float) -> None:
        nonlocal best_epoch, best_loss, best_snapshot
        state["epoch"] = epoch
        vl = _val_loss()
        if vl < best_loss:
            best_loss, best_epoch = vl, epoch
            best_snapshot = (
                [[list(row) for row in W] for W in net.W],
                [list(b) for b in net.b],
            )

    net.train(
        X, Y, epochs=cfg.epochs, lr=cfg.learning_rate,
        deadline=deadline, on_epoch=_on_epoch,
    )
    if best_snapshot:
        (W, b) = best_snapshot
        net.load_params_flat(W, b)
    if best_epoch == 0:  # 训练被 deadline 立即截断等极端情况
        best_epoch, best_loss = state["epoch"], _val_loss()
    return best_epoch, best_loss, best_snapshot


def _fill_metrics(rep: TrainReport, trues: List[List[float]], preds: List[List[float]]) -> None:
    """折外（或训练集）池化指标：log 空间 R² + 毫秒空间 MAE/MAPE。"""
    if not trues:
        return
    for ti, name in ((0, "ttft"), (1, "tpot")):
        y = [t[ti] for t in trues]
        p = [q[ti] for q in preds]
        mean_y = sum(y) / len(y)
        ss_tot = sum((v - mean_y) ** 2 for v in y)
        ss_res = sum((a - b) ** 2 for a, b in zip(y, p))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        y_ms = [_from_log(v) for v in y]
        p_ms = [_from_log(v) for v in p]
        mae = sum(abs(a - b) for a, b in zip(y_ms, p_ms)) / len(y_ms)
        mape = (
            sum(abs(a - b) / a for a, b in zip(y_ms, p_ms) if a > 1e-9) / max(1, len(y_ms))
        )
        if name == "ttft":
            rep.r2_ttft, rep.mae_ttft_ms, rep.mape_ttft = r2, mae, mape
        else:
            rep.r2_tpot, rep.mae_tpot_ms, rep.mape_tpot = r2, mae, mape


# ---------------------------------------------------------------------- #
# 数据集构建 / 仿真
# ---------------------------------------------------------------------- #
def build_dataset(store: Any, cfg: QosModelConfig) -> List[Sample]:
    """从数据库构建训练集：正式压力级的非预热 measured 成功请求。

    剔除：warmup / aborted / A1 失败 / 缺时延或长度非正的样本。
    A3（成功但超门限）保留——容量膝点区只有 A3 样本。
    """
    samples: List[Sample] = []
    for stage in store.all_stages(kind="stage"):
        sid = stage["id"]
        conc = stage["concurrency"]
        for row in store.stage_rows(sid, include_warmup=False):
            if row.get("status") != "ok":
                continue
            ttft, tpot = row.get("ttft_ms"), row.get("tpot_ms")
            if not ttft or not tpot or ttft <= 0 or tpot <= 0:
                continue
            prompt = row.get("prompt_tokens") or 0
            completion = row.get("completion_tokens") or 0
            if prompt <= 0 or completion <= 0:
                continue
            samples.append(
                Sample(
                    stage_id=sid,
                    concurrency=conc,
                    prompt_tokens=int(prompt),
                    completion_tokens=int(completion),
                    ttft_ms=float(ttft),
                    tpot_ms=float(tpot),
                )
            )
    return _cap_samples(samples, cfg.max_samples)


def _cap_samples(samples: List[Sample], cap: int) -> List[Sample]:
    """按 stage 分层封顶（系统抽样，确定性）。"""
    if len(samples) <= cap:
        return samples
    by_stage: Dict[int, List[Sample]] = {}
    for s in samples:
        by_stage.setdefault(s.stage_id, []).append(s)
    n_stages = len(by_stage)
    per_stage = max(1, cap // n_stages)
    out: List[Sample] = []
    for sid in sorted(by_stage):
        rows = by_stage[sid]
        if len(rows) <= per_stage:
            out.extend(rows)
            continue
        step = len(rows) / per_stage
        out.extend(rows[int(i * step)] for i in range(per_stage))
    return out


def mix_of(samples: List[Sample], buckets: Sequence[int], bucket_of: Any) -> List[MixPoint]:
    """把样本聚成流量构成：按长度桶取桶内 prompt/completion 均值，权重=计数占比。"""
    groups: Dict[int, List[Sample]] = {}
    for s in samples:
        b = bucket_of(s.prompt_tokens, buckets)
        groups.setdefault(b, []).append(s)
    total = len(samples)
    if total == 0:
        return []
    mix: List[MixPoint] = []
    for b in sorted(groups):
        rows = groups[b]
        mix.append(
            MixPoint(
                prompt_tokens=sum(r.prompt_tokens for r in rows) / len(rows),
                completion_tokens=sum(r.completion_tokens for r in rows) / len(rows),
                weight=len(rows) / total,
                bucket=b,
            )
        )
    return mix


def apply_limits(mix: List[MixPoint], bucket_limits: Dict[int, Tuple[float, float]]) -> None:
    """把各长度桶的生效门限写到 mix 点上（baseline_relative 下门限按桶不同）。"""
    for m in mix:
        if m.bucket in bucket_limits:
            m.ttft_limit_ms, m.tpot_limit_ms = bucket_limits[m.bucket]


@dataclass
class StagePrediction:
    concurrency: int
    n_samples: int
    pred_slo: float                 # 预测 A2 占比（双达标）
    pred_ttft_pass: float
    pred_tpot_pass: float
    pred_ttft_p50_ms: float         # 按桶权重加权的分位数
    pred_ttft_p99_ms: float
    pred_tpot_p50_ms: float
    pred_tpot_p99_ms: float
    extrapolated: bool


def simulate_stage(
    model: QoSModel,
    concurrency: int,
    mix: List[MixPoint],
    ttft_limit_ms: float,
    tpot_limit_ms: float,
) -> StagePrediction:
    """给定并发与流量构成，预测该点的通过率与分位数（桶加权）。

    mix 点可带各自的桶门限（baseline_relative 模式门限按桶不同），
    缺省用调用方传入的统一门限（absolute 模式）。
    """
    if not mix:
        return StagePrediction(concurrency, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
    slo = ttft_pass = tpot_pass = 0.0
    p50_t = p99_t = p50_p = p99_p = 0.0
    extrapolated = False
    for m in mix:
        pred = model.predict(concurrency, m.prompt_tokens, m.completion_tokens)
        rates = model.pass_rates(
            concurrency, m.prompt_tokens, m.completion_tokens,
            m.ttft_limit_ms if m.ttft_limit_ms else ttft_limit_ms,
            m.tpot_limit_ms if m.tpot_limit_ms else tpot_limit_ms,
        )
        w = m.weight
        slo += rates.both * w
        ttft_pass += rates.ttft * w
        tpot_pass += rates.tpot * w
        p50_t += pred.ttft_p50_ms * w
        p99_t += pred.ttft_p99_ms * w
        p50_p += pred.tpot_p50_ms * w
        p99_p += pred.tpot_p99_ms * w
        extrapolated = extrapolated or pred.extrapolated
    return StagePrediction(
        concurrency=concurrency,
        n_samples=int(sum(1 for _ in mix)),
        pred_slo=slo,
        pred_ttft_pass=ttft_pass,
        pred_tpot_pass=tpot_pass,
        pred_ttft_p50_ms=p50_t,
        pred_ttft_p99_ms=p99_t,
        pred_tpot_p50_ms=p50_p,
        pred_tpot_p99_ms=p99_p,
        extrapolated=extrapolated,
    )


@dataclass
class BoundaryScan:
    conc_min: int
    conc_max: int
    grid: List[int]
    predicted_a3_ratio: Dict[int, float]     # 1 - 预测双达标率（质量违约率投影）
    soft_boundary: Optional[int]             # 首个预测 A3 占比 ≥ 软门限的并发
    hard_boundary: Optional[int]             # 首个预测 A3 占比 ≥ 硬门限（质量崩塌）的并发
    quality_capacity: Optional[int]          # 预测 SLO ≥ 目标 的最高连续前缀
    extrapolated_from: Optional[int]         # 包络上限（超过此外推）


def scan_boundaries(
    model: QoSModel,
    mix: List[MixPoint],
    ttft_limit_ms: float,
    tpot_limit_ms: float,
    slo_target: float,
    wm: Any,
    conc_min: int,
    conc_max: int,
    max_points: int = 65,
) -> BoundaryScan:
    """在实测并发范围内细网格扫描预测边界（步进 1，点数超限时等距抽稀）。

    只预测质量侧（SLO/水位软硬门限的 A3 投影）；A1 驱动的硬水位边界
    （拒绝率）不在模型能力内——模型不建模失败概率。
    """
    span = max(1, conc_max - conc_min)
    step = 1 if span <= max_points - 1 else math.ceil(span / (max_points - 1))
    grid = list(range(conc_min, conc_max + 1, step))
    if grid[-1] != conc_max:
        grid.append(conc_max)
    a3_ratio: Dict[int, float] = {}
    for c in grid:
        sp = simulate_stage(model, c, mix, ttft_limit_ms, tpot_limit_ms)
        a3_ratio[c] = max(0.0, 1.0 - sp.pred_slo)

    def _first(threshold: float) -> Optional[int]:
        for c in grid:
            if a3_ratio[c] >= threshold:
                return c
        return None

    # 质量容量：预测 SLO 连续达标前缀的最高并发
    capacity: Optional[int] = None
    for c in grid:
        if 1.0 - a3_ratio[c] >= slo_target:
            capacity = c
        else:
            break

    return BoundaryScan(
        conc_min=conc_min,
        conc_max=conc_max,
        grid=grid,
        predicted_a3_ratio=a3_ratio,
        soft_boundary=_first(wm.soft_slo_violation_ratio),
        hard_boundary=_first(wm.hard_slo_violation_ratio),
        quality_capacity=capacity,
        extrapolated_from=int(model.feat_max[0]),
    )


# ---------------------------------------------------------------------- #
# 训练 / 加载入口（bench.py 调用）
# ---------------------------------------------------------------------- #
@dataclass
class QoSResult:
    """一次运行的预测模型产出：模型本体 + 训练报告 + 报告渲染所需的一切。"""

    model: Optional[QoSModel]
    report: TrainReport
    source: str = "skipped"                 # trained / loaded / skipped
    samples: List[Sample] = field(default_factory=list)
    reference_mix: List[MixPoint] = field(default_factory=list)
    stage_mix: Dict[int, List[MixPoint]] = field(default_factory=dict)
    stage_predictions: Dict[int, StagePrediction] = field(default_factory=dict)
    scan: Optional[BoundaryScan] = None
    model_path: str = ""
    load_warning: str = ""                  # 模型与当前服务不匹配等提示

    @property
    def usable(self) -> bool:
        return (
            self.model is not None
            and self.report.usable
            and not self.load_warning
        )


def train_or_load(
    cfg: Any,
    store: Any,
    force_train: bool = False,
    bucket_of: Any = None,
    log: Any = print,
) -> QoSResult:
    """按配置训练或加载预测模型。

    - enabled=false：跳过（报告不渲染预测章节）；
    - 模型文件存在且未要求重训：加载（--report-only 的默认行为）；
    - 否则：从数据库构建数据集并训练，落盘 save_path。
    """
    qcfg = cfg.qos_model
    result = QoSResult(model=None, report=TrainReport(), model_path="")
    if not qcfg.enabled:
        result.report.skip_reason = "qos_model.enabled=false，未启用预测模型"
        return result

    path = qcfg.save_path
    if not force_train and Path(path).exists():
        try:
            model = QoSModel.load(path)
            result.model = model
            result.source = "loaded"
            result.model_path = path
            result.report = TrainReport(
                n_samples=int(model.meta.get("n_samples", 0)),
                n_stages=int(model.meta.get("n_stages", 0)),
                cross_validated=bool(model.meta.get("cross_validated", False)),
                best_epochs_median=int(model.meta.get("best_epochs_median", 0)),
                r2_ttft=float(model.meta.get("r2_ttft", 0.0)),
                r2_tpot=float(model.meta.get("r2_tpot", 0.0)),
                mae_ttft_ms=float(model.meta.get("mae_ttft_ms", 0.0)),
                mae_tpot_ms=float(model.meta.get("mae_tpot_ms", 0.0)),
                mape_ttft=float(model.meta.get("mape_ttft", 0.0)),
                mape_tpot=float(model.meta.get("mape_tpot", 0.0)),
                seconds=float(model.meta.get("train_seconds", 0.0)),
                folds=[
                    FoldResult(
                        fold=int(f.get("fold", i)),
                        val_stage_ids=[int(c) for c in f.get("val_stage_ids", [])],
                        n_train=int(f.get("n_train", 0)),
                        n_val=int(f.get("n_val", 0)),
                        best_epoch=int(f.get("best_epoch", 0)),
                        val_loss=float(f.get("val_loss", 0.0)),
                    )
                    for i, f in enumerate(model.meta.get("folds", []))
                ],
                usable=(
                    float(model.meta.get("r2_ttft", 0.0)) >= qcfg.min_r2
                    and float(model.meta.get("r2_tpot", 0.0)) >= qcfg.min_r2
                ),
                skip_reason=(
                    ""
                    if (
                        float(model.meta.get("r2_ttft", 0.0)) >= qcfg.min_r2
                        and float(model.meta.get("r2_tpot", 0.0)) >= qcfg.min_r2
                    )
                    else "已加载模型的验证 R² 低于当前 min_r2 配置"
                ),
            )
            # 服务指纹比对：模型绑定当次服务配置，换服务必须重训
            fingerprint = f"{cfg.service.base_url} {cfg.service.model}"
            trained_for = str(model.meta.get("service", ""))
            if trained_for and trained_for != fingerprint:
                result.load_warning = (
                    f"模型训练于其他服务配置（{trained_for}），当前为 {fingerprint}；"
                    "预测可能不适用，建议加 --retrain-qos 重训"
                )
            log(f"[qos] 已加载预测模型: {path}"
                + (f"（R² TTFT={result.report.r2_ttft:.3f} / TPOT={result.report.r2_tpot:.3f}）"
                   if result.report.usable else "（R² 不达标，不出预测结论）"))
            return result
        except Exception as exc:  # noqa: BLE001 - 加载失败回落到重训
            log(f"[qos] 模型加载失败（{exc}），改为重新训练")

    samples = build_dataset(store, qcfg)
    if not samples:
        result.report.skip_reason = "数据库中没有可用的 measured 成功样本，无法训练"
        return result
    log(f"[qos] 训练预测模型：{len(samples)} 个样本 / "
        f"{len({s.stage_id for s in samples})} 个压力级（hidden={qcfg.hidden}，"
        f"上限 {qcfg.max_train_seconds:.0f}s）...")
    model, rep = QoSModel.train(
        samples, qcfg,
        service_fingerprint=f"{cfg.service.base_url} {cfg.service.model}",
    )
    result.model = model
    result.report = rep
    result.source = "trained"
    result.samples = samples
    if rep.skip_reason:
        log(f"[qos] 模型未生成：{rep.skip_reason}")
        return result
    try:
        model.save(path)
        result.model_path = path
        log(f"[qos] 训练完成（{rep.seconds:.1f}s，R² TTFT={rep.r2_ttft:.3f} / "
            f"TPOT={rep.r2_tpot:.3f}），模型已写入 {path}")
    except OSError as exc:
        log(f"[qos] 警告：模型写盘失败（{exc}），本次报告仍可使用内存模型")
    return result
