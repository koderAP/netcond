"""Train-synthetic-test-real utility: predict condition id from session features."""

from __future__ import annotations

import math

import numpy as np

from netcond.types import PRESET_NAMES, Trace

PRESET_INDEX = {n: i for i, n in enumerate(PRESET_NAMES)}


def session_features(tr: Trace) -> np.ndarray:
    a = np.array([e.a for e in tr.epochs], dtype=np.float64)
    b = np.array([e.b for e in tr.epochs], dtype=np.float64)
    t = np.array(tr.think_times() or [0.0], dtype=np.float64)
    rtt = tr.mean_rtt() or 0.0
    gp = tr.goodput_bps() or 0.0
    return np.array(
        [
            math.log(max(a.mean(), 1.0)),
            math.log(max(b.mean(), 1.0)),
            math.log(max(t.mean(), 1e-6)),
            rtt,
            math.log(max(gp, 1.0)),
        ],
        dtype=np.float64,
    )


def _xy(traces: list[Trace]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for tr in traces:
        name = tr.conditions.preset or tr.metadata.get("preset") or ""
        if name not in PRESET_INDEX:
            continue
        xs.append(session_features(tr))
        ys.append(PRESET_INDEX[name])
    if not xs:
        return np.zeros((0, 5)), np.zeros((0,), dtype=int)
    return np.stack(xs), np.asarray(ys, dtype=int)


def _softmax_fit(X: np.ndarray, y: np.ndarray, n_classes: int, steps: int = 400) -> np.ndarray:
    """Multinomial logistic regression via gradient descent (no sklearn)."""
    n, d = X.shape
    W = np.zeros((d, n_classes))
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Y = np.eye(n_classes)[y]
    lr = 0.3
    for _ in range(steps):
        logits = Xn @ W
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        p = exp / exp.sum(axis=1, keepdims=True)
        grad = Xn.T @ (p - Y) / n
        W -= lr * grad
    return W, X.mean(0), X.std(0) + 1e-6


def _predict(X, W, mean, std) -> np.ndarray:
    Xn = (X - mean) / std
    logits = Xn @ W
    return logits.argmax(axis=1)


def accuracy(yhat: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float((yhat == y).mean())


def tstr_condition_id(real_train: list[Trace], syn: list[Trace], real_test: list[Trace]) -> dict:
    """Train on synthetic features, test on real (TSTR). Also TRTR reference."""
    Xs, ys = _xy(syn)
    Xtr, ytr = _xy(real_train)
    Xte, yte = _xy(real_test)
    n_cl = len(PRESET_NAMES)
    out = {"n_syn": int(len(ys)), "n_real_test": int(len(yte))}
    if len(ys) < n_cl or len(yte) == 0:
        out["tstr_acc"] = float("nan")
        out["trtr_acc"] = float("nan")
        out["note"] = "not enough labeled presets"
        return out
    W_s, m_s, s_s = _softmax_fit(Xs, ys, n_cl)
    out["tstr_acc"] = accuracy(_predict(Xte, W_s, m_s, s_s), yte)
    if len(ytr) >= n_cl:
        W_r, m_r, s_r = _softmax_fit(Xtr, ytr, n_cl)
        out["trtr_acc"] = accuracy(_predict(Xte, W_r, m_r, s_r), yte)
    else:
        out["trtr_acc"] = float("nan")
    out["note"] = "TSTR = train classifier on synthetic, test on real interventional holdout."
    return out
