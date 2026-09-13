"""Fidelity: JSD + Wasserstein/EMD on a, b, t (think times only). Size CCDF."""

from __future__ import annotations

import math

import numpy as np

from netcond.types import Trace


def _hist(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(x, bins=bins, density=False)
    h = h.astype(np.float64)
    s = h.sum()
    if s <= 0:
        return np.ones_like(h) / len(h)
    return h / s


def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    n = max(len(x), len(y), 1)
    qx = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x) if len(x) else np.zeros(n)
    qy = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y)), y) if len(y) else np.zeros(n)
    return float(np.mean(np.abs(qx - qy)))


def _collect(traces: list[Trace], field: str) -> np.ndarray:
    xs: list[float] = []
    for tr in traces:
        if field == "a":
            xs.extend(float(e.a) for e in tr.epochs)
        elif field == "b":
            xs.extend(float(e.b) for e in tr.epochs)
        elif field == "t":
            xs.extend(tr.think_times())
        elif field == "transfer_b":
            xs.extend(float(e.transfer_time_b) for e in tr.epochs if e.transfer_time_b)
        elif field == "rtt":
            xs.extend(float(e.rtt) for e in tr.epochs if e.rtt)
        else:
            raise KeyError(field)
    return np.asarray(xs, dtype=np.float64)


def field_metrics(real: list[Trace], syn: list[Trace], field: str, log: bool = False) -> dict[str, float]:
    x = _collect(real, field)
    y = _collect(syn, field)
    if log:
        x = np.log(np.clip(x, 1e-8, None))
        y = np.log(np.clip(y, 1e-8, None))
    if len(x) == 0 or len(y) == 0:
        return {"jsd": math.nan, "emd": math.nan, "n_real": len(x), "n_syn": len(y)}
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if hi <= lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 24)
    return {
        "jsd": jsd(_hist(x, bins), _hist(y, bins)),
        "emd": wasserstein_1d(x, y),
        "n_real": int(len(x)),
        "n_syn": int(len(y)),
        "mean_real": float(x.mean()),
        "mean_syn": float(y.mean()),
        "qq_mae": _qq_mae(x, y),
    }


def _qq_mae(x: np.ndarray, y: np.ndarray, n: int = 50) -> float:
    qs = np.linspace(0.02, 0.98, n)
    return float(np.mean(np.abs(np.quantile(x, qs) - np.quantile(y, qs))))


def size_ccdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(values)
    n = len(v)
    if n == 0:
        return v, v
    ccdf = 1.0 - np.arange(1, n + 1) / n
    return v, ccdf


def fidelity_report(real: list[Trace], syn: list[Trace]) -> dict:
    return {
        "a": field_metrics(real, syn, "a", log=True),
        "b": field_metrics(real, syn, "b", log=True),
        "t_think": field_metrics(real, syn, "t", log=True),
        "transfer_b": field_metrics(real, syn, "transfer_b", log=True),
        "rtt": field_metrics(real, syn, "rtt", log=False),
        "note": "IAT is think-time t only; mixed bidirectional IAT is not a metric.",
    }
