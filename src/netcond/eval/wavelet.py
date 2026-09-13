"""Haar logscale energy of byte arrivals. Short traces skip with a documented reason."""

from __future__ import annotations

import math

import numpy as np

from netcond.types import Trace


def byte_arrival_series(trace: Trace, dt: float = 0.01, max_seconds: float = 32.0) -> np.ndarray | None:
    if not trace.packets:
        return None
    t0 = trace.packets[0].timestamp
    t1 = trace.packets[-1].timestamp
    duration = t1 - t0
    if duration < 1.0:
        return None
    n = int(min(duration, max_seconds) / dt)
    if n < 64:
        return None
    x = np.zeros(n)
    for p in trace.packets:
        idx = int((p.timestamp - t0) / dt)
        if 0 <= idx < n:
            x[idx] += p.payload_len
    return x


def haar_energy(x: np.ndarray, n_scales: int = 8) -> dict[str, list[float]]:
    """Simple Haar wavelet energy per dyadic scale (Abry–Veitch logscale proxy)."""
    energies = []
    scales = []
    cur = x.astype(np.float64)
    for j in range(n_scales):
        if len(cur) < 4:
            break
        if len(cur) % 2:
            cur = cur[:-1]
        a = cur[0::2]
        d = cur[1::2]
        detail = (a - d) / math.sqrt(2.0)
        energies.append(float(np.mean(detail**2) + 1e-18))
        scales.append(2 ** (j + 1))
        cur = (a + d) / math.sqrt(2.0)
    log_e = [math.log(e) for e in energies]
    return {"scale": scales, "energy": energies, "log_energy": log_e}


def wavelet_or_skip(traces: list[Trace]) -> dict:
    series = [byte_arrival_series(tr) for tr in traces]
    ok = [s for s in series if s is not None]
    if len(ok) < 2:
        return {
            "skipped": True,
            "reason": (
                "Wavelet logscale energy needs packet arrivals spanning tens of seconds. "
                "This slice's sessions are shorter; report Q–Q of think time t and transfer times instead."
            ),
        }
    stacked = haar_energy(np.concatenate(ok[:3]))
    stacked["skipped"] = False
    return stacked
