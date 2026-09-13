"""Pad traces into tensors for training."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random

import torch
from torch import Tensor

from netcond.types import Trace

APP_KIND_TO_ID = {"http_get": 0, "dash_like": 1}


@dataclass
class Batch:
    a: Tensor
    b: Tensor
    t: Tensor
    direction: Tensor
    transfer_a: Tensor
    transfer_b: Tensor
    rtt: Tensor
    loss: Tensor
    conditions: Tensor
    context: Tensor
    mask: Tensor  # (B,T) 1 = valid epoch
    adaptive: Tensor  # (B,) 1 if dash_like


def traces_to_batch(traces: list[Trace], device: str | None = None) -> Batch:
    T = max(len(tr.epochs) for tr in traces)
    B = len(traces)
    z = torch.zeros
    a = z(B, T)
    b = z(B, T)
    t = z(B, T)
    d = z(B, T)
    ta = z(B, T)
    tb = z(B, T)
    rtt = z(B, T)
    loss = z(B, T)
    mask = z(B, T)
    cond = z(B, 5)
    ctx = z(B, 4)
    adaptive = z(B)
    for i, tr in enumerate(traces):
        cond[i] = tr.conditions.as_tensor()
        kind = APP_KIND_TO_ID.get(tr.app_kind, 0)
        adaptive[i] = 1.0 if tr.app_kind == "dash_like" else 0.0
        ctx[i] = torch.tensor([float(kind), 0.0, 0.0, 0.0])
        for j, e in enumerate(tr.epochs):
            a[i, j] = max(e.a, 1)
            b[i, j] = max(e.b, 1)
            t[i, j] = max(e.t, 0.0)
            d[i, j] = e.direction
            ta[i, j] = e.transfer_time_a or 0.0
            tb[i, j] = e.transfer_time_b or 0.0
            rtt[i, j] = e.rtt or 0.0
            loss[i, j] = e.loss or 0.0
            mask[i, j] = 1.0
    batch = Batch(a, b, t, d, ta, tb, rtt, loss, cond, ctx, mask, adaptive)
    if device:
        for f in batch.__dataclass_fields__:
            setattr(batch, f, getattr(batch, f).to(device))
    return batch


def split_traces(traces: list[Trace], frac: float = 0.75, seed: int = 0) -> tuple[list[Trace], list[Trace]]:
    """Stratify by (app_kind, preset) so holdout covers every do(c) cell."""
    buckets: dict[tuple[str, str], list[Trace]] = defaultdict(list)
    for tr in traces:
        buckets[(tr.app_kind, tr.conditions.preset or "")].append(tr)
    rng = random.Random(seed)
    train: list[Trace] = []
    hold: list[Trace] = []
    for _k, group in buckets.items():
        g = list(group)
        rng.shuffle(g)
        if len(g) == 1:
            train.extend(g)
            continue
        n_train = max(1, min(len(g) - 1, int(round(frac * len(g)))))
        train.extend(g[:n_train])
        hold.extend(g[n_train:])
    return train, hold
