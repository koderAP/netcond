"""Tmix-style empirical resampling of a-b-t from the train set (no neural net)."""

from __future__ import annotations

import random

from netcond.realize.emulator import label_epochs
from netcond.types import Conditions, Epoch, Trace


def resample_traces(
    train: list[Trace],
    conditions: Conditions,
    *,
    n_sessions: int = 8,
    n_steps: int | None = None,
    seed: int = 0,
    app_kind: str | None = None,
) -> list[Trace]:
    pool = [tr for tr in train if app_kind is None or tr.app_kind == app_kind]
    if not pool:
        pool = train
    rng = random.Random(seed)
    all_epochs = [e for tr in pool for e in tr.epochs]
    out: list[Trace] = []
    for s in range(n_sessions):
        src = rng.choice(pool)
        k = n_steps or len(src.epochs)
        picked = [rng.choice(all_epochs) for _ in range(k)]
        epochs = [
            Epoch(
                a=e.a,
                b=e.b,
                t=e.t if i < k - 1 else 0.0,
                epoch_index=i,
                direction=e.direction,
            )
            for i, e in enumerate(picked)
        ]
        label_epochs(epochs, conditions, rng)
        out.append(
            Trace(
                epochs=epochs,
                conditions=conditions,
                app_kind=src.app_kind,
                session_id=s,
                metadata={"baseline": "tmix_resample"},
            )
        )
    return out
