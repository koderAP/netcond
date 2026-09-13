"""Tmix-style empirical resampling of a-b-t (no neural net).

Two modes:
- session vectors from the *same* named condition (in-distribution Tmix)
- replay of vectors collected under c, labeled under c' (Tmix cannot restar
  adaptive chunk sizes — that is the SoTA we beat)
"""

from __future__ import annotations

import random
from netcond.realize.emulator import label_epochs
from netcond.types import Conditions, Epoch, Trace


def _pool(train: list[Trace], app_kind: str | None, preset: str | None) -> list[Trace]:
    pool = [tr for tr in train if app_kind is None or tr.app_kind == app_kind]
    if preset:
        matched = [tr for tr in pool if tr.conditions.preset == preset]
        if matched:
            pool = matched
    return pool or train


def resample_traces(
    train: list[Trace],
    conditions: Conditions,
    *,
    n_sessions: int = 8,
    n_steps: int | None = None,
    seed: int = 0,
    app_kind: str | None = None,
    match_preset: bool = True,
    session_level: bool = True,
    source_preset: str | None = None,
) -> list[Trace]:
    preset = source_preset if source_preset is not None else (conditions.preset if match_preset else None)
    pool = _pool(train, app_kind, preset)
    rng = random.Random(seed)
    iid_epochs = [e for tr in pool for e in tr.epochs]
    out: list[Trace] = []
    for s in range(n_sessions):
        src = rng.choice(pool)
        k = n_steps or len(src.epochs)
        if session_level and src.epochs:
            picked = list(src.epochs)
            while len(picked) < k:
                picked.extend(src.epochs)
            picked = picked[:k]
        else:
            picked = [rng.choice(iid_epochs) for _ in range(k)]
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
        extra = 0.35 if (src.app_kind == "dash_like") else 0.0
        label_epochs(epochs, conditions, rng, extra_load=extra)
        out.append(
            Trace(
                epochs=epochs,
                conditions=conditions,
                app_kind=src.app_kind,
                session_id=s,
                metadata={
                    "baseline": "tmix_resample",
                    "source_preset": src.conditions.preset,
                    "session_level": session_level,
                },
            )
        )
    return out


def tmix_replay_counterfactual(
    train: list[Trace],
    *,
    from_preset: str,
    to_conditions: Conditions,
    app_kind: str,
    n_sessions: int = 8,
    n_steps: int = 8,
    seed: int = 0,
) -> list[Trace]:
    """Replay a-b-t collected under from_preset through to_conditions (Tmix/Swing)."""
    return resample_traces(
        train,
        to_conditions,
        n_sessions=n_sessions,
        n_steps=n_steps,
        seed=seed,
        app_kind=app_kind,
        match_preset=True,
        session_level=True,
        source_preset=from_preset,
    )
