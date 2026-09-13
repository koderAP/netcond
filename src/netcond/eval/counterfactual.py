"""Counterfactual check against interventional holdout (do(c), not passive).

Unique metric: generate under c' unseen for that holdout flow; compare
transfer-time / RTT / goodput to emulator capture under c'. 2× base RTT
must move mean RTT in the predicted direction.
"""

from __future__ import annotations

from dataclasses import replace

from netcond.couple.loop import CoupledLoop
from netcond.eval.baseline_resample import resample_traces
from netcond.eval.fidelity import field_metrics
from netcond.generate import generate
from netcond.realize.emulator import label_epochs
from netcond.types import Conditions, Trace, preset_conditions


def mean_metric(traces: list[Trace], name: str) -> float:
    xs: list[float] = []
    for tr in traces:
        if name == "rtt":
            v = tr.mean_rtt()
            if v is not None:
                xs.append(v)
        elif name == "transfer_b":
            xs.extend(e.transfer_time_b for e in tr.epochs if e.transfer_time_b)
        elif name == "goodput":
            g = tr.goodput_bps()
            if g is not None:
                xs.append(g)
    return sum(xs) / len(xs) if xs else float("nan")


def relabel_under(traces: list[Trace], c: Conditions) -> list[Trace]:
    """Oracle emulator relabel of *exogenous* epochs under a new condition (ground truth)."""
    out: list[Trace] = []
    for tr in traces:
        epochs = [
            replace(
                e,
                transfer_time_a=None,
                transfer_time_b=None,
                rtt=None,
                loss=None,
            )
            for e in tr.epochs
        ]
        label_epochs(epochs, c)
        out.append(replace(tr, epochs=epochs, conditions=c))
    return out


def rtt_doubling_check(loop: CoupledLoop, app_kind: str = "dash_like", n_steps: int = 6) -> dict:
    base = preset_conditions("wan")
    doubled = replace(base, base_rtt_s=base.base_rtt_s * 2.0, preset="wan_2x_rtt")
    a = generate(loop, "wan", app_kind=app_kind, n_steps=n_steps, n_sessions=6, seed=1)
    # force doubled conditions through rollout
    import torch

    kind = 1.0 if app_kind == "dash_like" else 0.0
    ctx = torch.tensor([kind, doubled.loss_rate, 0.0, 0.0])
    b = [loop.rollout(doubled, ctx, n_steps, adaptive_feedback=app_kind == "dash_like") for _ in range(6)]
    m0 = mean_metric(a, "rtt")
    m1 = mean_metric(b, "rtt")
    return {
        "rtt_base": m0,
        "rtt_2x": m1,
        "direction_ok": bool(m1 > m0),
        "ratio": (m1 / m0) if m0 else float("nan"),
        "expected": "mean RTT increases when base RTT doubles",
    }


def counterfactual_table(
    loop: CoupledLoop,
    holdout: list[Trace],
    train: list[Trace],
    from_preset: str,
    to_preset: str,
    *,
    app_kind: str = "dash_like",
    n_sessions: int = 8,
) -> dict:
    """Holdout traces collected under from_preset; intervene to to_preset."""
    c_to = preset_conditions(to_preset)
    real_to = [tr for tr in holdout if tr.conditions.preset == to_preset and tr.app_kind == app_kind]
    if not real_to:
        # interventional GT: relabel holdout exogenous vectors if same app is invariant;
        # for dash, use collected traces under to_preset only.
        real_to = [tr for tr in holdout if tr.app_kind == app_kind and tr.conditions.preset == to_preset]
    gen = generate(loop, to_preset, app_kind=app_kind, n_sessions=n_sessions, n_steps=8, seed=2)
    base = resample_traces(train, c_to, n_sessions=n_sessions, n_steps=8, seed=2, app_kind=app_kind)
    return {
        "from": from_preset,
        "to": to_preset,
        "app_kind": app_kind,
        "neural_vs_real": {
            "transfer_b": field_metrics(real_to, gen, "transfer_b", log=True) if real_to else {},
            "rtt": field_metrics(real_to, gen, "rtt") if real_to else {},
            "mean_transfer_neural": mean_metric(gen, "transfer_b"),
            "mean_transfer_real": mean_metric(real_to, "transfer_b") if real_to else None,
            "mean_rtt_neural": mean_metric(gen, "rtt"),
            "mean_rtt_real": mean_metric(real_to, "rtt") if real_to else None,
        },
        "resample_vs_real": {
            "transfer_b": field_metrics(real_to, base, "transfer_b", log=True) if real_to else {},
            "rtt": field_metrics(real_to, base, "rtt") if real_to else {},
            "mean_transfer_resample": mean_metric(base, "transfer_b"),
            "mean_rtt_resample": mean_metric(base, "rtt"),
        },
        "causal_caveat": "Holdout is interventional P(traffic | do(c)). Do not read this as CausalSim adjustment of passive traces.",
    }
