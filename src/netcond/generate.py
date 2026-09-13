"""Generate traces under named network conditions."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from netcond.couple.loop import CoupledLoop
from netcond.realize.emulator import DumbbellRealizer
from netcond.train import load_checkpoint
from netcond.types import Trace, preset_conditions


def generate(
    loop: CoupledLoop,
    preset: str,
    *,
    app_kind: str = "dash_like",
    n_steps: int = 8,
    n_sessions: int = 8,
    adaptive_feedback: bool | None = None,
    seed: int = 0,
    realize: bool = False,
    unconditioned: bool = False,
) -> list[Trace]:
    if adaptive_feedback is None:
        adaptive_feedback = app_kind == "dash_like"
    cond = preset_conditions(preset)
    kind = 1.0 if app_kind == "dash_like" else 0.0
    ctx = torch.tensor([kind, 0.0, 0.0, 0.0])
    traces: list[Trace] = []
    g = torch.Generator()
    g.manual_seed(seed)
    for i in range(n_sessions):
        tr = loop.rollout(
            cond,
            ctx,
            n_steps,
            adaptive_feedback=adaptive_feedback and not unconditioned,
            unconditioned=unconditioned,
            generator=g,
        )
        tr.app_kind = app_kind
        tr.session_id = i
        tr.metadata["preset"] = preset
        tr.metadata["unconditioned"] = unconditioned
        if realize:
            DumbbellRealizer().realize_trace(tr, seed=seed + i)
        traces.append(tr)
    return traces


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--preset", default="wan")
    p.add_argument("--app", default="dash_like")
    p.add_argument("--out", default="output/gen.json")
    p.add_argument("--n-sessions", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=8)
    args = p.parse_args(argv)
    loop = load_checkpoint(args.ckpt)
    traces = generate(loop, args.preset, app_kind=args.app, n_sessions=args.n_sessions, n_steps=args.n_steps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"traces": [t.without_packets().to_dict() for t in traces]}, indent=2))
    print(f"wrote {len(traces)} traces to {args.out}")


if __name__ == "__main__":
    main()
