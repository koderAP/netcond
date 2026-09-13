"""Minimal demo: collect a few do(c) runs, train briefly, print lan vs congested means."""

from netcond.data.intervene import collect_run
from netcond.generate import generate
from netcond.train import train_loop


def main() -> None:
    traces = [
        collect_run(preset=p, app_kind="dash_like", session_id=i, seed=i)
        for i, p in enumerate(["lan", "wan", "congested"] * 4)
    ]
    loop = train_loop(traces, epochs=9, seed=0)["loop"]
    for name in ("lan", "congested"):
        gens = generate(loop, name, app_kind="dash_like", n_sessions=4, n_steps=6, seed=0)
        rtts = [t.mean_rtt() or 0.0 for t in gens]
        tbs = [sum(e.transfer_time_b or 0.0 for e in t.epochs) / len(t.epochs) for t in gens]
        print(f"{name:12s} mean RTT {sum(rtts)/len(rtts):.4f}s  mean transfer_b {sum(tbs)/len(tbs):.4f}s")


if __name__ == "__main__":
    main()
