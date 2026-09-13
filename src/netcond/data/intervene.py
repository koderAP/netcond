"""Interventional data: randomize physical conditions, run a real app model, capture labels.

This is P(traffic | do(c)). The DASH-like client picks the next chunk size from
the last measured throughput (adaptive). HTTP GET uses a-b-t that is approximately
network-invariant (Tmix §5.1).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from netcond.realize.emulator import DumbbellRealizer, label_epochs, transfer_time
from netcond.types import Conditions, Epoch, Trace, preset_conditions

BITRATE_LADDER_BPS = (200_000, 500_000, 1_000_000, 2_500_000, 5_000_000)
CHUNK_DURATION_S = 2.0
HTTP_REQUEST = 400


def _lognormal(rng: random.Random, mu: float, sigma: float) -> float:
    return math.exp(rng.gauss(mu, sigma))


def http_get_epochs(rng: random.Random, n_files: int = 4) -> list[Epoch]:
    """Exogenous sizes/think times — should not move under do(c)."""
    epochs: list[Epoch] = []
    for i in range(n_files):
        a = int(max(80, rng.gauss(HTTP_REQUEST, 40)))
        b = int(_lognormal(rng, math.log(80_000), 0.7))
        b = max(1500, min(b, 2_000_000))
        t = _lognormal(rng, math.log(0.35), 0.6) if i < n_files - 1 else 0.0
        epochs.append(Epoch(a=a, b=b, t=t, epoch_index=i, direction=0))
    return epochs


def dash_like_epochs(c: Conditions, rng: random.Random, n_chunks: int = 8) -> list[Epoch]:
    """Adaptive: next `b` depends on last goodput — feedback is justified."""
    epochs: list[Epoch] = []
    ema = c.capacity_bps * 0.5
    if c.cross_traffic_id not in ("", "none"):
        ema *= 0.55
    extra = 0.35 if c.cross_traffic_id not in ("", "none") else 0.0
    for i in range(n_chunks):
        # pick highest ladder rate under 0.8 * ema
        budget = 0.8 * ema
        rate = BITRATE_LADDER_BPS[0]
        for r in BITRATE_LADDER_BPS:
            if r <= budget:
                rate = r
        a = int(max(200, rng.gauss(350, 30)))
        b = int(rate * CHUNK_DURATION_S / 8.0)
        ta, _, _ = transfer_time(a, c, rng, extra_load=extra)
        tb, rtt, loss = transfer_time(b, c, rng, extra_load=extra)
        goodput = (8.0 * b) / max(tb, 1e-6)
        ema = 0.6 * ema + 0.4 * goodput
        # player: request next chunk shortly after previous completes
        t = 0.04 + abs(rng.gauss(0.02, 0.01)) if i < n_chunks - 1 else 0.0
        epochs.append(
            Epoch(
                a=a,
                b=b,
                t=t,
                epoch_index=i,
                direction=0,
                transfer_time_a=ta,
                transfer_time_b=tb,
                rtt=rtt,
                loss=loss,
            )
        )
    return epochs


def collect_run(
    *,
    preset: str,
    app_kind: str,
    session_id: int,
    seed: int,
    write_pcap: bool = False,
    pcap_dir: Path | None = None,
) -> Trace:
    rng = random.Random(seed)
    c = preset_conditions(preset)
    # small continuous jitter so c is randomized, not a single point
    c = Conditions(
        capacity_bps=c.capacity_bps * rng.uniform(0.85, 1.15),
        base_rtt_s=c.base_rtt_s * rng.uniform(0.9, 1.1),
        buffer_bytes=c.buffer_bytes * rng.uniform(0.8, 1.2),
        aqm=c.aqm,
        loss_rate=min(0.2, c.loss_rate * rng.uniform(0.5, 1.5) if c.loss_rate else 0.0),
        cross_traffic_id=c.cross_traffic_id,
        preset=preset,
    )
    if app_kind == "dash_like":
        epochs = dash_like_epochs(c, rng)
    elif app_kind == "http_get":
        epochs = http_get_epochs(rng)
        epochs = label_epochs(epochs, c, rng)
    else:
        raise ValueError(app_kind)
    for e in epochs:
        e.session_id = session_id
        e.connection_id = session_id
    trace = Trace(epochs=epochs, conditions=c, app_kind=app_kind, session_id=session_id, metadata={"seed": seed, "do_c": True})
    if write_pcap:
        realizer = DumbbellRealizer()
        trace = realizer.realize_trace(trace, seed=seed)
        if pcap_dir is not None:
            pcap_dir.mkdir(parents=True, exist_ok=True)
            realizer.write_pcap(trace.packets, str(pcap_dir / f"sess{session_id}_{preset}_{app_kind}.pcap"))
    return trace


def collect_dataset(
    out_json: str | Path,
    *,
    n_per_preset: int = 8,
    presets: tuple[str, ...] = ("lan", "wan", "congested"),
    apps: tuple[str, ...] = ("http_get", "dash_like"),
    seed: int = 0,
    write_pcap: bool = False,
    pcap_dir: str | Path | None = None,
) -> list[Trace]:
    traces: list[Trace] = []
    sid = 0
    pdir = Path(pcap_dir) if pcap_dir else None
    for preset in presets:
        for app in apps:
            for k in range(n_per_preset):
                traces.append(
                    collect_run(
                        preset=preset,
                        app_kind=app,
                        session_id=sid,
                        seed=seed + sid * 17,
                        write_pcap=write_pcap and k == 0,
                        pcap_dir=pdir,
                    )
                )
                sid += 1
    payload = {"traces": [t.without_packets().to_dict() for t in traces]}
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    return traces


def load_traces(path: str | Path) -> list[Trace]:
    raw = json.loads(Path(path).read_text())
    return [Trace.from_dict(t) for t in raw["traces"]]
