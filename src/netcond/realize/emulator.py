"""Userspace dumbbell TCP realizer.

Not the old MSS stub: packet timestamps come from capacity, base RTT, buffer,
loss, and a windowed sender. Linux `tc netem` is used when available; this
model is the portable interventional engine (macOS has no netem).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from netcond.realize.pcap_io import write_pcap as write_pcap_file
from netcond.types import Conditions, Epoch, PacketRecord, Trace

MSS = 1460
INIT_CWND_MSS = 10
RTO_MIN = 0.2


@dataclass
class _FlowClock:
    t: float = 0.0
    seq_c: int = 1
    seq_s: int = 1
    ack_c: int = 1
    ack_s: int = 1


def _serialization_s(nbytes: int, capacity_bps: float) -> float:
    return (nbytes * 8.0) / max(capacity_bps, 1.0)


def _queue_delay(queue_bytes: float, capacity_bps: float) -> float:
    return _serialization_s(int(queue_bytes), capacity_bps)


def transfer_time(
    nbytes: int,
    c: Conditions,
    rng: random.Random,
    *,
    extra_load: float = 0.0,
) -> tuple[float, float, float]:
    """Return (transfer_time, observed_rtt, loss_frac) for one ADU."""
    if nbytes <= 0:
        return 0.0, c.base_rtt_s, 0.0
    cap = c.capacity_bps * max(0.15, 1.0 - extra_load)
    if c.cross_traffic_id not in ("", "none"):
        cap *= 0.55
    rtt = c.base_rtt_s
    mss = MSS
    n_seg = max(1, math.ceil(nbytes / mss))
    cwnd = float(INIT_CWND_MSS)
    ssthresh = 1e9
    sent = 0
    t = 0.0
    losses = 0
    queue = 0.0
    rtts: list[float] = []
    steps = 0
    while sent < n_seg:
        steps += 1
        if steps > n_seg * 40:
            # bounded: remaining bytes at drain rate
            t += _serialization_s((n_seg - sent) * mss, cap) + rtt
            sent = n_seg
            break
        window = max(1, int(cwnd))
        burst = min(window, n_seg - sent)
        dropped = 0
        for _ in range(burst):
            pkt_bytes = mss
            if queue + pkt_bytes > c.buffer_bytes:
                dropped += 1
                losses += 1
                continue
            if c.loss_rate > 0 and rng.random() < c.loss_rate:
                dropped += 1
                losses += 1
                continue
            queue += pkt_bytes
        delivered = burst - dropped
        if delivered <= 0:
            queue = 0.0
            cwnd = 1.0
            t += max(RTO_MIN, 2.0 * rtt)
            continue
        q_d = _queue_delay(queue, cap)
        ser = _serialization_s(delivered * mss, cap)
        rtt_obs = rtt + 2.0 * q_d
        rtts.append(rtt_obs)
        t += ser + rtt_obs
        queue = max(0.0, queue - delivered * mss * 0.85)
        sent += delivered
        if dropped:
            ssthresh = max(2.0, cwnd / 2.0)
            cwnd = 1.0
            t += max(RTO_MIN, 2.0 * rtt_obs)
        else:
            if cwnd < ssthresh:
                cwnd += delivered
            else:
                cwnd += delivered / max(cwnd, 1.0)
    loss_frac = losses / max(n_seg, 1)
    mean_rtt = sum(rtts) / max(len(rtts), 1)
    return t, mean_rtt, loss_frac


def label_epochs(
    epochs: list[Epoch],
    c: Conditions,
    rng: random.Random | None = None,
    extra_load: float = 0.0,
) -> list[Epoch]:
    rng = rng or random.Random(0)
    out: list[Epoch] = []
    for e in epochs:
        ta, rtt_a, la = transfer_time(e.a, c, rng, extra_load=extra_load)
        tb, rtt_b, lb = transfer_time(e.b, c, rng, extra_load=extra_load)
        e.transfer_time_a = ta
        e.transfer_time_b = tb
        e.rtt = 0.5 * (rtt_a + rtt_b)
        e.loss = 0.5 * (la + lb)
        out.append(e)
    return out


class DumbbellRealizer:
    """Wire-valid TCP handshake + data + FIN with physically timed packets."""

    def __init__(self, client_ip: str = "10.0.0.1", server_ip: str = "10.0.0.2") -> None:
        self.client_ip = client_ip
        self.server_ip = server_ip

    def realize(self, epochs: list[Epoch], conditions: Conditions, *, seed: int = 0) -> list[PacketRecord]:
        rng = random.Random(seed)
        sport = 10_000 + (seed % 50_000)
        dport = 80
        clk = _FlowClock()
        pkts: list[PacketRecord] = []

        def add(
            ts: float,
            direction: int,
            seq: int,
            ack: int,
            flags: str,
            plen: int,
        ) -> None:
            if direction == 0:
                src, dst, sp, dp = self.client_ip, self.server_ip, sport, dport
            else:
                src, dst, sp, dp = self.server_ip, self.client_ip, dport, sport
            pkts.append(
                PacketRecord(
                    timestamp=ts,
                    src=src,
                    dst=dst,
                    src_port=sp,
                    dst_port=dp,
                    seq=seq,
                    ack=ack,
                    flags=flags,
                    payload_len=plen,
                    direction=direction,
                )
            )

        # handshake
        add(clk.t, 0, 0, 0, "S", 0)
        clk.t += conditions.base_rtt_s / 2
        add(clk.t, 1, 0, 1, "SA", 0)
        clk.t += conditions.base_rtt_s / 2
        add(clk.t, 0, 1, 1, "A", 0)

        for e in epochs:
            ta, rtt, _ = transfer_time(e.a, conditions, rng)
            tb, rtt2, _ = transfer_time(e.b, conditions, rng)
            e.transfer_time_a = ta
            e.transfer_time_b = tb
            e.rtt = 0.5 * (rtt + rtt2)
            t_req = self._send_adu(add, clk, direction=0, nbytes=e.a, duration=max(ta, 1e-4), rtt=rtt)
            clk.t = t_req + max(ta, conditions.base_rtt_s / 2)
            t_resp = self._send_adu(add, clk, direction=1, nbytes=e.b, duration=max(tb, 1e-4), rtt=rtt2)
            clk.t = t_resp + max(0.0, e.t)

        add(clk.t, 0, clk.seq_c, clk.seq_s, "FA", 0)
        clk.t += conditions.base_rtt_s / 2
        add(clk.t, 1, clk.seq_s, clk.seq_c + 1, "FA", 0)
        clk.t += conditions.base_rtt_s / 2
        add(clk.t, 0, clk.seq_c + 1, clk.seq_s + 1, "A", 0)
        return pkts

    def _send_adu(self, add, clk: _FlowClock, *, direction: int, nbytes: int, duration: float, rtt: float) -> float:
        if nbytes <= 0:
            return clk.t
        n_seg = max(1, math.ceil(nbytes / MSS))
        dt = duration / max(n_seg, 1)
        remaining = nbytes
        t = clk.t
        last_payload = t
        for i in range(n_seg):
            plen = min(MSS, remaining)
            remaining -= plen
            if direction == 0:
                add(t, 0, clk.seq_c, clk.seq_s, "PA", plen)
                clk.seq_c += plen
                last_payload = t
                add(t, 1, clk.seq_s, clk.seq_c, "A", 0)
            else:
                add(t, 1, clk.seq_s, clk.seq_c, "PA", plen)
                clk.seq_s += plen
                last_payload = t
                add(t, 0, clk.seq_c, clk.seq_s, "A", 0)
            if i < n_seg - 1:
                t += dt
        # Clock sits on last payload so think time is Tmix-correct (resp-complete → next request).
        clk.t = last_payload
        return last_payload

    def write_pcap(self, packets: list[PacketRecord], path: str) -> None:
        write_pcap_file(path, packets)

    def realize_trace(self, trace: Trace, *, seed: int = 0) -> Trace:
        pkts = self.realize(trace.epochs, trace.conditions, seed=seed)
        trace.packets = pkts
        return trace
