"""Tmix-style connection-vector extractor (seq/ack ADUs, not IAT bursts).

Think time t is time from last response byte received to next request byte sent.
Transfer time is *not* an application target.

Completeness: SYN or SYN+ACK present, terminated by FIN or RST. Incomplete
connections are reported, not silently dropped.

Concurrent connections (both sides have unacked data) are split into two
unidirectional epoch streams (Tmix §3.1).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from netcond.types import (
    Conditions,
    Epoch,
    ExtractStats,
    PacketRecord,
    Trace,
    preset_conditions,
)

IDLE_THRESHOLD_S = 4.0


@dataclass
class _ConnKey:
    a: tuple[str, int]
    b: tuple[str, int]

    @property
    def canonical(self) -> tuple[tuple[str, int], tuple[str, int]]:
        return (self.a, self.b) if self.a <= self.b else (self.b, self.a)


def _key(pkt: PacketRecord) -> tuple[tuple[str, int], tuple[str, int]]:
    a = (pkt.src, pkt.src_port)
    b = (pkt.dst, pkt.dst_port)
    return (a, b) if a <= b else (b, a)


def _orient(packets: list[PacketRecord]) -> tuple[list[PacketRecord], tuple[str, int]]:
    """Set direction 0 = SYN initiator → acceptor. Fallback: first packet src."""
    initiator = None
    for p in packets:
        if "S" in p.flags and "A" not in p.flags:
            initiator = (p.src, p.src_port)
            break
        if "S" in p.flags and "A" in p.flags:
            initiator = (p.dst, p.dst_port)
            break
    if initiator is None:
        initiator = (packets[0].src, packets[0].src_port)
    oriented: list[PacketRecord] = []
    for p in packets:
        d = 0 if (p.src, p.src_port) == initiator else 1
        oriented.append(
            PacketRecord(
                timestamp=p.timestamp,
                src=p.src,
                dst=p.dst,
                src_port=p.src_port,
                dst_port=p.dst_port,
                seq=p.seq,
                ack=p.ack,
                flags=p.flags,
                payload_len=p.payload_len,
                direction=d,
                window=p.window,
            )
        )
    return oriented, initiator


def is_complete(packets: list[PacketRecord]) -> bool:
    has_syn = any("S" in p.flags for p in packets)
    has_end = any(("F" in p.flags) or ("R" in p.flags) for p in packets)
    return has_syn and has_end


def is_concurrent(packets: list[PacketRecord]) -> bool:
    """Tmix: exists p,q with p.seq > q.ack and q.seq > p.ack (unacked data both ways)."""
    data = [p for p in packets if p.payload_len > 0]
    for i, p in enumerate(data):
        for q in data[i + 1 :]:
            if p.direction == q.direction:
                continue
            if p.seq > q.ack and q.seq > p.ack:
                return True
    return False


def min_rtt(packets: list[PacketRecord]) -> float | None:
    """Aikat et al. IMC 2003 style: min (ACK time − matching data send). Network param, not app."""
    samples: list[float] = []
    outstanding: dict[tuple[int, int], float] = {}
    for p in packets:
        if p.payload_len > 0:
            end_seq = (p.seq + p.payload_len) & 0xFFFFFFFF
            outstanding[(p.direction, end_seq)] = p.timestamp
        if "A" in p.flags and p.ack > 0:
            key = (1 - p.direction, p.ack)
            if key in outstanding:
                rtt = p.timestamp - outstanding[key]
                if 0 < rtt < 10.0:
                    samples.append(rtt)
    if not samples:
        return None
    return min(samples)


def advertised_window(packets: list[PacketRecord]) -> int | None:
    wins = [p.window for p in packets if p.window > 0]
    return max(wins) if wins else None


def extract_epochs_sequential(packets: list[PacketRecord]) -> list[Epoch]:
    """Alternate advances in initiator seq vs acceptor seq (Smith / Tmix)."""
    packets = sorted(packets, key=lambda p: (p.timestamp, p.seq))
    a_acc = 0
    b_acc = 0
    req_start: float | None = None
    req_end: float | None = None
    resp_start: float | None = None
    resp_end: float | None = None
    epochs: list[Epoch] = []
    side: int | None = None
    idx = 0

    def emit(think: float) -> None:
        nonlocal a_acc, b_acc, req_start, req_end, resp_start, resp_end, idx
        if a_acc == 0 and b_acc == 0:
            return
        tt_a = (
            max(0.0, req_end - req_start)
            if req_start is not None and req_end is not None
            else None
        )
        tt_b = (
            max(0.0, resp_end - resp_start)
            if resp_start is not None and resp_end is not None
            else None
        )
        epochs.append(
            Epoch(
                a=a_acc,
                b=b_acc,
                t=max(0.0, think),
                epoch_index=idx,
                direction=0,
                transfer_time_a=tt_a,
                transfer_time_b=tt_b,
            )
        )
        idx += 1
        a_acc = 0
        b_acc = 0
        req_start = req_end = resp_start = resp_end = None

    for p in packets:
        if p.payload_len <= 0:
            continue
        if p.direction == 0:
            if side == 1:
                think = 0.0 if resp_end is None else max(0.0, p.timestamp - resp_end)
                emit(think)
            if req_start is None:
                req_start = p.timestamp
            req_end = p.timestamp
            a_acc += p.payload_len
            side = 0
        else:
            if resp_start is None:
                resp_start = p.timestamp
            resp_end = p.timestamp
            b_acc += p.payload_len
            side = 1

    emit(0.0)
    return epochs


def extract_epochs_unidirectional(packets: list[PacketRecord], direction: int) -> list[Epoch]:
    segs = [p for p in packets if p.payload_len > 0 and p.direction == direction]
    segs.sort(key=lambda p: p.timestamp)
    if not segs:
        return []
    epochs: list[Epoch] = []
    acc = 0
    start = segs[0].timestamp
    end = segs[0].timestamp
    last_end: float | None = None
    idx = 0
    prev_seq_end = segs[0].seq

    def flush(think: float) -> None:
        nonlocal acc, start, end, idx, last_end
        if acc <= 0:
            return
        tt = max(0.0, end - start)
        if direction == 0:
            ep = Epoch(a=acc, b=0, t=think, epoch_index=idx, direction=0, concurrent=True, transfer_time_a=tt)
        else:
            ep = Epoch(a=0, b=acc, t=think, epoch_index=idx, direction=1, concurrent=True, transfer_time_b=tt)
        epochs.append(ep)
        idx += 1
        last_end = end
        acc = 0

    gap_idle = IDLE_THRESHOLD_S
    for p in segs:
        gap = p.timestamp - end if acc else 0.0
        seq_jump = p.seq > prev_seq_end + 1 and acc > 0
        if acc and (gap > gap_idle or seq_jump):
            think = max(0.0, p.timestamp - end)
            flush(think)
            start = p.timestamp
        if acc == 0:
            start = p.timestamp
        acc += p.payload_len
        end = p.timestamp
        prev_seq_end = p.seq + p.payload_len
    flush(0.0)
    return epochs


def extract_connection(packets: list[PacketRecord], connection_id: int = 0) -> tuple[list[Epoch], dict]:
    if not packets:
        return [], {"complete": False, "concurrent": False}
    oriented, _ = _orient(packets)
    complete = is_complete(oriented)
    concurrent = is_concurrent(oriented)
    rtt = min_rtt(oriented)
    win = advertised_window(oriented)
    if concurrent:
        epochs = extract_epochs_unidirectional(oriented, 0) + extract_epochs_unidirectional(oriented, 1)
        epochs.sort(key=lambda e: e.epoch_index)
        for i, e in enumerate(epochs):
            e.epoch_index = i
            e.connection_id = connection_id
            e.rtt = rtt
    else:
        epochs = extract_epochs_sequential(oriented)
        for e in epochs:
            e.connection_id = connection_id
            e.rtt = rtt
    meta = {
        "complete": complete,
        "concurrent": concurrent,
        "min_rtt": rtt,
        "adv_window": win,
    }
    return epochs, meta


def extract_pcap_packets(
    packets: list[PacketRecord],
    *,
    conditions: Conditions | None = None,
    require_complete: bool = True,
    app_kind: str = "unknown",
) -> Trace:
    groups: dict[tuple, list[PacketRecord]] = defaultdict(list)
    for p in packets:
        groups[_key(p)].append(p)

    stats = ExtractStats(n_connections=len(groups))
    all_epochs: list[Epoch] = []
    cid = 0
    kept_packets: list[PacketRecord] = []
    for _k, pkts in groups.items():
        pkts = sorted(pkts, key=lambda p: p.timestamp)
        epochs, meta = extract_connection(pkts, connection_id=cid)
        complete = bool(meta["complete"])
        if meta["concurrent"]:
            stats.n_concurrent += 1
        if complete:
            stats.n_complete += 1
        else:
            stats.n_incomplete += 1
            stats.notes.append(f"conn {cid}: incomplete (syn/fin filter)")
            if require_complete:
                cid += 1
                continue
        for e in epochs:
            e.session_id = 0
        all_epochs.extend(epochs)
        oriented, _ = _orient(pkts)
        kept_packets.extend(oriented)
        cid += 1

    stats.n_epochs = len(all_epochs)
    stats.drop_rate = (
        stats.n_incomplete / stats.n_connections if stats.n_connections else 0.0
    )
    cond = conditions or preset_conditions("wan")
    return Trace(
        epochs=all_epochs,
        conditions=cond,
        packets=kept_packets,
        app_kind=app_kind,
        extract=stats,
        metadata={"require_complete": require_complete},
    )


def extract_pcap_file(path: str, **kwargs) -> Trace:
    from netcond.realize.pcap_io import read_pcap

    return extract_pcap_packets(read_pcap(path), **kwargs)
