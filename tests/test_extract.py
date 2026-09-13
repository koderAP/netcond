"""Tmix extractor tests on a hand-crafted packet sequence with known a, b, t."""

from pathlib import Path

from netcond.extract.tmix import (
    extract_connection,
    extract_epochs_sequential,
    extract_pcap_packets,
    is_complete,
    is_concurrent,
)
from netcond.realize.emulator import DumbbellRealizer
from netcond.realize.pcap_io import read_pcap, write_pcap
from netcond.types import Epoch, PacketRecord, preset_conditions


def _pkt(ts, seq, ack, plen, flags, direction) -> PacketRecord:
    if direction == 0:
        src, dst, sp, dp = "10.0.0.1", "10.0.0.2", 4000, 80
    else:
        src, dst, sp, dp = "10.0.0.2", "10.0.0.1", 80, 4000
    return PacketRecord(
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


def _conversation() -> list[PacketRecord]:
    """Two sequential epochs: (a=100,b=200,t=0.5) then (a=50,b=80,t=0)."""
    return [
        _pkt(0.00, 0, 0, 0, "S", 0),
        _pkt(0.05, 0, 1, 0, "SA", 1),
        _pkt(0.10, 1, 1, 0, "A", 0),
        _pkt(0.20, 1, 1, 100, "PA", 0),
        _pkt(0.25, 1, 101, 0, "A", 1),
        _pkt(0.40, 1, 101, 200, "PA", 1),
        _pkt(0.45, 101, 201, 0, "A", 0),
        # think 0.50s after last response byte at t=0.40
        _pkt(0.90, 101, 201, 50, "PA", 0),
        _pkt(0.95, 201, 151, 0, "A", 1),
        _pkt(1.10, 201, 151, 80, "PA", 1),
        _pkt(1.15, 151, 281, 0, "A", 0),
        _pkt(1.20, 151, 281, 0, "FA", 0),
        _pkt(1.25, 281, 152, 0, "FA", 1),
        _pkt(1.30, 152, 282, 0, "A", 0),
    ]


def test_known_abt():
    epochs = extract_epochs_sequential(_conversation())
    assert len(epochs) == 2
    assert epochs[0].a == 100
    assert epochs[0].b == 200
    assert abs(epochs[0].t - 0.50) < 1e-9
    assert epochs[1].a == 50
    assert epochs[1].b == 80
    assert epochs[1].t == 0.0


def test_complete_and_incomplete():
    full = _conversation()
    assert is_complete(full)
    no_fin = [p for p in full if "F" not in p.flags]
    assert not is_complete(no_fin)
    tr = extract_pcap_packets(no_fin, require_complete=True)
    assert tr.extract is not None
    assert tr.extract.n_incomplete == 1
    assert tr.extract.drop_rate == 1.0
    assert tr.epochs == []
    kept = extract_pcap_packets(no_fin, require_complete=False)
    assert len(kept.epochs) == 2


def test_concurrent_split():
    pkts = [
        _pkt(0.0, 0, 0, 0, "S", 0),
        _pkt(0.01, 0, 1, 0, "SA", 1),
        _pkt(0.02, 1, 1, 0, "A", 0),
        # both sides have unacked data
        _pkt(0.10, 10, 1, 100, "P", 0),
        _pkt(0.11, 20, 1, 80, "P", 1),
        _pkt(1.0, 110, 100, 0, "FA", 0),
        _pkt(1.1, 100, 111, 0, "R", 1),
    ]
    assert is_concurrent(pkts)
    epochs, meta = extract_connection(pkts, 0)
    assert meta["concurrent"] is True
    assert any(e.concurrent for e in epochs)
    assert any(e.a == 100 for e in epochs)
    assert any(e.b == 80 for e in epochs)


def test_pcap_roundtrip_extract(tmp_path: Path):
    realizer = DumbbellRealizer()
    epochs = [Epoch(a=100, b=200, t=0.25, epoch_index=0), Epoch(a=50, b=80, t=0.0, epoch_index=1)]
    c = preset_conditions("wan")
    pkts = realizer.realize(epochs, c, seed=0)
    path = tmp_path / "t.pcap"
    write_pcap(path, pkts)
    back = read_pcap(path)
    assert len(back) == len(pkts)
    tr = extract_pcap_packets(back, conditions=c, require_complete=True)
    assert tr.extract.n_complete == 1
    assert len(tr.epochs) == 2
    assert tr.epochs[0].a == 100
    assert tr.epochs[0].b == 200
    assert abs(tr.epochs[0].t - 0.25) < 0.05
    assert tr.epochs[1].a == 50
    assert tr.epochs[1].b == 80
