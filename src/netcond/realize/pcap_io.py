"""Minimal PCAP I/O for IPv4/TCP (no scapy required)."""

from __future__ import annotations

import socket
import struct
from pathlib import Path

from netcond.types import PacketRecord

PCAP_GLOBAL = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
ETH_IPV4 = 0x0800
TCP_PROTO = 6


def _ip_to_bytes(addr: str) -> bytes:
    return socket.inet_aton(addr)


def _checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    s = sum(struct.unpack("!%dH" % (len(data) // 2), data))
    s = (s >> 16) + (s & 0xFFFF)
    s += s >> 16
    return (~s) & 0xFFFF


def _flag_bits(flags: str) -> int:
    bits = 0
    mapping = {"F": 0x01, "S": 0x02, "R": 0x04, "P": 0x08, "A": 0x10, "U": 0x20}
    upper = flags.upper()
    for k, v in mapping.items():
        if k in upper:
            bits |= v
    return bits


def _bits_to_flags(bits: int) -> str:
    out = []
    if bits & 0x02:
        out.append("S")
    if bits & 0x10:
        out.append("A")
    if bits & 0x08:
        out.append("P")
    if bits & 0x01:
        out.append("F")
    if bits & 0x04:
        out.append("R")
    return "".join(out) or "."


def build_tcp_frame(pkt: PacketRecord, payload: bytes | None = None) -> bytes:
    payload = payload if payload is not None else (b"X" * pkt.payload_len)
    if len(payload) != pkt.payload_len:
        payload = payload[: pkt.payload_len].ljust(pkt.payload_len, b"X")
    ihl = 20
    tcp_off = 20
    total = ihl + tcp_off + len(payload)
    src_ip = _ip_to_bytes(pkt.src)
    dst_ip = _ip_to_bytes(pkt.dst)
    ip = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total,
        0,
        0,
        64,
        TCP_PROTO,
        0,
        src_ip,
        dst_ip,
    )
    ip = ip[:10] + struct.pack("!H", _checksum(ip)) + ip[12:]
    flags = _flag_bits(pkt.flags)
    tcp = struct.pack(
        "!HHIIBBHHH",
        pkt.src_port,
        pkt.dst_port,
        pkt.seq & 0xFFFFFFFF,
        pkt.ack & 0xFFFFFFFF,
        (5 << 4),
        flags,
        pkt.window,
        0,
        0,
    )
    pseudo = src_ip + dst_ip + struct.pack("!BBH", 0, TCP_PROTO, tcp_off + len(payload))
    csum = _checksum(pseudo + tcp + payload)
    tcp = tcp[:16] + struct.pack("!H", csum) + tcp[18:]
    eth = b"\x02\x00\x00\x00\x00\x01\x02\x00\x00\x00\x00\x02" + struct.pack("!H", ETH_IPV4)
    return eth + ip + tcp + payload


def write_pcap(path: str | Path, packets: list[PacketRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(PCAP_GLOBAL)
        for pkt in packets:
            frame = build_tcp_frame(pkt)
            ts = pkt.timestamp
            sec = int(ts)
            usec = int(round((ts - sec) * 1e6))
            if usec >= 1_000_000:
                sec += 1
                usec -= 1_000_000
            f.write(struct.pack("<IIII", sec, usec, len(frame), len(frame)))
            f.write(frame)


def read_pcap(path: str | Path) -> list[PacketRecord]:
    data = Path(path).read_bytes()
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic == 0xA1B2C3D4:
        endian = "<"
    elif magic == 0xD4C3B2A1:
        endian = ">"
    else:
        raise ValueError(f"not a pcap: {path}")
    off = 24
    packets: list[PacketRecord] = []
    while off + 16 <= len(data):
        sec, usec, incl, _orig = struct.unpack_from(endian + "IIII", data, off)
        off += 16
        frame = data[off : off + incl]
        off += incl
        ts = sec + usec * 1e-6
        rec = _parse_frame(frame, ts)
        if rec is not None:
            packets.append(rec)
    return packets


def _parse_frame(frame: bytes, ts: float) -> PacketRecord | None:
    if len(frame) < 34:
        return None
    # Ethernet
    ethertype = struct.unpack_from("!H", frame, 12)[0]
    ip_off = 14
    if ethertype == 0x8100:
        ip_off = 18
        ethertype = struct.unpack_from("!H", frame, 16)[0]
    if ethertype != ETH_IPV4:
        # maybe raw IP
        if frame[0] >> 4 == 4:
            ip_off = 0
        else:
            return None
    if len(frame) < ip_off + 20:
        return None
    vihl = frame[ip_off]
    if vihl >> 4 != 4:
        return None
    ihl = (vihl & 0x0F) * 4
    proto = frame[ip_off + 9]
    if proto != TCP_PROTO:
        return None
    src = socket.inet_ntoa(frame[ip_off + 12 : ip_off + 16])
    dst = socket.inet_ntoa(frame[ip_off + 16 : ip_off + 20])
    tcp_off = ip_off + ihl
    if len(frame) < tcp_off + 20:
        return None
    sport, dport, seq, ack, off_res, flags, window = struct.unpack_from("!HHIIBBH", frame, tcp_off)
    doff = (off_res >> 4) * 4
    payload_len = max(0, len(frame) - (tcp_off + doff))
    return PacketRecord(
        timestamp=ts,
        src=src,
        dst=dst,
        src_port=sport,
        dst_port=dport,
        seq=seq,
        ack=ack,
        flags=_bits_to_flags(flags),
        payload_len=payload_len,
        direction=0,
        window=window,
    )
