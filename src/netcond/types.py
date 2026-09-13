"""Core types: exogenous ADUs vs endogenous network quantities vs physical conditions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

import torch
from torch import Tensor

# Condition tensor: [log10(Mbps), log10(RTT ms), log10(buffer kB), loss, xt_id]
CONDITION_DIM = 5

PRESET_NAMES = ("lan", "wan", "congested", "lossy", "satellite")


def _buffer_for(capacity_bps: float, base_rtt_s: float, bdp_mult: float = 2.0) -> float:
    bdp = capacity_bps * base_rtt_s / 8.0
    return max(bdp * bdp_mult, 32_768.0)


@dataclass(frozen=True)
class Conditions:
    """NetReplica-style split: static envelope + dynamic pressure."""

    capacity_bps: float
    base_rtt_s: float
    buffer_bytes: float
    aqm: str = "droptail"
    loss_rate: float = 0.0
    cross_traffic_id: str = "none"
    preset: str = ""

    def as_tensor(self, device: torch.device | str | None = None) -> Tensor:
        xt = 0.0 if self.cross_traffic_id in ("", "none") else 1.0
        values = [
            torch.log10(torch.tensor(max(self.capacity_bps, 1.0) / 1e6)),
            torch.log10(torch.tensor(max(self.base_rtt_s, 1e-6) * 1e3)),
            torch.log10(torch.tensor(max(self.buffer_bytes, 1.0) / 1e3)),
            torch.tensor(float(self.loss_rate)),
            torch.tensor(xt),
        ]
        t = torch.stack(values).to(dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t

    @classmethod
    def from_tensor(cls, tensor: Tensor, *, aqm: str = "droptail", preset: str = "") -> Conditions:
        x = tensor.detach().cpu().flatten().tolist()
        if len(x) != CONDITION_DIM:
            raise ValueError(f"condition tensor dim {len(x)} != {CONDITION_DIM}")
        capacity = 10 ** x[0] * 1e6
        rtt = 10 ** x[1] / 1e3
        buffer = 10 ** x[2] * 1e3
        loss = max(0.0, float(x[3]))
        xt = "onoff" if x[4] >= 0.5 else "none"
        return cls(
            capacity_bps=capacity,
            base_rtt_s=rtt,
            buffer_bytes=buffer,
            aqm=aqm,
            loss_rate=loss,
            cross_traffic_id=xt,
            preset=preset,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Conditions:
        return cls(
            capacity_bps=float(d["capacity_bps"]),
            base_rtt_s=float(d["base_rtt_s"]),
            buffer_bytes=float(d["buffer_bytes"]),
            aqm=str(d.get("aqm", "droptail")),
            loss_rate=float(d.get("loss_rate", 0.0)),
            cross_traffic_id=str(d.get("cross_traffic_id", "none")),
            preset=str(d.get("preset", "")),
        )


def preset_conditions(name: str) -> Conditions:
    name = name.lower()
    table: dict[str, Conditions] = {
        "lan": Conditions(
            capacity_bps=1e9,
            base_rtt_s=0.001,
            buffer_bytes=_buffer_for(1e9, 0.001),
            preset="lan",
        ),
        "wan": Conditions(
            capacity_bps=50e6,
            base_rtt_s=0.040,
            buffer_bytes=_buffer_for(50e6, 0.040),
            preset="wan",
        ),
        "congested": Conditions(
            capacity_bps=8e6,
            base_rtt_s=0.060,
            buffer_bytes=_buffer_for(8e6, 0.060, bdp_mult=0.5),
            cross_traffic_id="onoff",
            preset="congested",
        ),
        "lossy": Conditions(
            capacity_bps=20e6,
            base_rtt_s=0.050,
            buffer_bytes=_buffer_for(20e6, 0.050),
            loss_rate=0.02,
            preset="lossy",
        ),
        "satellite": Conditions(
            capacity_bps=25e6,
            base_rtt_s=0.600,
            buffer_bytes=_buffer_for(25e6, 0.600, bdp_mult=1.0),
            preset="satellite",
        ),
    }
    if name not in table:
        raise KeyError(f"unknown preset {name!r}; choose from {PRESET_NAMES}")
    return table[name]


@dataclass
class Epoch:
    """Exogenous Tmix a-b-t epoch. Transfer time / RTT are labels, not app targets."""

    a: int
    b: int
    t: float
    epoch_index: int = 0
    direction: int = 0
    connection_id: int = 0
    session_id: int = 0
    concurrent: bool = False
    transfer_time_a: float | None = None
    transfer_time_b: float | None = None
    rtt: float | None = None
    loss: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Epoch:
        return cls(
            a=int(d["a"]),
            b=int(d["b"]),
            t=float(d["t"]),
            epoch_index=int(d.get("epoch_index", 0)),
            direction=int(d.get("direction", 0)),
            connection_id=int(d.get("connection_id", 0)),
            session_id=int(d.get("session_id", 0)),
            concurrent=bool(d.get("concurrent", False)),
            transfer_time_a=_opt_float(d.get("transfer_time_a")),
            transfer_time_b=_opt_float(d.get("transfer_time_b")),
            rtt=_opt_float(d.get("rtt")),
            loss=_opt_float(d.get("loss")),
        )


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    return float(v)


@dataclass
class PacketRecord:
    timestamp: float
    src: str
    dst: str
    src_port: int
    dst_port: int
    seq: int
    ack: int
    flags: str
    payload_len: int
    direction: int  # 0 initiator→acceptor, 1 reverse
    window: int = 65535

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PacketRecord:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})  # type: ignore[arg-type]


@dataclass
class ExtractStats:
    n_connections: int = 0
    n_complete: int = 0
    n_incomplete: int = 0
    n_concurrent: int = 0
    n_epochs: int = 0
    drop_rate: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class Trace:
    epochs: list[Epoch]
    conditions: Conditions
    packets: list[PacketRecord] = field(default_factory=list)
    app_kind: str = "http_get"
    session_id: int = 0
    extract: ExtractStats | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def think_times(self) -> list[float]:
        return [e.t for e in self.epochs if e.t > 0]

    def request_sizes(self) -> list[int]:
        return [e.a for e in self.epochs]

    def response_sizes(self) -> list[int]:
        return [e.b for e in self.epochs]

    def transfer_times(self) -> list[float]:
        out: list[float] = []
        for e in self.epochs:
            if e.transfer_time_b is not None:
                out.append(e.transfer_time_b)
            if e.transfer_time_a is not None:
                out.append(e.transfer_time_a)
        return out

    def mean_rtt(self) -> float | None:
        vals = [e.rtt for e in self.epochs if e.rtt is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def goodput_bps(self) -> float | None:
        total_b = 0.0
        total_t = 0.0
        for e in self.epochs:
            if e.transfer_time_b and e.transfer_time_b > 0:
                total_b += e.b
                total_t += e.transfer_time_b
        if total_t <= 0:
            return None
        return 8.0 * total_b / total_t

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "epochs": [e.to_dict() for e in self.epochs],
            "conditions": self.conditions.to_dict(),
            "packets": [p.to_dict() for p in self.packets],
            "app_kind": self.app_kind,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }
        if self.extract is not None:
            d["extract"] = asdict(self.extract)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Trace:
        extract = None
        if "extract" in d and d["extract"] is not None:
            extract = ExtractStats(**d["extract"])
        return cls(
            epochs=[Epoch.from_dict(e) for e in d.get("epochs", [])],
            conditions=Conditions.from_dict(d["conditions"]),
            packets=[PacketRecord.from_dict(p) for p in d.get("packets", [])],
            app_kind=str(d.get("app_kind", "http_get")),
            session_id=int(d.get("session_id", 0)),
            extract=extract,
            metadata=dict(d.get("metadata") or {}),
        )

    def without_packets(self) -> Trace:
        return replace(self, packets=[])
