"""Narrow protocols. App and network interact only through the orchestrator."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from netcond.types import Conditions, Epoch, PacketRecord, Trace


@runtime_checkable
class ApplicationGenerator(Protocol):
    feedback_dim: int

    def initial_state(self, batch_size: int, device: str | None = None) -> Tensor: ...

    def step(
        self,
        hidden: Tensor,
        context: Tensor,
        z: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor]: ...

    def forward_sequence(
        self,
        marks: dict[str, Tensor],
        z: Tensor,
        context: Tensor,
    ) -> dict[str, Tensor]: ...


@runtime_checkable
class NetworkModel(Protocol):
    latent_dim: int

    def initial_state(
        self, conditions: Tensor, batch_size: int, device: str | None = None
    ) -> Tensor: ...

    def step(
        self,
        latent: Tensor,
        epoch_feat: Tensor,
        conditions: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]: ...

    def forward_sequence(
        self,
        epoch_feat: Tensor,
        conditions: Tensor,
        latent0: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]: ...


@runtime_checkable
class PacketRealizer(Protocol):
    def realize(self, epochs: list[Epoch], conditions: Conditions) -> list[PacketRecord]: ...

    def write_pcap(self, packets: list[PacketRecord], path: str) -> None: ...

    def realize_trace(self, trace: Trace) -> Trace: ...
