"""m4-inspired flow-level network head.

Inputs: physical condition tensor c and emitted ADU sizes.
Outputs: latent z, transfer time, RTT. Trained on emulator labels, not PCAP heuristics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from netcond.types import CONDITION_DIM


class FlowNetwork(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_enc = nn.Sequential(
            nn.Linear(CONDITION_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.adu_enc = nn.Linear(2, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim + latent_dim, latent_dim)
        self.tt_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))
        self.rtt_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.loss_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def encode_conditions(self, conditions: Tensor) -> Tensor:
        return self.cond_enc(conditions.to(next(self.parameters()).device, dtype=torch.float32))

    def initial_state(self, conditions: Tensor, batch_size: int = 1, device: str | torch.device | None = None) -> Tensor:
        if conditions.dim() == 1:
            conditions = conditions.unsqueeze(0)
        if conditions.size(0) == 1 and batch_size > 1:
            conditions = conditions.expand(batch_size, -1)
        z = self.encode_conditions(conditions)
        if device is not None:
            z = z.to(device)
        return z

    def step(
        self, latent: Tensor, epoch_feat: Tensor, conditions: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        cond_z = self.encode_conditions(conditions)
        x = torch.cat([self.adu_enc(epoch_feat.to(cond_z.device)), cond_z], dim=-1)
        z = self.gru(x, latent.to(cond_z.device))
        c = conditions.to(z.device, dtype=torch.float32)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        cap_bps = (10.0 ** c[:, 0]) * 1e6
        base_rtt = (10.0 ** c[:, 1]) / 1e3
        sizes = epoch_feat.to(z.device).exp()
        xt = c[:, 4].clamp(0, 1)
        cap_eff = cap_bps * (1.0 - 0.45 * xt).clamp(min=0.15 * cap_bps)
        serial_a = sizes[:, 0] * 8.0 / cap_eff.clamp_min(1.0)
        serial_b = sizes[:, 1] * 8.0 / cap_eff.clamp_min(1.0)
        rtt = base_rtt * (1.0 + torch.nn.functional.softplus(self.rtt_head(z)).squeeze(-1))
        nseg_b = (sizes[:, 1] / 1460.0).clamp(min=1.0)
        rounds = torch.log2((nseg_b / 10.0).clamp(min=1.0)) + 1.0
        res = torch.tanh(self.tt_head(z))
        transfer_a = (serial_a + rtt).clamp_min(1e-6) * torch.exp(res[:, 0])
        transfer_b = (serial_b + rtt * rounds * (1.0 + 0.5 * xt)).clamp_min(1e-6) * torch.exp(res[:, 1])
        loss_p = torch.sigmoid(self.loss_head(z))
        obs = {
            "log_transfer_a": transfer_a.clamp_min(1e-6).log(),
            "log_transfer_b": transfer_b.clamp_min(1e-6).log(),
            "rtt": rtt,
            "loss": loss_p.squeeze(-1),
            "transfer_a": transfer_a,
            "transfer_b": transfer_b,
        }
        return z, obs

    def forward_sequence(
        self,
        epoch_feat: Tensor,
        conditions: Tensor,
        latent0: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """epoch_feat (B,T,2) log-sizes; conditions (B,C)."""
        B, T, _ = epoch_feat.shape
        z = latent0 if latent0 is not None else self.initial_state(conditions, B, device=epoch_feat.device)
        zs = []
        bag: dict[str, list[Tensor]] = {
            "transfer_a": [],
            "transfer_b": [],
            "rtt": [],
            "loss": [],
            "log_transfer_a": [],
            "log_transfer_b": [],
        }
        for i in range(T):
            z, obs = self.step(z, epoch_feat[:, i], conditions)
            zs.append(z)
            for k in bag:
                bag[k].append(obs[k])
        stacked = {k: torch.stack(v, dim=1) for k, v in bag.items()}
        return torch.stack(zs, dim=1), stacked
