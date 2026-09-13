"""Intensity-free marked TPP (Shchur, Biloš, Günnemann, ICLR 2019).

Marks: request size a, response size b, think time t, direction.
Think time is a log-normal mixture — never MSE on raw seconds.
Conditioned on network latent z_t (concat). Backbone is GRU; not the claim.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

LOG2PI = math.log(2.0 * math.pi)
EPS = 1e-6


def mix_lognormal_nll(x: Tensor, log_w: Tensor, mu: Tensor, log_s: Tensor) -> Tensor:
    """x: (...,) positive; mixture params (..., K). Returns (...,) NLL."""
    logx = torch.log(x.clamp_min(EPS))
    s = log_s.exp().clamp(1e-4, 10.0)
    log_comp = (
        -logx.unsqueeze(-1)
        - log_s
        - 0.5 * LOG2PI
        - 0.5 * ((logx.unsqueeze(-1) - mu) / s) ** 2
    )
    log_w = torch.log_softmax(log_w, dim=-1)
    return -torch.logsumexp(log_w + log_comp, dim=-1)


def sample_mix_lognormal(log_w: Tensor, mu: Tensor, log_s: Tensor, generator: torch.Generator | None = None) -> Tensor:
    probs = torch.softmax(log_w, dim=-1)
    k = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1, generator=generator).view(*probs.shape[:-1])
    mu_k = mu.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    s_k = log_s.gather(-1, k.unsqueeze(-1)).squeeze(-1).exp().clamp(1e-4, 10.0)
    eps = torch.randn(mu_k.shape, generator=generator, device=mu_k.device, dtype=mu_k.dtype)
    return torch.exp(mu_k + s_k * eps)


class ConditionedMarkedTPP(nn.Module):
    def __init__(
        self,
        feedback_dim: int = 16,
        hidden_dim: int = 64,
        context_dim: int = 4,
        n_mix: int = 3,
    ) -> None:
        super().__init__()
        self.feedback_dim = feedback_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.n_mix = n_mix
        # prev marks: log a, log b, log t, direction
        self.in_dim = 4 + context_dim + feedback_dim
        self.gru = nn.GRUCell(self.in_dim, hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        k = n_mix
        self.a_w = nn.Linear(hidden_dim, k)
        self.a_mu = nn.Linear(hidden_dim, k)
        self.a_s = nn.Linear(hidden_dim, k)
        self.b_w = nn.Linear(hidden_dim, k)
        self.b_mu = nn.Linear(hidden_dim, k)
        self.b_s = nn.Linear(hidden_dim, k)
        self.t_w = nn.Linear(hidden_dim, k)
        self.t_mu = nn.Linear(hidden_dim, k)
        self.t_s = nn.Linear(hidden_dim, k)
        self.dir = nn.Linear(hidden_dim, 2)

    def initial_state(self, batch_size: int, device: str | None = None) -> Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def _pack_in(self, prev_marks: Tensor, context: Tensor, z: Tensor) -> Tensor:
        return torch.cat([prev_marks, context, z], dim=-1)

    def _params(self, h: Tensor) -> dict[str, Tensor]:
        f = self.head(h)
        return {
            "a_w": self.a_w(f),
            "a_mu": self.a_mu(f),
            "a_s": self.a_s(f),
            "b_w": self.b_w(f),
            "b_mu": self.b_mu(f),
            "b_s": self.b_s(f),
            "t_w": self.t_w(f),
            "t_mu": self.t_mu(f),
            "t_s": self.t_s(f),
            "dir_logits": self.dir(f),
        }

    def step(
        self, hidden: Tensor, context: Tensor, z: Tensor, prev_marks: Tensor | None = None
    ) -> tuple[dict[str, Tensor], Tensor]:
        bsz = hidden.size(0)
        if prev_marks is None:
            prev_marks = torch.zeros(bsz, 4, device=hidden.device, dtype=hidden.dtype)
        x = self._pack_in(prev_marks, context, z)
        h = self.gru(x, hidden)
        return self._params(h), h

    def forward_sequence(
        self,
        marks: dict[str, Tensor],
        z: Tensor,
        context: Tensor,
    ) -> dict[str, Tensor]:
        """Teacher-forced NLL. marks: a,b,t,direction (B,T); z (B,T,Z); context (B,C) or (B,T,C)."""
        a, b, t, d = marks["a"], marks["b"], marks["t"], marks["direction"]
        B, T = a.shape
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(B, T, -1)
        h = self.initial_state(B, device=a.device)
        prev = torch.zeros(B, 4, device=a.device, dtype=a.dtype)
        nll_a = []
        nll_b = []
        nll_t = []
        nll_d = []
        params_t = []
        e_logt = []
        for i in range(T):
            p, h = self.step(h, context[:, i], z[:, i], prev)
            nll_a.append(mix_lognormal_nll(a[:, i].float(), p["a_w"], p["a_mu"], p["a_s"]))
            nll_b.append(mix_lognormal_nll(b[:, i].float(), p["b_w"], p["b_mu"], p["b_s"]))
            think = t[:, i].float()
            nll_ti = mix_lognormal_nll(think.clamp_min(EPS), p["t_w"], p["t_mu"], p["t_s"])
            mask = (think > EPS).float()
            nll_t.append(nll_ti * mask)
            w = torch.softmax(p["t_w"], dim=-1)
            e_logt.append((w * p["t_mu"]).sum(-1))
            nll_d.append(
                torch.nn.functional.cross_entropy(p["dir_logits"], d[:, i].long(), reduction="none")
            )
            params_t.append(p)
            prev = torch.stack(
                [
                    torch.log(a[:, i].float().clamp_min(1.0)),
                    torch.log(b[:, i].float().clamp_min(1.0)),
                    torch.log(think.clamp_min(EPS)),
                    d[:, i].float(),
                ],
                dim=-1,
            )
        out = {
            "nll_a": torch.stack(nll_a, dim=1),
            "nll_b": torch.stack(nll_b, dim=1),
            "nll_t": torch.stack(nll_t, dim=1),
            "nll_d": torch.stack(nll_d, dim=1),
            "t": t,
            "log_t": torch.log(t.float().clamp_min(EPS)),
            "pred_log_t": torch.stack(e_logt, dim=1),
        }
        # last-step params for sampling diagnostics
        out["params_last"] = params_t[-1] if params_t else {}
        return out

    def sample_sequence(
        self,
        z: Tensor,
        context: Tensor,
        n_steps: int,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        B = z.size(0)
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(B, n_steps, -1)
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(B, n_steps, -1)
        h = self.initial_state(B, device=z.device)
        prev = torch.zeros(B, 4, device=z.device)
        a_s, b_s, t_s, d_s = [], [], [], []
        for i in range(n_steps):
            p, h = self.step(h, context[:, i], z[:, i], prev)
            a = sample_mix_lognormal(p["a_w"], p["a_mu"], p["a_s"], generator).clamp(80, 20_000)
            b = sample_mix_lognormal(p["b_w"], p["b_mu"], p["b_s"], generator).clamp(500, 5_000_000)
            t = sample_mix_lognormal(p["t_w"], p["t_mu"], p["t_s"], generator).clamp(1e-3, 30.0)
            d = torch.argmax(p["dir_logits"], dim=-1)
            if i == n_steps - 1:
                t = torch.zeros_like(t)
            a_s.append(a)
            b_s.append(b)
            t_s.append(t)
            d_s.append(d.float())
            prev = torch.stack([a.log(), b.log(), t.clamp_min(EPS).log(), d.float()], dim=-1)
        return {
            "a": torch.stack(a_s, dim=1),
            "b": torch.stack(b_s, dim=1),
            "t": torch.stack(t_s, dim=1),
            "direction": torch.stack(d_s, dim=1),
        }
