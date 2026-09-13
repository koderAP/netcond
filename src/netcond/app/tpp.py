"""Intensity-free marked TPP (Shchur et al., ICLR 2019).

GRU runs on exogenous history only (Tmix: think time is not network-produced).
Network latent z_t modulates the DASH bitrate-ladder head — that is the
adaptive mark. HTTP sizes stay a log-normal mixture on h.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from netcond.ladder import LADDER_BYTES, N_LADDER

LOG2PI = math.log(2.0 * math.pi)
EPS = 1e-6
_LADDER = None


def ladder_tensor(device, dtype) -> Tensor:
    global _LADDER
    t = torch.tensor(LADDER_BYTES, device=device, dtype=dtype)
    return t


def mix_lognormal_nll(x: Tensor, log_w: Tensor, mu: Tensor, log_s: Tensor) -> Tensor:
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
    gen = generator if log_w.device.type == "cpu" else None
    k = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1, generator=gen).view(*probs.shape[:-1])
    mu_k = mu.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    s_k = log_s.gather(-1, k.unsqueeze(-1)).squeeze(-1).exp().clamp(1e-4, 10.0)
    eps = torch.randn(mu_k.shape, generator=gen, device=mu_k.device, dtype=mu_k.dtype)
    return torch.exp(mu_k + s_k * eps)


class ConditionedMarkedTPP(nn.Module):
    def __init__(
        self,
        feedback_dim: int = 32,
        hidden_dim: int = 128,
        context_dim: int = 4,
        n_mix: int = 3,
    ) -> None:
        super().__init__()
        self.feedback_dim = feedback_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.n_mix = n_mix
        self.in_dim = 4 + context_dim  # no z in the application recurrence
        self.gru = nn.GRUCell(self.in_dim, hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.hz = nn.Sequential(nn.Linear(hidden_dim + feedback_dim, hidden_dim), nn.SiLU())
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
        self.ladder = nn.Linear(hidden_dim, N_LADDER)

    def initial_state(self, batch_size: int, device: str | None = None) -> Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def _params(self, h: Tensor, z: Tensor) -> dict[str, Tensor]:
        f = self.head(h)
        g = self.hz(torch.cat([h, z], dim=-1))
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
            "ladder_logits": self.ladder(g),
        }

    def step(
        self, hidden: Tensor, context: Tensor, z: Tensor, prev_marks: Tensor | None = None
    ) -> tuple[dict[str, Tensor], Tensor]:
        bsz = hidden.size(0)
        if prev_marks is None:
            prev_marks = torch.zeros(bsz, 4, device=hidden.device, dtype=hidden.dtype)
        x = torch.cat([prev_marks, context], dim=-1)
        h = self.gru(x, hidden)
        return self._params(h, z), h

    def sample_marks(self, params: dict[str, Tensor], context: Tensor, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        a = sample_mix_lognormal(params["a_w"], params["a_mu"], params["a_s"], generator).clamp(80, 20_000)
        t = sample_mix_lognormal(params["t_w"], params["t_mu"], params["t_s"], generator).clamp(1e-3, 30.0)
        d = torch.argmax(params["dir_logits"], dim=-1)
        mix_b = sample_mix_lognormal(params["b_w"], params["b_mu"], params["b_s"], generator).clamp(500, 5_000_000)
        probs = torch.softmax(params["ladder_logits"], dim=-1)
        gen = generator if params["ladder_logits"].device.type == "cpu" else None
        k = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        lad = ladder_tensor(a.device, a.dtype)[k]
        dash = (context[:, 0] > 0.5).to(a.dtype)
        b = dash * lad + (1.0 - dash) * mix_b
        return {"a": a, "b": b, "t": t, "direction": d.float()}

    def forward_sequence(
        self,
        marks: dict[str, Tensor],
        z: Tensor,
        context: Tensor,
    ) -> dict[str, Tensor]:
        a, b, t, d = marks["a"], marks["b"], marks["t"], marks["direction"]
        B, T = a.shape
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(B, T, -1)
        h = self.initial_state(B, device=a.device)
        prev = torch.zeros(B, 4, device=a.device, dtype=a.dtype)
        nll_a, nll_b, nll_t, nll_d, nll_lad = [], [], [], [], []
        e_logt = []
        log_ladder = ladder_tensor(a.device, a.dtype).log()
        for i in range(T):
            p, h = self.step(h, context[:, i], z[:, i], prev)
            nll_a.append(mix_lognormal_nll(a[:, i].float(), p["a_w"], p["a_mu"], p["a_s"]))
            nll_mix_b = mix_lognormal_nll(b[:, i].float(), p["b_w"], p["b_mu"], p["b_s"])
            logb = b[:, i].float().clamp_min(1.0).log().unsqueeze(-1)
            idx = (logb - log_ladder).abs().argmin(dim=-1)
            nll_l = torch.nn.functional.cross_entropy(p["ladder_logits"], idx, reduction="none")
            ad = (context[:, i, 0] > 0.5).float()
            nll_b.append(ad * nll_l + (1.0 - ad) * nll_mix_b)
            nll_lad.append(nll_l)
            think = t[:, i].float()
            nll_ti = mix_lognormal_nll(think.clamp_min(EPS), p["t_w"], p["t_mu"], p["t_s"])
            nll_t.append(nll_ti * (think > EPS).float())
            w = torch.softmax(p["t_w"], dim=-1)
            e_logt.append((w * p["t_mu"]).sum(-1))
            nll_d.append(torch.nn.functional.cross_entropy(p["dir_logits"], d[:, i].long(), reduction="none"))
            prev = torch.stack(
                [
                    torch.log(a[:, i].float().clamp_min(1.0)),
                    torch.log(b[:, i].float().clamp_min(1.0)),
                    torch.log(think.clamp_min(EPS)),
                    d[:, i].float(),
                ],
                dim=-1,
            )
        return {
            "nll_a": torch.stack(nll_a, dim=1),
            "nll_b": torch.stack(nll_b, dim=1),
            "nll_t": torch.stack(nll_t, dim=1),
            "nll_d": torch.stack(nll_d, dim=1),
            "nll_ladder": torch.stack(nll_lad, dim=1),
            "t": t,
            "log_t": torch.log(t.float().clamp_min(EPS)),
            "pred_log_t": torch.stack(e_logt, dim=1),
        }

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
            s = self.sample_marks(p, context[:, i], generator)
            if i == n_steps - 1:
                s["t"] = torch.zeros_like(s["t"])
            a_s.append(s["a"])
            b_s.append(s["b"])
            t_s.append(s["t"])
            d_s.append(s["direction"])
            prev = torch.stack([s["a"].log(), s["b"].log(), s["t"].clamp_min(EPS).log(), s["direction"]], dim=-1)
        return {
            "a": torch.stack(a_s, dim=1),
            "b": torch.stack(b_s, dim=1),
            "t": torch.stack(t_s, dim=1),
            "direction": torch.stack(d_s, dim=1),
        }
