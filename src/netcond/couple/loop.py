"""Single coupling point. App and network must not call each other elsewhere."""

from __future__ import annotations

import torch
from torch import Tensor

from netcond.app.tpp import ConditionedMarkedTPP
from netcond.net.flow_state import FlowNetwork
from netcond.types import Conditions, Epoch, Trace


class CoupledLoop:
    def __init__(self, app: ConditionedMarkedTPP, net: FlowNetwork) -> None:
        if app.feedback_dim != net.latent_dim:
            raise ValueError(
                f"feedback_dim {app.feedback_dim} != latent_dim {net.latent_dim}"
            )
        self.app = app
        self.net = net

    def rollout(
        self,
        conditions: Conditions,
        context: Tensor,
        n_steps: int,
        *,
        adaptive_feedback: bool = True,
        unconditioned: bool = False,
        generator: torch.Generator | None = None,
    ) -> Trace:
        device = next(self.app.parameters()).device
        if context.dim() == 1:
            context = context.unsqueeze(0)
        context = context.to(device)
        c = conditions.as_tensor(device).unsqueeze(0)
        z = self.net.initial_state(c, 1, device=device)
        h = self.app.initial_state(1, device=device)
        prev = torch.zeros(1, 4, device=device)
        epochs: list[Epoch] = []
        z_frozen = z
        gen = generator if device.type == "cpu" else None
        for i in range(n_steps):
            if unconditioned:
                z_in = torch.zeros_like(z)
            else:
                z_in = z if adaptive_feedback else z_frozen.detach()
            params, h = self.app.step(h, context, z_in, prev)
            from netcond.app.tpp import sample_mix_lognormal

            a = sample_mix_lognormal(params["a_w"], params["a_mu"], params["a_s"], gen).clamp(80, 20_000)
            b = sample_mix_lognormal(params["b_w"], params["b_mu"], params["b_s"], gen).clamp(500, 5_000_000)
            t = sample_mix_lognormal(params["t_w"], params["t_mu"], params["t_s"], gen).clamp(1e-3, 30.0)
            if i == n_steps - 1:
                t = torch.zeros_like(t)
            d = torch.argmax(params["dir_logits"], dim=-1)
            feat = torch.stack([a.log(), b.log()], dim=-1)
            z, obs = self.net.step(z, feat, c)
            epochs.append(
                Epoch(
                    a=int(a.item()),
                    b=int(b.item()),
                    t=float(t.item()),
                    epoch_index=i,
                    direction=int(d.item()),
                    transfer_time_a=float(obs["transfer_a"].item()),
                    transfer_time_b=float(obs["transfer_b"].item()),
                    rtt=float(obs["rtt"].item()),
                    loss=float(obs["loss"].item()),
                )
            )
            prev = torch.stack([a.log(), b.log(), t.clamp_min(1e-6).log(), d.float()], dim=-1)
        return Trace(
            epochs=epochs,
            conditions=conditions,
            metadata={"adaptive_feedback": adaptive_feedback, "unconditioned": unconditioned},
        )

    def counterfactual_network(
        self,
        conditions_list: list[Conditions],
        context: Tensor,
        n_steps: int,
        *,
        adaptive_feedback: bool = True,
        generator: torch.Generator | None = None,
    ) -> list[Trace]:
        """Fix application weights; vary do(c)."""
        return [
            self.rollout(c, context, n_steps, adaptive_feedback=adaptive_feedback, generator=generator)
            for c in conditions_list
        ]
