"""Train conditioned TPP + flow network on interventional traces."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import Adam

from netcond.app.tpp import ConditionedMarkedTPP
from netcond.couple.loop import CoupledLoop
from netcond.data.dataset import Batch, traces_to_batch
from netcond.data.intervene import load_traces
from netcond.net.flow_state import FlowNetwork
from netcond.train_losses import total_loss


def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_models(latent_dim: int = 32, hidden_dim: int = 128) -> CoupledLoop:
    app = ConditionedMarkedTPP(feedback_dim=latent_dim, hidden_dim=hidden_dim)
    net = FlowNetwork(latent_dim=latent_dim, hidden_dim=hidden_dim)
    return CoupledLoop(app, net)


def _epoch_feat(batch: Batch) -> torch.Tensor:
    return torch.stack([batch.a.clamp_min(1).log(), batch.b.clamp_min(1).log()], dim=-1)


def _slice_batch(batch: Batch, idx: torch.Tensor) -> Batch:
    kwargs = {}
    for f in batch.__dataclass_fields__:
        t = getattr(batch, f)
        kwargs[f] = t[idx]
    return Batch(**kwargs)


def train_loop(
    traces,
    *,
    epochs: int = 20,
    seed: int = 0,
    device: str = "auto",
    adaptive_joint: bool = True,
    unconditioned: bool = False,
    batch_size: int = 32,
    hidden_dim: int = 128,
    latent_dim: int = 32,
) -> dict:
    device = resolve_device(device)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    loop = build_models(latent_dim=latent_dim, hidden_dim=hidden_dim)
    loop.app.to(device)
    loop.net.to(device)
    opt = Adam(list(loop.app.parameters()) + list(loop.net.parameters()), lr=3e-3)
    history: list[dict] = []
    n_app = max(1, epochs // 3)
    n_net = max(1, epochs // 3)
    full = traces_to_batch(traces, device=device)
    n = full.a.size(0)
    for ep in range(epochs):
        if ep < n_app:
            phase = "pretrain_app"
        elif ep < n_app + n_net:
            phase = "pretrain_net"
        else:
            phase = "joint"
        perm = torch.randperm(n, device=device)
        ep_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = _slice_batch(full, idx)
            z0 = loop.net.initial_state(batch.conditions, batch.a.size(0), device=device)
            feat = _epoch_feat(batch)
            T = batch.a.size(1)
            if unconditioned:
                z_seq = torch.zeros(batch.a.size(0), T, loop.net.latent_dim, device=device)
                app_out = loop.app.forward_sequence(
                    {"a": batch.a, "b": batch.b, "t": batch.t, "direction": batch.direction},
                    z_seq,
                    batch.context,
                )
                _z_path, net_obs = loop.net.forward_sequence(feat, batch.conditions, z0)
                if phase == "pretrain_app":
                    net_obs = {k: v.detach() for k, v in net_obs.items()}
                elif phase == "pretrain_net":
                    app_out = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in app_out.items()}
                elif unconditioned:
                    # joint but no z feedback: app sees zeros, both heads train
                    pass
            elif phase == "pretrain_app":
                z_seq = z0.unsqueeze(1).expand(-1, T, -1).detach()
                app_out = loop.app.forward_sequence(
                    {"a": batch.a, "b": batch.b, "t": batch.t, "direction": batch.direction},
                    z_seq,
                    batch.context,
                )
                _z_path, net_obs = loop.net.forward_sequence(feat, batch.conditions, z0)
                net_obs = {k: v.detach() for k, v in net_obs.items()}
            elif phase == "pretrain_net":
                _z_path, net_obs = loop.net.forward_sequence(feat, batch.conditions, z0)
                z_seq = _z_path.detach()
                app_out = loop.app.forward_sequence(
                    {"a": batch.a, "b": batch.b, "t": batch.t, "direction": batch.direction},
                    z_seq,
                    batch.context,
                )
                app_out = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in app_out.items()}
            else:
                z_path, net_obs = loop.net.forward_sequence(feat, batch.conditions, z0)
                z_det = z_path.detach()
                mix = batch.adaptive.view(-1, 1, 1)
                z_in = mix * z_path + (1.0 - mix) * z_det
                app_out = loop.app.forward_sequence(
                    {"a": batch.a, "b": batch.b, "t": batch.t, "direction": batch.direction},
                    z_in,
                    batch.context,
                )
            losses = total_loss(app_out, net_obs, batch, phase=phase)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(list(loop.app.parameters()) + list(loop.net.parameters()), 5.0)
            opt.step()
            ep_losses.append(losses)
        last = ep_losses[-1]
        row = {k: float(v.item()) if torch.is_tensor(v) else v for k, v in last.items()}
        row["phase"] = phase
        row["epoch"] = ep
        history.append(row)
    return {"loop": loop, "history": history, "device": device}


def save_checkpoint(loop: CoupledLoop, path: str | Path, history=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "app": loop.app.state_dict(),
            "net": loop.net.state_dict(),
            "history": history,
            "cfg": {"latent_dim": loop.net.latent_dim, "hidden_dim": loop.app.hidden_dim},
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> CoupledLoop:
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    cfg = blob.get("cfg") or {}
    loop = build_models(latent_dim=cfg.get("latent_dim", 32), hidden_dim=cfg.get("hidden_dim", 128))
    loop.app.load_state_dict(blob["app"])
    loop.net.load_state_dict(blob["net"])
    loop.app.to(device)
    loop.net.to(device)
    return loop


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="checkpoints/netcond.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--unconditioned", action="store_true")
    args = p.parse_args(argv)
    traces = load_traces(args.data)
    result = train_loop(
        traces,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        unconditioned=args.unconditioned,
    )
    save_checkpoint(result["loop"], args.out, result["history"])
    Path(args.out).with_suffix(".history.json").write_text(json.dumps(result["history"], indent=2))
    print(json.dumps({"device": result["device"], **result["history"][-1]}, indent=2))


if __name__ == "__main__":
    main()
