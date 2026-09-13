"""Evaluate neural vs unconditioned TPP vs Tmix resample; multi-seed mean ± std."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from netcond.eval.baseline_resample import resample_traces
from netcond.eval.counterfactual import counterfactual_table, rtt_doubling_check
from netcond.eval.fidelity import fidelity_report
from netcond.eval.privacy import ngram_overlap
from netcond.eval.utility import tstr_condition_id
from netcond.eval.wavelet import wavelet_or_skip
from netcond.generate import generate
from netcond.train import load_checkpoint
from netcond.types import preset_conditions


def _mean_std(rows: list[dict], path: tuple[str, ...]) -> dict[str, float]:
    vals = []
    for r in rows:
        cur = r
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and isinstance(cur, (int, float)) and cur == cur:
            vals.append(float(cur))
    if not vals:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def _gen_by_preset(loop, presets, app_kind, n_sessions, n_steps, seed, unconditioned=False):
    out = []
    for p in presets:
        out.extend(
            generate(
                loop,
                p,
                app_kind=app_kind,
                n_sessions=n_sessions,
                n_steps=n_steps,
                seed=seed,
                unconditioned=unconditioned,
            )
        )
    return out


def evaluate(
    ckpts: list[str],
    train,
    holdout,
    *,
    preset: str = "wan",
    app_kind: str = "dash_like",
    uncond_ckpts: list[str] | None = None,
) -> dict:
    presets = ("lan", "wan", "congested")
    real = [tr for tr in holdout if tr.app_kind == app_kind and (not preset or tr.conditions.preset == preset)]
    if not real:
        real = [tr for tr in holdout if tr.app_kind == app_kind]
    real_all = [tr for tr in holdout if tr.app_kind == app_kind]
    train_app = [tr for tr in train if tr.app_kind == app_kind]
    per_seed = []
    for i, ck in enumerate(ckpts):
        loop = load_checkpoint(ck)
        syn = generate(loop, preset, app_kind=app_kind, n_sessions=max(8, len(real)), n_steps=8, seed=10 + i)
        syn_all = _gen_by_preset(loop, presets, app_kind, 8, 8, 20 + i)
        base = resample_traces(
            train,
            preset_conditions(preset),
            n_sessions=len(syn),
            n_steps=8,
            seed=10 + i,
            app_kind=app_kind,
        )
        fid_n = fidelity_report(real, syn)
        fid_b = fidelity_report(real, base)
        uncond_fid = {}
        if uncond_ckpts and i < len(uncond_ckpts):
            uloop = load_checkpoint(uncond_ckpts[i])
            usyn = generate(
                uloop,
                preset,
                app_kind=app_kind,
                n_sessions=len(syn),
                n_steps=8,
                seed=10 + i,
                unconditioned=True,
            )
            uncond_fid = fidelity_report(real, usyn)
        lan = generate(loop, "lan", app_kind=app_kind, n_sessions=8, n_steps=8, seed=30 + i)
        cong = generate(loop, "congested", app_kind=app_kind, n_sessions=8, n_steps=8, seed=30 + i)
        real_lan = [tr for tr in holdout if tr.app_kind == app_kind and tr.conditions.preset == "lan"]
        real_c = [tr for tr in holdout if tr.app_kind == app_kind and tr.conditions.preset == "congested"]

        def _mean_b(trs):
            xs = [e.b for tr in trs for e in tr.epochs]
            return float(sum(xs) / len(xs)) if xs else float("nan")

        b_shift = {
            "neural_lan": _mean_b(lan),
            "neural_congested": _mean_b(cong),
            "real_lan": _mean_b(real_lan),
            "real_congested": _mean_b(real_c),
            "neural_lan_gt_cong": _mean_b(lan) > _mean_b(cong),
            "real_lan_gt_cong": _mean_b(real_lan) > _mean_b(real_c) if real_lan and real_c else None,
        }
        cf = counterfactual_table(
            loop, holdout, train, from_preset="lan", to_preset="congested", app_kind=app_kind
        )
        rtt2 = rtt_doubling_check(loop, app_kind=app_kind)
        per_seed.append(
            {
                "seed_index": i,
                "neural": fid_n,
                "resample": fid_b,
                "uncond": uncond_fid,
                "counterfactual": cf,
                "rtt_2x": rtt2,
                "b_shift": b_shift,
                "privacy": ngram_overlap(train, syn),
                "wavelet": wavelet_or_skip(syn),
                "utility_tstr": tstr_condition_id(train_app, syn_all, real_all),
            }
        )
    summary = {
        "seeds": len(ckpts),
        "neural_emd_log_b": _mean_std(per_seed, ("neural", "b", "emd")),
        "resample_emd_log_b": _mean_std(per_seed, ("resample", "b", "emd")),
        "uncond_emd_log_b": _mean_std(per_seed, ("uncond", "b", "emd")),
        "neural_emd_think": _mean_std(per_seed, ("neural", "t_think", "emd")),
        "resample_emd_think": _mean_std(per_seed, ("resample", "t_think", "emd")),
        "neural_emd_transfer": _mean_std(per_seed, ("neural", "transfer_b", "emd")),
        "resample_emd_transfer": _mean_std(per_seed, ("resample", "transfer_b", "emd")),
        "neural_qq_b": _mean_std(per_seed, ("neural", "b", "qq_mae")),
        "tstr_acc": _mean_std(per_seed, ("utility_tstr", "tstr_acc")),
        "trtr_acc": _mean_std(per_seed, ("utility_tstr", "trtr_acc")),
        "rtt_2x_direction_ok": [r["rtt_2x"]["direction_ok"] for r in per_seed],
        "neural_b_lan_gt_cong": [r["b_shift"]["neural_lan_gt_cong"] for r in per_seed],
        "mean_b_lan": _mean_std(per_seed, ("b_shift", "neural_lan")),
        "mean_b_congested": _mean_std(per_seed, ("b_shift", "neural_congested")),
        "headline": "mean±std over seeds; not a single best iteration. Not KS pass-rate.",
    }
    return {"summary": summary, "per_seed": per_seed}


def format_table(report: dict) -> str:
    s = report["summary"]

    def fmt(d):
        if not isinstance(d, dict) or d.get("mean") != d.get("mean"):
            return "n/a"
        return f"{d['mean']:.4f} ± {d['std']:.4f}"

    lines = [
        "metric (holdout vs generated) | neural | uncond TPP | tmix-resample",
        "------------------------------|--------|------------|---------------",
        f"EMD log response size b       | {fmt(s['neural_emd_log_b'])} | {fmt(s['uncond_emd_log_b'])} | {fmt(s['resample_emd_log_b'])}",
        f"EMD log think time t          | {fmt(s['neural_emd_think'])} | n/a | {fmt(s['resample_emd_think'])}",
        f"EMD log transfer time         | {fmt(s['neural_emd_transfer'])} | n/a | {fmt(s['resample_emd_transfer'])}",
        f"Q-Q MAE log b                 | {fmt(s['neural_qq_b'])} | n/a | n/a",
        f"TSTR condition-id acc         | {fmt(s['tstr_acc'])} | n/a | TRTR {fmt(s['trtr_acc'])}",
        f"2×RTT direction ok            | {s['rtt_2x_direction_ok']} | n/a | n/a",
        f"mean b lan vs congested       | {fmt(s['mean_b_lan'])} vs {fmt(s['mean_b_congested'])} | n/a | n/a",
        f"lan>cong chunk size           | {s['neural_b_lan_gt_cong']} | n/a | n/a",
        f"seeds                         | {s['seeds']} | {s['seeds']} | {s['seeds']}",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    import argparse

    from netcond.data.dataset import split_traces
    from netcond.data.intervene import load_traces

    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--uncond-ckpts", nargs="*", default=None)
    p.add_argument("--out", default="output/eval.json")
    args = p.parse_args(argv)
    traces = load_traces(args.data)
    train, hold = split_traces(traces, frac=0.8, seed=0)
    report = evaluate(args.ckpts, train, hold, uncond_ckpts=args.uncond_ckpts or None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(format_table(report))


if __name__ == "__main__":
    main()
