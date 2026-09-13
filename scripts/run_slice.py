"""End-to-end slice: collect → train 3 seeds (+ uncond ablation) → generate → eval table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from netcond.data.intervene import collect_dataset, load_traces
from netcond.eval.report import evaluate, format_table
from netcond.generate import generate
from netcond.realize.emulator import DumbbellRealizer
from netcond.realize.pcap_io import write_pcap
from netcond.train import save_checkpoint, train_loop


def run(
    *,
    data_path: Path,
    out_dir: Path,
    n_per_preset: int = 12,
    train_epochs: int = 40,
    seeds: tuple[int, ...] = (0, 1, 2),
    collect: bool = True,
    device: str = "auto",
    presets: tuple[str, ...] = ("lan", "wan", "congested"),
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if collect or not data_path.exists():
        collect_dataset(
            data_path,
            n_per_preset=n_per_preset,
            presets=presets,
            apps=("http_get", "dash_like"),
            seed=11,
            write_pcap=True,
            pcap_dir=out_dir / "pcaps",
        )
    traces = load_traces(data_path)
    n = len(traces)
    train, hold = traces[: int(0.75 * n)], traces[int(0.75 * n) :]
    ckpts = []
    uncond = []
    for s in seeds:
        result = train_loop(train, epochs=train_epochs, seed=s, device=device, adaptive_joint=True)
        ck = out_dir / f"netcond_seed{s}.pt"
        save_checkpoint(result["loop"], ck, result["history"])
        ckpts.append(str(ck))
        ures = train_loop(
            train, epochs=train_epochs, seed=s, device=device, unconditioned=True, adaptive_joint=False
        )
        uck = out_dir / f"netcond_uncond_seed{s}.pt"
        save_checkpoint(ures["loop"], uck, ures["history"])
        uncond.append(str(uck))
        gen_lan = generate(result["loop"], "lan", app_kind="dash_like", n_sessions=4, n_steps=6, seed=s)
        gen_c = generate(result["loop"], "congested", app_kind="dash_like", n_sessions=4, n_steps=6, seed=s)
        (out_dir / f"gen_lan_seed{s}.json").write_text(
            json.dumps({"traces": [t.to_dict() for t in gen_lan]})
        )
        (out_dir / f"gen_congested_seed{s}.json").write_text(
            json.dumps({"traces": [t.to_dict() for t in gen_c]})
        )
        if s == seeds[0]:
            realizer = DumbbellRealizer()
            tr = realizer.realize_trace(gen_lan[0], seed=s)
            pcap_dir = out_dir / "pcaps"
            pcap_dir.mkdir(parents=True, exist_ok=True)
            write_pcap(pcap_dir / "generated_lan.pcap", tr.packets)
            print(f"train device={result['device']} last_loss={result['history'][-1]}")
    report = evaluate(
        ckpts, train, hold, preset="wan", app_kind="dash_like", uncond_ckpts=uncond
    )
    (out_dir / "eval.json").write_text(json.dumps(report, indent=2, default=str))
    table = format_table(report)
    (out_dir / "eval_table.txt").write_text(table)
    print(table)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="netcond first slice")
    p.add_argument("--data", default="data/samples/intervene.json")
    p.add_argument("--out", default="output/slice")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-per-preset", type=int, default=12)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-collect", action="store_true")
    args = p.parse_args()
    run(
        data_path=Path(args.data),
        out_dir=Path(args.out),
        n_per_preset=args.n_per_preset,
        train_epochs=args.epochs,
        collect=not args.no_collect,
        device=args.device,
    )


if __name__ == "__main__":
    main()
