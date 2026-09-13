#!/usr/bin/env python
"""Collect interventional traces (do(c)) with a DASH-like client + HTTP GET."""

from __future__ import annotations

import argparse

from netcond.data.intervene import collect_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/samples/intervene.json")
    p.add_argument("--n-per-preset", type=int, default=8)
    p.add_argument("--pcap-dir", default=None)
    p.add_argument("--write-pcap", action="store_true")
    args = p.parse_args()
    traces = collect_dataset(
        args.out,
        n_per_preset=args.n_per_preset,
        write_pcap=args.write_pcap,
        pcap_dir=args.pcap_dir,
    )
    print(f"wrote {len(traces)} interventional traces to {args.out}")


if __name__ == "__main__":
    main()
