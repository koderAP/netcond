"""Ingest a public header PCAP (MAWI / Wireshark sample) via the Tmix extractor.

If the download fails, use the recorded sample under data/samples/.
Passive traces are for fidelity only — not causal counterfactuals.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from netcond.extract.tmix import extract_pcap_file
from netcond.types import preset_conditions

SAMPLE_HTTP = "https://wiki.wireshark.org/uploads/__moin_import__/attachments/SampleCaptures/http.cap"


def download(url: str, dest: Path, timeout: int = 30) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return dest.exists() and dest.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001 — network optional
        print(f"download failed ({exc}); use a local pcap or data/samples/")
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=SAMPLE_HTTP)
    p.add_argument("--pcap", default="data/samples/public_http.cap")
    p.add_argument("--out", default="data/samples/public_extract.json")
    args = p.parse_args()
    path = Path(args.pcap)
    if not path.exists():
        ok = download(args.url, path)
        if not ok:
            raise SystemExit("no pcap available")
    tr = extract_pcap_file(str(path), conditions=preset_conditions("wan"), require_complete=True, app_kind="public")
    stats = tr.extract
    print(
        f"connections={stats.n_connections} complete={stats.n_complete} "
        f"incomplete={stats.n_incomplete} drop_rate={stats.drop_rate:.3f} "
        f"concurrent={stats.n_concurrent} epochs={stats.n_epochs}"
    )
    print("Passive extract: fidelity only. Do not claim do(c) counterfactuals.")
    Path(args.out).write_text(json.dumps(tr.without_packets().to_dict(), indent=2))


if __name__ == "__main__":
    main()
