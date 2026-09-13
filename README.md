# netcond

A **network-conditioned** generative model of application epochs and packet timing, trained on **interventional** data (`P(traffic | do(c))`) so named-condition “what-if” traces are **causally valid**.

Headline use: generate DASH-like / chunked-download traces under a WAN you specify; counterfactual “same session, 2× RTT”; rare-regime data for ABR / QoE models.

This repository **replaces** the old NetGen coupled-SSM design. It is **not** a Mamba traffic generator and **not** a claim to replace NetSSM on raw bytes.

## What is novel (and what is not)

Ancestry (classical closed-loop, not the claim): **Tmix** (Weigle et al., CCR 2006), **Swing** (Vishwanath & Vahdat, ToN 2009), Floyd & Paxson 2001 (source-level, closed-loop).

SOTA we do **not** replace on packet bytes: **NetSSM** (Chu et al., CoNEXT 2026 — timestamps sampled from a real PCAP; their §6 asks for a topology/conditions modality), NetDiffusion, TrafficGPT.

SOTA we extend:

- **TempoNet** — add controllable physical network conditions `c` / latent `z`.
- **NetReplica** — add a *generative* reactive application model instead of replaying cross traffic.
- **CausalSim** — train on `do(c)` instead of naive `P(traffic | observed network)` from passive traces.

| System | Content | Time | Conditions | Generative | Causal CF |
|---|---|---|---|---|---|
| NetSSM / NetDiffusion | bytes | no* | no | yes | no |
| TempoNet | headers | yes | no | yes | no |
| Tmix / Swing / NetReplica | replay | via stack | yes | no | partial |
| m4 / DeepQueueNet | n/a | performance | yes | no | n/a |
| **netcond** | ADU a-b-t | yes | yes | yes | yes (interventional) |

\*NetSSM samples IAT from ground truth.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python examples/minimal_slice.py
python scripts/run_slice.py --epochs 18 --n-per-preset 8
```

## Prithvi A40 (3 seeds, mean ± std)

Trained on `prithvi.cse.iitd.ac.in` (NVIDIA A40). **Run 3** is the headline: DASH bitrate ladder conditioned on `z`, think-time head without `z` (Tmix), slow-start/cross-traffic transfer decoder, stratified `do(c)` split.

**Run 3** (`output/prithvi3`, 80 epochs, 24 traces/preset/app):

| metric | neural | uncond TPP | tmix-resample |
|---|---|---|---|
| EMD log response `b` | **0.081 ± 0.062** | 1.084 ± 0.447 | 0.899 ± 0.109 |
| EMD log think `t` | 0.247 ± 0.079 | — | **0.059 ± 0.022** |
| EMD log transfer | **0.345 ± 0.118** | — | 0.797 ± 0.031 |
| Q–Q MAE log `b` | **0.074 ± 0.051** | — | — |
| TSTR condition-id | **1.00 ± 0.00** | — | TRTR 1.00 |
| 2× base RTT | **3/3 True** (ratio 1.6–1.9) | — | — |
| mean chunk `b` lan vs congested | **1.25e6 vs 1.30e5** (3/3) | — | — |

Conditioning on `do(c)` beats Tmix resampling **and** a TempoNet-style unconditioned TPP on the adaptive mark `b` and on transfer time. TSTR matches train-real-test-real. Think time remains a resample problem (exogenous). Congested transfer is still a bit short vs the emulator (~2–4s vs ~4.6s).

Run 2 (continuous log-normal `b`): neural lost to resample on `b` (1.55 vs 0.98). Run 1 used a contiguous split (TSTR invalid).

`run_slice.py` collects interventional traces, trains 3 seeds plus an unconditioned TPP, and prints the table. This repo does not modify `netgen_cod892`.

### Prithvi (IITD CSE A40)

```bash
# from a laptop with SSH host `prithvi`
rsync -az --exclude .venv --exclude output . prithvi:~/netcond/
ssh prithvi 'cd ~/netcond && bash scripts/prithvi_setup.sh'
# results: ~/netcond/output/prithvi/eval_table.txt
```

### Collect / train / generate / eval separately

```bash
python scripts/collect_intervene.py --out data/samples/intervene.json --n-per-preset 8
python scripts/train.py --data data/samples/intervene.json --out checkpoints/netcond.pt --epochs 25
python scripts/generate.py --ckpt checkpoints/netcond.pt --preset congested --out output/gen.json
python scripts/evaluate.py --data data/samples/intervene.json --ckpts checkpoints/netcond.pt --out output/eval.json
```

### Public (passive) PCAPs

```bash
python scripts/ingest_public_pcap.py --pcap path/to/file.pcap
```

Extractor is Tmix seq/ack, not IAT-burst grouping. Incomplete connections are counted (`drop_rate`); concurrent connections are split into two unidirectional streams. **Passive extracts are fidelity-only** — do not claim counterfactuals from MAWI without CausalSim-style adjustment.

## Representation

- **Exogenous epoch:** `a` (request bytes), `b` (response bytes), `t` (**think time**). Think time is not request→response latency (Tmix).
- **Endogenous:** ADU transfer time, RTT, loss, packet timestamps (network head + realizer).
- **Conditions `c`:** capacity, base RTT, buffer, AQM, cross-traffic id, loss. Presets: `lan`, `wan`, `congested`, `lossy`, `satellite`.

Feedback `z_t →` next app step is **justified for adaptive apps** (DASH-like chunk bitrate). For HTTP GET, a-b-t is approximately network-invariant; the loop still conditions the TPP on `z` but detaches feedback in the joint phase for those sessions.

## Realizer / environment

v1 uses a **userspace dumbbell TCP** (windowed sender + buffer drops + capacity/RTT). Linux `tc netem` and ns-3 Full-TCP are the intended production realizers; they are not required to train this slice. macOS cannot run `tc netem`. Packets are wire-valid IPv4/TCP PCAPs, not the old MSS stub.

## Evaluation (not KS pass-rate)

- Fidelity: JSD + EMD on `a`, `b`, think-time `t` (not mixed IAT).
- Transfer time / RTT vs interventional holdout.
- Baseline: Tmix-style empirical resample of train a-b-t **and** unconditioned TPP (no `z`).
- Utility: train-synthetic-test-real condition-id classifier (TSTR vs TRTR).
- Counterfactual: generate under `congested` vs holdout `do(congested)`; 2× base RTT must increase mean RTT.
- ≥3 seeds, mean ± std. Never a single best seed.
- Wavelet logscale: skipped when sessions are shorter than tens of seconds (documented); Q–Q of `t` and transfer times still run.
- Privacy stub: n-gram overlap. **Synthetic ≠ private.**

## Blunder checklist

- [x] No “first SSM traffic generator”
- [x] No counterfactuals claimed from passive traces without CausalSim caveat
- [x] No evaluation on only marginal KS of mixed IAT
- [x] No single-seed headline
- [x] No size buckets as app types
- [x] Incomplete / concurrent connections documented, not silently dropped
- [x] Synthetic ≠ private

## Citations

- Weigle, Adurthi, Hernández-Campos, Jeffay, Smith. *Tmix.* CCR 36(3), 2006. DOI 10.1145/1140086.1140094
- Vishwanath & Vahdat. *Swing.* IEEE/ACM ToN 17(3), 2009. DOI 10.1109/TNET.2009.2020830
- Shchur, Biloš, Günnemann. *Intensity-free TPP.* ICLR 2019
- Alomar et al. *CausalSim.* NSDI 2023
- Chu et al. *NetSSM.* CoNEXT 2026. https://github.com/noise-lab/netssm
- Jiang et al. *NetDiffusion.* SIGMETRICS 2024
- Yin et al. *NetShare.* SIGCOMM 2022
- TempoNet. arXiv:2601.15663
- NetReplica / NetForge. arXiv:2507.13476
- m4. arXiv:2503.01770
- Survey: arXiv:2507.01976
- TraceBleed. arXiv:2508.11742

Design notes: [`docs/DESIGN.md`](docs/DESIGN.md).
