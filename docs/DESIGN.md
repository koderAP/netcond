# netcond — design (v1 slice)

Research claim: a generative model of **application behaviour and packet timing** that is **conditioned on physically meaningful, controllable network conditions**, trained on **interventional** data so “what-if” traces are **causally valid**, and checked against **real interventional ground truth**.

This is **not** the old NetGen coupled SSM/Mamba + 8-d PCAP-heuristic loop.

## Representation

Exogenous (application / user) — Tmix connection vectors:

- `a` request ADU bytes, `b` response ADU bytes
- `t` **think time** (response-complete → next request). Not request→response latency.

Endogenous (network / realizer): transfer time, RTT, loss, packet timestamps.

Physical conditions `c` follow NetReplica’s split: static envelope (capacity, base RTT, buffer, AQM) + dynamic pressure (cross-traffic id / ON-OFF). Named presets: `lan`, `wan`, `congested`, `lossy`, `satellite`.

Do not store mixed bidirectional IAT. That conflates think time with transfer time.

## Modules

```
Application (marked TPP, GRU backbone)
  inputs: app context u, network latent z_t
  outputs: next exogenous epoch (a, b, t, marks)

Network (flow-level head)
  inputs: physical c, emitted ADUs
  outputs: z_{t+1}, transfer time, RTT, loss
  trained on emulator labels — not PCAP heuristics

Realizer
  userspace dumbbell TCP (capacity / RTT / buffer / loss / window)
  optional later: ns-3 Full-TCP or Linux tc netem

Orchestrator (only coupling point)
  app.step → net.step → z' into next app step
  Feedback is used for adaptive apps (DASH-like). HTTP GET a-b-t is treated as network-invariant.
```

## Causal training

Collector randomizes `c` and runs HTTP GET (invariant a-b-t) and a DASH-like chunked downloader (next `b` from last goodput). Labels are `P(traffic | do(c))`.

Passive PCAPs (MAWI / public samples) go through the same Tmix extractor for **fidelity only**. Counterfactuals from passive traces would need CausalSim-style latent adjustment; this slice does not claim that.

## What we are not claiming

- Not “first SSM / Mamba traffic generator” (NetSSM exists; we do not generate raw bytes).
- Not “first closed-loop generator” (Tmix, Swing).
- Not replacing NetSSM / NetDiffusion on packet bytes.
- Synthetic ≠ private (n-gram overlap stub only).

## Related work (positioning)

| System | We take | We add |
|---|---|---|
| Tmix (CCR 2006) | a-b-t, think time ≠ transfer time | learned generator + conditions |
| Swing (ToN 2009) | network conditions dominate fine time scales | generative app, not replay |
| TempoNet (2026) | marked TPP timing | condition on controllable `c` |
| NetReplica (2025) | envelope + cross-traffic | generative reactive app instead of replay |
| CausalSim (NSDI 2023) | passive traces bias counterfactuals | train on `do(c)` |
| NetSSM (CoNEXT 2026) | bytes, no learned time; they ask for a topology/conditions modality | that modality is `z`/`c` |

## Losses

`L = L_marks + L_think + L_net + λ_q L_think_quantile + λ_ac L_think_autocorr`

Phases: pretrain app (`z` detached) → pretrain net on labeled transfer/RTT → joint with feedback **only** on adaptive sessions.

## Evaluation (v1)

JSD + EMD on `a`,`b`,`t` (think times only); transfer time / RTT; Tmix resample baseline; 2×RTT direction check; ≥3 seeds mean±std; wavelet skipped on short sessions (documented); privacy n-gram stub.

## Environment note

This slice’s realizer is a **userspace dumbbell**, not ns-3. macOS has no `tc netem`. Packet headers are wire-valid IPv4/TCP. Congestion control is a simplified windowed sender, not Linux Cubic/BBR. That is a limitation, not a claim of stack fidelity.
