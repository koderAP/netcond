# AGENTS.md — netcond (COD892)

Read `docs/DESIGN.md` first. The old NetGen SSM+GRU loop is obsolete.

## Mission

Network-conditioned marked TPP + flow network head, trained on **interventional** `do(c)` data. Exogenous a-b-t vs endogenous transfer/RTT. Single coupling point: `couple/loop.py`.

## Layout

| Task | Files |
|------|--------|
| Types | `src/netcond/types.py` |
| Tmix extractor | `src/netcond/extract/tmix.py` |
| TPP | `src/netcond/app/tpp.py` |
| Network head | `src/netcond/net/flow_state.py` |
| Coupling | `src/netcond/couple/loop.py` |
| Realizer | `src/netcond/realize/emulator.py` |
| Interventional data | `src/netcond/data/intervene.py` |
| Train / generate | `src/netcond/train.py`, `generate.py` |
| Eval | `src/netcond/eval/` |

## Constraints

- `app.feedback_dim == net.latent_dim` (CoupledLoop)
- Think time ≠ mixed IAT ≠ transfer time
- No size-bucket message types
- No PCAP-heuristic 8-d observables as `c`
- No stub MSS realizer as the story
- Do not claim causal CF from passive traces
- `pytest -q` after changes under `src/netcond/`
