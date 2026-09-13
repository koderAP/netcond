# context.md — netcond

Network-conditioned marked TPP + flow network, trained on do(c). Independent of old NetGen.

```
u, h, z  →  ConditionedMarkedTPP.step  →  epoch (a,b,t)
epoch, c, z →  FlowNetwork.step         →  z', transfer, RTT
z' → next app step if adaptive; zeros if --unconditioned (TempoNet-style ablation)
epochs + c → DumbbellRealizer → PCAP
```

Entry: `python scripts/run_slice.py --device auto`
Prithvi: `bash scripts/prithvi_setup.sh && python scripts/run_slice.py --epochs 60 --n-per-preset 16 --device cuda`
