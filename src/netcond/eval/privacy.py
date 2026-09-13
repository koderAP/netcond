"""Privacy stub: exact epoch 5-tuple overlap with train. Synthetic ≠ private."""

from __future__ import annotations

from netcond.types import Trace


def _key(tr: Trace, e_i: int) -> tuple:
    e = tr.epochs[e_i]
    return (e.a, e.b, round(e.t, 4), tr.app_kind)


def ngram_overlap(train: list[Trace], syn: list[Trace], n: int = 3) -> dict:
    def grams(traces: list[Trace]) -> set[tuple]:
        s: set[tuple] = set()
        for tr in traces:
            keys = [_key(tr, i) for i in range(len(tr.epochs))]
            for i in range(len(keys) - n + 1):
                s.add(tuple(keys[i : i + n]))
        return s

    t = grams(train)
    g = grams(syn)
    if not g:
        return {"n": n, "overlap_frac": 0.0, "n_syn_ngrams": 0, "n_train_ngrams": len(t)}
    hit = len(t & g)
    return {
        "n": n,
        "overlap_frac": hit / len(g),
        "n_syn_ngrams": len(g),
        "n_train_ngrams": len(t),
        "exact_hits": hit,
        "note": "Synthetic does not imply private. This is a stub n-gram overlap, not a MIA.",
    }
