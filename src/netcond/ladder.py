"""DASH bitrate ladder (shared by collector, TPP mark head, eval)."""

from __future__ import annotations

import math

BITRATE_LADDER_BPS = (200_000, 500_000, 1_000_000, 2_500_000, 5_000_000)
CHUNK_DURATION_S = 2.0
LADDER_BYTES = tuple(int(r * CHUNK_DURATION_S / 8.0) for r in BITRATE_LADDER_BPS)
N_LADDER = len(LADDER_BYTES)


def nearest_ladder_index(nbytes: float) -> int:
    logb = math.log(max(nbytes, 1.0))
    best, best_d = 0, 1e18
    for i, sz in enumerate(LADDER_BYTES):
        d = abs(math.log(sz) - logb)
        if d < best_d:
            best, best_d = i, d
    return best
