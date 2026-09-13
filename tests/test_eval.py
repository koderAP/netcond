import math

from netcond.eval.fidelity import fidelity_report, jsd, wasserstein_1d
from netcond.types import Epoch, Trace, preset_conditions
import numpy as np


def _trace(sizes):
    c = preset_conditions("wan")
    epochs = [
        Epoch(a=a, b=b, t=t, epoch_index=i, transfer_time_b=tb, rtt=0.04)
        for i, (a, b, t, tb) in enumerate(sizes)
    ]
    return Trace(epochs=epochs, conditions=c, app_kind="http_get")


def test_identical_traces_distance_near_zero():
    tr = _trace([(100, 1000, 0.2, 0.05), (120, 2000, 0.3, 0.08), (90, 1500, 0.0, 0.06)])
    report = fidelity_report([tr, tr], [tr, tr])
    assert report["a"]["emd"] < 1e-9
    assert report["b"]["emd"] < 1e-9
    assert report["t_think"]["jsd"] < 1e-9
    assert report["t_think"]["emd"] < 1e-9


def test_jsd_identical_hist():
    p = np.array([0.2, 0.5, 0.3])
    assert jsd(p, p) < 1e-12


def test_wasserstein_shift():
    x = np.array([0.0, 1.0, 2.0])
    y = x + 3.0
    assert abs(wasserstein_1d(x, y) - 3.0) < 1e-6
    assert math.isfinite(wasserstein_1d(x, y))
