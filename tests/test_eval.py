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


def test_tmix_replay_does_not_shift_dash_b(tmp_path):
    from netcond.data.intervene import collect_dataset
    from netcond.eval.baseline_resample import tmix_replay_counterfactual
    from netcond.types import preset_conditions

    traces = collect_dataset(
        tmp_path / "cf.json",
        n_per_preset=3,
        presets=("lan", "congested"),
        apps=("dash_like",),
        seed=4,
    )
    replay = tmix_replay_counterfactual(
        traces,
        from_preset="lan",
        to_conditions=preset_conditions("congested"),
        app_kind="dash_like",
        n_sessions=4,
        n_steps=8,
        seed=0,
    )
    lan_b = sum(e.b for tr in traces if tr.conditions.preset == "lan" for e in tr.epochs) / max(
        1, sum(len(tr.epochs) for tr in traces if tr.conditions.preset == "lan")
    )
    replay_b = sum(e.b for tr in replay for e in tr.epochs) / max(1, sum(len(tr.epochs) for tr in replay))
    cong_b = sum(e.b for tr in traces if tr.conditions.preset == "congested" for e in tr.epochs) / max(
        1, sum(len(tr.epochs) for tr in traces if tr.conditions.preset == "congested")
    )
    assert replay_b > 0.5 * lan_b
    assert replay_b > cong_b


def test_wasserstein_shift():
    x = np.array([0.0, 1.0, 2.0])
    y = x + 3.0
    assert abs(wasserstein_1d(x, y) - 3.0) < 1e-6
    assert math.isfinite(wasserstein_1d(x, y))
