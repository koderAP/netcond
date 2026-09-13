from netcond.data.intervene import collect_dataset, dash_like_epochs, http_get_epochs
from netcond.realize.emulator import transfer_time
from netcond.types import preset_conditions
import random


def test_http_abt_approximately_invariant():
    rng = random.Random(0)
    epochs = http_get_epochs(rng, n_files=5)
    sizes = [(e.a, e.b, round(e.t, 6)) for e in epochs]
    rng2 = random.Random(0)
    again = [(e.a, e.b, round(e.t, 6)) for e in http_get_epochs(rng2, n_files=5)]
    assert sizes == again


def test_dash_shifts_under_do_c():
    lan = dash_like_epochs(preset_conditions("lan"), random.Random(1), n_chunks=6)
    sat = dash_like_epochs(preset_conditions("satellite"), random.Random(1), n_chunks=6)
    mean_b_lan = sum(e.b for e in lan) / len(lan)
    mean_b_sat = sum(e.b for e in sat) / len(sat)
    # adaptive bitrate should pick smaller chunks on satellite/high RTT / lower goodput
    assert mean_b_lan > mean_b_sat


def test_transfer_time_increases_with_rtt():
    rng = random.Random(2)
    t0, r0, _ = transfer_time(50_000, preset_conditions("wan"), rng)
    rng = random.Random(2)
    slow = preset_conditions("satellite")
    t1, r1, _ = transfer_time(50_000, slow, rng)
    assert r1 > r0
    assert t1 > t0


def test_collect_small(tmp_path):
    traces = collect_dataset(
        tmp_path / "d.json",
        n_per_preset=2,
        presets=("lan", "congested"),
        apps=("http_get", "dash_like"),
        seed=3,
    )
    assert len(traces) == 8
    assert all(tr.metadata.get("do_c") for tr in traces)
    assert all(e.transfer_time_b is not None for tr in traces for e in tr.epochs)
