"""Tiny train smoke: a few steps, generate lan vs congested."""

from netcond.data.intervene import collect_dataset
from netcond.generate import generate
from netcond.train import train_loop


def test_train_and_generate(tmp_path):
    traces = collect_dataset(
        tmp_path / "d.json",
        n_per_preset=3,
        presets=("lan", "wan", "congested"),
        apps=("dash_like",),
        seed=0,
    )
    result = train_loop(traces, epochs=6, seed=0, device="cpu")
    loop = result["loop"]
    lan = generate(loop, "lan", app_kind="dash_like", n_sessions=2, n_steps=4, seed=1)
    cong = generate(loop, "congested", app_kind="dash_like", n_sessions=2, n_steps=4, seed=1)
    assert len(lan[0].epochs) == 4
    assert lan[0].conditions.preset == "lan"
    assert cong[0].conditions.preset == "congested"
    assert all("total" in h for h in result["history"])
