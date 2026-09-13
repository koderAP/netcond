from netcond.couple.loop import CoupledLoop
from netcond.app.tpp import ConditionedMarkedTPP
from netcond.net.flow_state import FlowNetwork
from netcond.types import CONDITION_DIM, Conditions, preset_conditions
import pytest
import torch


def test_condition_tensor_roundtrip():
    for name in ("lan", "wan", "congested", "lossy", "satellite"):
        c = preset_conditions(name)
        t = c.as_tensor()
        assert t.shape == (CONDITION_DIM,)
        back = Conditions.from_tensor(t, aqm=c.aqm, preset=c.preset)
        assert abs(back.capacity_bps / c.capacity_bps - 1) < 1e-5
        assert abs(back.base_rtt_s / c.base_rtt_s - 1) < 1e-5
        assert abs(back.buffer_bytes / c.buffer_bytes - 1) < 1e-5
        assert abs(back.loss_rate - c.loss_rate) < 1e-5


def test_coupling_dim_mismatch():
    app = ConditionedMarkedTPP(feedback_dim=8, hidden_dim=16)
    net = FlowNetwork(latent_dim=4, hidden_dim=16)
    with pytest.raises(ValueError, match="feedback_dim"):
        CoupledLoop(app, net)


def test_tpp_sample_shapes():
    app = ConditionedMarkedTPP(feedback_dim=8, hidden_dim=16)
    net = FlowNetwork(latent_dim=8, hidden_dim=16)
    loop = CoupledLoop(app, net)
    B, T = 3, 5
    c = preset_conditions("lan").as_tensor().unsqueeze(0).expand(B, -1)
    z = net.initial_state(c, B)
    z = z.unsqueeze(1).expand(B, T, -1)
    ctx = torch.zeros(B, 4)
    out = app.sample_sequence(z, ctx, n_steps=T)
    assert out["a"].shape == (B, T)
    assert out["b"].shape == (B, T)
    assert out["t"].shape == (B, T)
    assert torch.all(out["t"][:, -1] == 0)
    tr = loop.rollout(preset_conditions("wan"), ctx[0], n_steps=4)
    assert len(tr.epochs) == 4
    assert tr.epochs[0].transfer_time_b is not None
    z0 = loop.rollout(preset_conditions("wan"), ctx[0], n_steps=4, unconditioned=True)
    assert z0.metadata["unconditioned"] is True
