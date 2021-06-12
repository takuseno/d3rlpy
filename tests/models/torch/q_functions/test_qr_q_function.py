import numpy as np
import pytest
import torch

from d3rlpy.models.torch.q_functions import (
    ContinuousQRQFunction,
    DiscreteQRQFunction,
)
from d3rlpy.models.torch.q_functions.qr_q_function import _make_taus
from d3rlpy.models.torch.q_functions.utility import (
    pick_quantile_value_by_action,
)

from ..model_test import (
    DummyEncoder,
    check_parameter_updates,
    ref_quantile_huber_loss,
)


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("gamma", [0.99])
def test_discrete_qr_q_function(
    feature_size, action_size, n_quantiles, batch_size, gamma
):
    encoder = DummyEncoder(feature_size)
    q_func = DiscreteQRQFunction(encoder, action_size, n_quantiles)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)

    # check taus
    taus = _make_taus(encoder(x), n_quantiles)
    step = 1 / n_quantiles
    for i in range(n_quantiles):
        assert np.allclose(taus[0][i].numpy(), i * step + step / 2.0)

    # check compute_target
    action = torch.randint(high=action_size, size=(batch_size,))
    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, n_quantiles)

    # check compute_target with action=None
    targets = q_func.compute_target(x)
    assert targets.shape == (batch_size, action_size, n_quantiles)

    # check quantile huber loss
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.randint(action_size, size=(batch_size,))
    rew_tp1 = torch.rand(batch_size, 1)
    q_tp1 = torch.rand(batch_size, n_quantiles)
    ter_tp1 = torch.randint(2, size=(batch_size, 1))
    # shape check
    loss = q_func.compute_error(
        obs_t, act_t, rew_tp1, q_tp1, ter_tp1, reduction="none"
    )
    assert loss.shape == (batch_size, 1)
    # mean loss
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1, ter_tp1)

    target = rew_tp1.numpy() + gamma * q_tp1.numpy() * (1 - ter_tp1.numpy())
    y = pick_quantile_value_by_action(
        q_func._compute_quantiles(encoder(obs_t), taus), act_t
    )

    reshaped_target = np.reshape(target, (batch_size, -1, 1))
    reshaped_y = np.reshape(y.detach().numpy(), (batch_size, 1, -1))
    reshaped_taus = np.reshape(taus, (1, 1, -1))

    ref_loss = ref_quantile_huber_loss(
        reshaped_y, reshaped_target, reshaped_taus, n_quantiles
    )
    assert np.allclose(loss.cpu().detach(), ref_loss.mean())

    # check layer connection
    check_parameter_updates(q_func, (obs_t, act_t, rew_tp1, q_tp1, ter_tp1))


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("gamma", [0.99])
def test_continuous_qr_q_function(
    feature_size, action_size, n_quantiles, batch_size, gamma
):
    encoder = DummyEncoder(feature_size, action_size, concat=True)
    q_func = ContinuousQRQFunction(encoder, n_quantiles)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)

    # check taus
    taus = _make_taus(encoder(x, action), n_quantiles)
    step = 1 / n_quantiles
    for i in range(n_quantiles):
        assert np.allclose(taus[0][i].numpy(), i * step + step / 2.0)

    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, n_quantiles)

    # check quantile huber loss
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.rand(batch_size, action_size)
    rew_tp1 = torch.rand(batch_size, 1)
    q_tp1 = torch.rand(batch_size, n_quantiles)
    ter_tp1 = torch.randint(2, size=(batch_size, 1))
    # check shape
    loss = q_func.compute_error(
        obs_t, act_t, rew_tp1, q_tp1, ter_tp1, reduction="none"
    )
    assert loss.shape == (batch_size, 1)
    # mean loss
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1, ter_tp1)

    target = rew_tp1.numpy() + gamma * q_tp1.numpy() * (1 - ter_tp1.numpy())
    y = q_func._compute_quantiles(encoder(obs_t, act_t), taus).detach().numpy()

    reshaped_target = target.reshape((batch_size, -1, 1))
    reshaped_y = y.reshape((batch_size, 1, -1))
    reshaped_taus = taus.reshape((1, 1, -1))

    ref_loss = ref_quantile_huber_loss(
        reshaped_y, reshaped_target, reshaped_taus, n_quantiles
    )
    assert np.allclose(loss.cpu().detach(), ref_loss.mean())

    # check layer connection
    check_parameter_updates(q_func, (obs_t, act_t, rew_tp1, q_tp1, ter_tp1))
