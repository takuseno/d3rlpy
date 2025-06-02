# pylint: disable=protected-access
import numpy as np
import pytest
import torch

from d3rlpy.models.torch import (
    ContinuousQRQFunction,
    ContinuousQRQFunctionForwarder,
    DiscreteQRQFunction,
    DiscreteQRQFunctionForwarder,
    QuantileTargetOutput,
)
from d3rlpy.models.torch.q_functions.utility import (
    pick_quantile_value_by_action,
)
from d3rlpy.types import Shape

from ....testing_utils import create_torch_observations
from ..model_test import (
    DummyEncoder,
    DummyEncoderWithAction,
    check_parameter_updates,
    ref_quantile_huber_loss,
)


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("batch_size", [32])
def test_discrete_qr_q_function(
    observation_shape: Shape,
    action_size: int,
    n_quantiles: int,
    batch_size: int,
) -> None:
    encoder = DummyEncoder(observation_shape)
    q_func = DiscreteQRQFunction(
        encoder, encoder.get_feature_size(), action_size, n_quantiles
    )

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    y = q_func(x)
    assert y.q_value.shape == (batch_size, action_size)
    assert y.quantiles is not None and y.taus is not None
    assert y.quantiles.shape == (batch_size, action_size, n_quantiles)
    assert y.taus.shape == (1, n_quantiles)
    assert torch.allclose(y.q_value, y.quantiles.mean(dim=2))

    # check taus
    step = 1 / n_quantiles
    for i in range(n_quantiles):
        assert np.allclose(y.taus[0][i].numpy(), i * step + step / 2.0)

    # check layer connection
    check_parameter_updates(q_func, (x,))


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("gamma", [0.99])
def test_discrete_qr_q_function_forwarder(
    observation_shape: Shape,
    action_size: int,
    n_quantiles: int,
    batch_size: int,
    gamma: float,
) -> None:
    encoder = DummyEncoder(observation_shape)
    q_func = DiscreteQRQFunction(
        encoder, encoder.get_feature_size(), action_size, n_quantiles
    )
    forwarder = DiscreteQRQFunctionForwarder(q_func, n_quantiles)

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    y = forwarder.compute_expected_q(x)
    assert y.shape == (batch_size, action_size)

    # check compute_target
    action = torch.randint(high=action_size, size=(batch_size,))
    target_output = forwarder.compute_target(x, action)
    assert target_output.quantile.shape == (batch_size, n_quantiles)

    # check add
    offset = torch.rand(batch_size, 1)
    modified_target_output = target_output.add(offset)
    assert torch.all(
        modified_target_output.q_value == target_output.q_value + offset
    )
    assert torch.all(
        modified_target_output.quantile == target_output.quantile + offset
    )

    # check compute_target with action=None
    target_outputs = forwarder.compute_target(x)
    assert target_outputs.quantile.shape == (
        batch_size,
        action_size,
        n_quantiles,
    )

    # check add with action=None
    offset = torch.rand(batch_size, 1)
    modified_target_outputs = target_outputs.add(offset)
    assert torch.all(
        modified_target_outputs.q_value == target_outputs.q_value + offset
    )
    assert torch.all(
        modified_target_outputs.quantile
        == target_outputs.quantile + offset.view(batch_size, 1, 1)
    )

    # check quantile huber loss
    obs_t = create_torch_observations(observation_shape, batch_size)
    act_t = torch.randint(action_size, size=(batch_size,))
    rew_tp1 = torch.rand(batch_size, 1)
    q_tp1 = torch.rand(batch_size, n_quantiles)
    ter_tp1 = torch.randint(2, size=(batch_size, 1))
    # shape check
    loss = forwarder.compute_error(
        observations=obs_t,
        actions=act_t,
        rewards=rew_tp1,
        target=QuantileTargetOutput(q_tp1, q_tp1),
        terminals=ter_tp1,
        reduction="none",
    )
    assert loss.shape == (batch_size, 1)
    # mean loss
    loss = forwarder.compute_error(
        observations=obs_t,
        actions=act_t,
        rewards=rew_tp1,
        target=QuantileTargetOutput(q_tp1, q_tp1),
        terminals=ter_tp1,
    )

    target = rew_tp1.numpy() + gamma * q_tp1.numpy() * (1 - ter_tp1.numpy())
    y = q_func(obs_t)
    quantiles = y.quantiles
    taus = y.taus
    assert quantiles is not None
    assert taus is not None
    y = pick_quantile_value_by_action(quantiles, act_t)

    reshaped_target = np.reshape(target, (batch_size, -1, 1))
    reshaped_y = np.reshape(y.detach().numpy(), (batch_size, 1, -1))
    reshaped_taus = np.reshape(taus.detach().numpy(), (1, 1, -1))

    ref_loss = ref_quantile_huber_loss(
        reshaped_y, reshaped_target, reshaped_taus, n_quantiles
    )
    assert np.allclose(loss.cpu().detach(), ref_loss.mean())


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("batch_size", [32])
def test_continuous_qr_q_function(
    observation_shape: Shape,
    action_size: int,
    n_quantiles: int,
    batch_size: int,
) -> None:
    encoder = DummyEncoderWithAction(observation_shape, action_size)
    q_func = ContinuousQRQFunction(
        encoder, encoder.get_feature_size(), n_quantiles
    )

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.q_value.shape == (batch_size, 1)
    assert y.quantiles is not None
    assert y.quantiles.shape == (batch_size, n_quantiles)
    assert torch.allclose(y.q_value, y.quantiles.mean(dim=1, keepdim=True))
    assert y.taus is not None
    assert y.taus.shape == (1, n_quantiles)

    # check taus
    step = 1 / n_quantiles
    for i in range(n_quantiles):
        assert np.allclose(y.taus[0][i].numpy(), i * step + step / 2.0)

    # check layer connection
    check_parameter_updates(q_func, (x, action))


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("gamma", [0.99])
def test_continuous_qr_q_function_forwarder(
    observation_shape: Shape,
    action_size: int,
    n_quantiles: int,
    batch_size: int,
    gamma: float,
) -> None:
    encoder = DummyEncoderWithAction(observation_shape, action_size)
    q_func = ContinuousQRQFunction(
        encoder, encoder.get_feature_size(), n_quantiles
    )
    forwarder = ContinuousQRQFunctionForwarder(q_func, n_quantiles)

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    action = torch.rand(batch_size, action_size)
    y = forwarder.compute_expected_q(x, action)
    assert y.shape == (batch_size, 1)

    target_output = forwarder.compute_target(x, action)
    assert target_output.quantile.shape == (batch_size, n_quantiles)

    # check quantile huber loss
    obs_t = create_torch_observations(observation_shape, batch_size)
    act_t = torch.rand(batch_size, action_size)
    rew_tp1 = torch.rand(batch_size, 1)
    q_tp1 = torch.rand(batch_size, n_quantiles)
    ter_tp1 = torch.randint(2, size=(batch_size, 1))
    # check shape
    loss = forwarder.compute_error(
        observations=obs_t,
        actions=act_t,
        rewards=rew_tp1,
        target=QuantileTargetOutput(q_tp1, q_tp1),
        terminals=ter_tp1,
        reduction="none",
    )
    assert loss.shape == (batch_size, 1)
    # mean loss
    loss = forwarder.compute_error(
        observations=obs_t,
        actions=act_t,
        rewards=rew_tp1,
        target=QuantileTargetOutput(q_tp1, q_tp1),
        terminals=ter_tp1,
    )

    target = rew_tp1.numpy() + gamma * q_tp1.numpy() * (1 - ter_tp1.numpy())
    y = q_func(obs_t, act_t)
    assert y.quantiles is not None
    assert y.taus is not None
    quantiles = y.quantiles.detach().numpy()
    taus = y.taus.detach().numpy()

    reshaped_target = target.reshape((batch_size, -1, 1))
    reshaped_y = quantiles.reshape((batch_size, 1, -1))
    reshaped_taus = taus.reshape((1, 1, -1))

    ref_loss = ref_quantile_huber_loss(
        reshaped_y, reshaped_target, reshaped_taus, n_quantiles
    )
    assert np.allclose(loss.cpu().detach(), ref_loss.mean())
