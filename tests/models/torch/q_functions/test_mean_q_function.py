import numpy as np
import pytest
import torch

from d3rlpy.models.torch import (
    ContinuousMeanQFunction,
    ContinuousMeanQFunctionForwarder,
    DiscreteMeanQFunction,
    DiscreteMeanQFunctionForwarder,
)
from d3rlpy.types import NDArray, Shape

from ....testing_utils import create_torch_observations
from ..model_test import (
    DummyEncoder,
    DummyEncoderWithAction,
    check_parameter_updates,
    ref_huber_loss,
)


def filter_by_action(
    value: NDArray, action: NDArray, action_size: int
) -> NDArray:
    act_one_hot = np.identity(action_size)[np.reshape(action, (-1,))]
    return (value * act_one_hot).sum(axis=1)  # type: ignore


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
def test_discrete_mean_q_function(
    observation_shape: Shape, action_size: int, batch_size: int
) -> None:
    encoder = DummyEncoder(observation_shape)
    q_func = DiscreteMeanQFunction(
        encoder, encoder.get_feature_size(), action_size
    )

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    y = q_func(x)
    assert y.q_value.shape == (batch_size, action_size)
    assert y.quantiles is None
    assert y.taus is None

    # check layer connection
    check_parameter_updates(q_func, (x,))


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("gamma", [0.99])
def test_discrete_mean_q_function_forwarder(
    observation_shape: Shape, action_size: int, batch_size: int, gamma: float
) -> None:
    encoder = DummyEncoder(observation_shape)
    q_func = DiscreteMeanQFunction(
        encoder, encoder.get_feature_size(), action_size
    )
    forwarder = DiscreteMeanQFunctionForwarder(q_func, action_size)

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    y = forwarder.compute_expected_q(x)
    assert y.shape == (batch_size, action_size)

    # check compute_target
    action = torch.randint(high=action_size, size=(batch_size,))
    target = forwarder.compute_target(x, action)
    assert target.shape == (batch_size, 1)
    assert torch.allclose(y[torch.arange(batch_size), action], target.view(-1))

    # check compute_target with action=None
    targets = forwarder.compute_target(x)
    assert targets.shape == (batch_size, action_size)
    assert (y == targets).all()

    # check td calculation
    q_tp1 = np.random.random((batch_size, 1))
    rew_tp1 = np.random.random((batch_size, 1))
    ter_tp1 = np.random.randint(2, size=(batch_size, 1))
    target = rew_tp1 + gamma * q_tp1 * (1 - ter_tp1)

    obs_t = create_torch_observations(observation_shape, batch_size)
    act_t = np.random.randint(action_size, size=(batch_size, 1))
    q_t = filter_by_action(
        q_func(obs_t).q_value.detach().numpy(), act_t, action_size
    )
    ref_loss = ref_huber_loss(q_t.reshape((-1, 1)), target)

    act_t = torch.tensor(act_t, dtype=torch.int64)
    rew_tp1 = torch.tensor(rew_tp1, dtype=torch.float32)
    q_tp1 = torch.tensor(q_tp1, dtype=torch.float32)
    ter_tp1 = torch.tensor(ter_tp1, dtype=torch.float32)
    loss = forwarder.compute_error(
        observations=obs_t,
        actions=act_t,
        rewards=rew_tp1,
        target=q_tp1,
        terminals=ter_tp1,
        gamma=gamma,
    )
    assert np.allclose(loss.detach().numpy(), ref_loss)


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
def test_continuous_mean_q_function(
    observation_shape: Shape,
    action_size: int,
    batch_size: int,
) -> None:
    encoder = DummyEncoderWithAction(observation_shape, action_size)
    q_func = ContinuousMeanQFunction(encoder, encoder.get_feature_size())

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.q_value.shape == (batch_size, 1)
    assert y.quantiles is None
    assert y.taus is None

    # check layer connection
    check_parameter_updates(q_func, (x, action))


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("gamma", [0.99])
def test_continuous_mean_q_function_forwarder(
    observation_shape: Shape,
    action_size: int,
    batch_size: int,
    gamma: float,
) -> None:
    encoder = DummyEncoderWithAction(observation_shape, action_size)
    q_func = ContinuousMeanQFunction(encoder, encoder.get_feature_size())
    forwarder = ContinuousMeanQFunctionForwarder(q_func)

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    action = torch.rand(batch_size, action_size)
    y = forwarder.compute_expected_q(x, action)
    assert y.shape == (batch_size, 1)

    # check compute_target
    target = forwarder.compute_target(x, action)
    assert target.shape == (batch_size, 1)
    assert (target == y).all()

    # check td calculation
    q_tp1 = np.random.random((batch_size, 1))
    rew_tp1 = np.random.random((batch_size, 1))
    ter_tp1 = np.random.randint(2, size=(batch_size, 1))
    target = rew_tp1 + gamma * q_tp1 * (1 - ter_tp1)

    obs_t = create_torch_observations(observation_shape, batch_size)
    act_t = torch.rand(batch_size, action_size)
    q_t = q_func(obs_t, act_t).q_value.detach().numpy()
    ref_loss = ((q_t - target) ** 2).mean()

    rew_tp1 = torch.tensor(rew_tp1, dtype=torch.float32)
    q_tp1 = torch.tensor(q_tp1, dtype=torch.float32)
    ter_tp1 = torch.tensor(ter_tp1, dtype=torch.float32)
    loss = forwarder.compute_error(
        observations=obs_t,
        actions=act_t,
        rewards=rew_tp1,
        target=q_tp1,
        terminals=ter_tp1,
        gamma=gamma,
    )

    assert np.allclose(loss.detach().numpy(), ref_loss)
