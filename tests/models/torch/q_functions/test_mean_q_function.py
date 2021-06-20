import numpy as np
import pytest
import torch

from d3rlpy.models.torch import ContinuousMeanQFunction, DiscreteMeanQFunction

from ..model_test import DummyEncoder, check_parameter_updates, ref_huber_loss


def filter_by_action(value, action, action_size):
    act_one_hot = np.identity(action_size)[np.reshape(action, (-1,))]
    return (value * act_one_hot).sum(axis=1)


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("gamma", [0.99])
def test_discrete_mean_q_function(feature_size, action_size, batch_size, gamma):
    encoder = DummyEncoder(feature_size)
    q_func = DiscreteMeanQFunction(encoder, action_size)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)

    # check compute_target
    action = torch.randint(high=action_size, size=(batch_size,))
    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, 1)
    assert torch.allclose(y[torch.arange(batch_size), action], target.view(-1))

    # check compute_target with action=None
    targets = q_func.compute_target(x)
    assert targets.shape == (batch_size, action_size)

    # check td calculation
    q_tp1 = np.random.random((batch_size, 1))
    rew_tp1 = np.random.random((batch_size, 1))
    ter_tp1 = np.random.randint(2, size=(batch_size, 1))
    target = rew_tp1 + gamma * q_tp1 * (1 - ter_tp1)

    obs_t = torch.rand(batch_size, feature_size)
    act_t = np.random.randint(action_size, size=(batch_size, 1))
    q_t = filter_by_action(q_func(obs_t).detach().numpy(), act_t, action_size)
    ref_loss = ref_huber_loss(q_t.reshape((-1, 1)), target)

    act_t = torch.tensor(act_t, dtype=torch.int64)
    rew_tp1 = torch.tensor(rew_tp1, dtype=torch.float32)
    q_tp1 = torch.tensor(q_tp1, dtype=torch.float32)
    ter_tp1 = torch.tensor(ter_tp1, dtype=torch.float32)
    loss = q_func.compute_error(
        obs_t, act_t, rew_tp1, q_tp1, ter_tp1, gamma=gamma
    )

    assert np.allclose(loss.detach().numpy(), ref_loss)

    # check layer connection
    check_parameter_updates(q_func, (obs_t, act_t, rew_tp1, q_tp1, ter_tp1))


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("gamma", [0.99])
def test_continuous_mean_q_function(
    feature_size, action_size, batch_size, gamma
):
    encoder = DummyEncoder(feature_size, action_size, concat=True)
    q_func = ContinuousMeanQFunction(encoder)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)

    # check compute_target
    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, 1)
    assert (target == y).all()

    # check td calculation
    q_tp1 = np.random.random((batch_size, 1))
    rew_tp1 = np.random.random((batch_size, 1))
    ter_tp1 = np.random.randint(2, size=(batch_size, 1))
    target = rew_tp1 + gamma * q_tp1 * (1 - ter_tp1)

    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.rand(batch_size, action_size)
    q_t = q_func(obs_t, act_t).detach().numpy()
    ref_loss = ((q_t - target) ** 2).mean()

    rew_tp1 = torch.tensor(rew_tp1, dtype=torch.float32)
    q_tp1 = torch.tensor(q_tp1, dtype=torch.float32)
    ter_tp1 = torch.tensor(ter_tp1, dtype=torch.float32)
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1, ter_tp1, gamma)

    assert np.allclose(loss.detach().numpy(), ref_loss)

    # check layer connection
    check_parameter_updates(q_func, (obs_t, act_t, rew_tp1, q_tp1, ter_tp1))
