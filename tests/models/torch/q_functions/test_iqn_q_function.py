import pytest
import torch

from d3rlpy.models.torch.q_functions import (
    ContinuousIQNQFunction,
    DiscreteIQNQFunction,
)

from ..model_test import DummyEncoder, check_parameter_updates


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("n_greedy_quantiles", [32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("embed_size", [64])
@pytest.mark.parametrize("gamma", [0.99])
def test_discrete_iqn_q_function(
    feature_size,
    action_size,
    n_quantiles,
    n_greedy_quantiles,
    batch_size,
    embed_size,
    gamma,
):
    encoder = DummyEncoder(feature_size)
    q_func = DiscreteIQNQFunction(
        encoder, action_size, n_quantiles, n_greedy_quantiles, embed_size
    )

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)

    # check eval mode
    q_func.eval()
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)
    q_func.train()

    # check compute_target
    action = torch.randint(high=action_size, size=(batch_size,))
    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, n_quantiles)

    # check compute_target with action=None
    targets = q_func.compute_target(x)
    assert targets.shape == (batch_size, action_size, n_quantiles)

    # TODO: check quantile huber loss
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.randint(action_size, size=(batch_size,))
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

    # check layer connection
    check_parameter_updates(q_func, (obs_t, act_t, rew_tp1, q_tp1, ter_tp1))


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("n_greedy_quantiles", [32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("embed_size", [64])
@pytest.mark.parametrize("gamma", [0.99])
def test_continuous_iqn_q_function(
    feature_size,
    action_size,
    n_quantiles,
    n_greedy_quantiles,
    batch_size,
    embed_size,
    gamma,
):
    encoder = DummyEncoder(feature_size, action_size)
    q_func = ContinuousIQNQFunction(
        encoder, n_quantiles, n_greedy_quantiles, embed_size
    )

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)

    # check eval mode
    q_func.eval()
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)
    q_func.train()

    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, n_quantiles)

    # TODO: check quantile huber loss
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.randint(action_size, size=(batch_size,))
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

    # check layer connection
    check_parameter_updates(q_func, (obs_t, act_t, rew_tp1, q_tp1, ter_tp1))
