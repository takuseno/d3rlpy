import pytest
import torch

from d3rlpy.models.torch import (
    ContinuousIQNQFunction,
    ContinuousIQNQFunctionForwarder,
    DiscreteIQNQFunction,
    DiscreteIQNQFunctionForwarder,
)

from ..model_test import (
    DummyEncoder,
    DummyEncoderWithAction,
    check_parameter_updates,
)


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("n_greedy_quantiles", [32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("embed_size", [64])
def test_discrete_iqn_q_function(
    feature_size: int,
    action_size: int,
    n_quantiles: int,
    n_greedy_quantiles: int,
    batch_size: int,
    embed_size: int,
) -> None:
    encoder = DummyEncoder(feature_size)
    q_func = DiscreteIQNQFunction(
        encoder,
        feature_size,
        action_size,
        n_quantiles,
        n_greedy_quantiles,
        embed_size,
    )

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.q_value.shape == (batch_size, action_size)
    assert y.quantiles is not None
    assert y.taus is not None
    assert y.quantiles.shape == (batch_size, action_size, n_quantiles)
    assert y.taus.shape == (batch_size, n_quantiles)
    assert (y.q_value == y.quantiles.mean(dim=2)).all()

    # check eval mode
    q_func.eval()
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.q_value.shape == (batch_size, action_size)
    assert y.quantiles is not None
    assert y.taus is not None
    assert y.quantiles.shape == (batch_size, action_size, n_greedy_quantiles)
    assert y.taus.shape == (batch_size, n_greedy_quantiles)
    q_func.train()

    # check layer connection
    check_parameter_updates(q_func, (x,))


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("n_greedy_quantiles", [32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("embed_size", [64])
def test_discrete_iqn_q_function_forwarder(
    feature_size: int,
    action_size: int,
    n_quantiles: int,
    n_greedy_quantiles: int,
    batch_size: int,
    embed_size: int,
) -> None:
    encoder = DummyEncoder(feature_size)
    q_func = DiscreteIQNQFunction(
        encoder,
        feature_size,
        action_size,
        n_quantiles,
        n_greedy_quantiles,
        embed_size,
    )
    forwarder = DiscreteIQNQFunctionForwarder(q_func, n_quantiles)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = forwarder.compute_expected_q(x)
    assert y.shape == (batch_size, action_size)

    # check eval mode
    q_func.eval()
    x = torch.rand(batch_size, feature_size)
    y = forwarder.compute_expected_q(x)
    assert y.shape == (batch_size, action_size)
    q_func.train()

    # check compute_target
    action = torch.randint(high=action_size, size=(batch_size,))
    target = forwarder.compute_target(x, action)
    assert target.shape == (batch_size, n_quantiles)

    # check compute_target with action=None
    targets = forwarder.compute_target(x)
    assert targets.shape == (batch_size, action_size, n_quantiles)

    # TODO: check quantile huber loss
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.randint(action_size, size=(batch_size,))
    rew_tp1 = torch.rand(batch_size, 1)
    q_tp1 = torch.rand(batch_size, n_quantiles)
    ter_tp1 = torch.randint(2, size=(batch_size, 1))
    # check shape
    loss = forwarder.compute_error(
        observations=obs_t,
        actions=act_t,
        rewards=rew_tp1,
        target=q_tp1,
        terminals=ter_tp1,
        reduction="none",
    )
    assert loss.shape == (batch_size, 1)
    # mean loss
    loss = forwarder.compute_error(
        observations=obs_t,
        actions=act_t,
        rewards=rew_tp1,
        target=q_tp1,
        terminals=ter_tp1,
    )


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("n_greedy_quantiles", [32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("embed_size", [64])
def test_continuous_iqn_q_function(
    feature_size: int,
    action_size: int,
    n_quantiles: int,
    n_greedy_quantiles: int,
    batch_size: int,
    embed_size: int,
) -> None:
    encoder = DummyEncoderWithAction(feature_size, action_size)
    q_func = ContinuousIQNQFunction(
        encoder,
        feature_size,
        n_quantiles,
        n_greedy_quantiles,
        embed_size,
    )

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.q_value.shape == (batch_size, 1)
    assert y.quantiles is not None
    assert y.taus is not None
    assert y.quantiles.shape == (batch_size, n_quantiles)
    assert y.taus.shape == (batch_size, n_quantiles)
    assert (y.q_value == y.quantiles.mean(dim=1, keepdim=True)).all()

    # check eval mode
    q_func.eval()
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.q_value.shape == (batch_size, 1)
    assert y.quantiles is not None
    assert y.taus is not None
    assert y.quantiles.shape == (batch_size, n_greedy_quantiles)
    assert y.taus.shape == (batch_size, n_greedy_quantiles)
    q_func.train()

    # check layer connection
    check_parameter_updates(q_func, (x, action))


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("n_greedy_quantiles", [32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("embed_size", [64])
def test_continuous_iqn_q_function_forwarder(
    feature_size: int,
    action_size: int,
    n_quantiles: int,
    n_greedy_quantiles: int,
    batch_size: int,
    embed_size: int,
) -> None:
    encoder = DummyEncoderWithAction(feature_size, action_size)
    q_func = ContinuousIQNQFunction(
        encoder,
        feature_size,
        n_quantiles,
        n_greedy_quantiles,
        embed_size,
    )
    forwarder = ContinuousIQNQFunctionForwarder(q_func, n_quantiles)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = forwarder.compute_expected_q(x, action)
    assert y.shape == (batch_size, 1)

    target = forwarder.compute_target(x, action)
    assert target.shape == (batch_size, n_quantiles)

    # TODO: check quantile huber loss
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.rand(batch_size, action_size)
    rew_tp1 = torch.rand(batch_size, 1)
    q_tp1 = torch.rand(batch_size, n_quantiles)
    ter_tp1 = torch.randint(2, size=(batch_size, 1))
    # check shape
    loss = forwarder.compute_error(
        observations=obs_t,
        actions=act_t,
        rewards=rew_tp1,
        target=q_tp1,
        terminals=ter_tp1,
        reduction="none",
    )
    assert loss.shape == (batch_size, 1)
    # mean loss
    loss = forwarder.compute_error(
        observations=obs_t,
        actions=act_t,
        rewards=rew_tp1,
        target=q_tp1,
        terminals=ter_tp1,
    )
