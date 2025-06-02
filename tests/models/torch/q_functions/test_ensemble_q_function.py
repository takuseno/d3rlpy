import pytest
import torch

from d3rlpy.models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    ContinuousIQNQFunction,
    ContinuousIQNQFunctionForwarder,
    ContinuousMeanQFunction,
    ContinuousMeanQFunctionForwarder,
    ContinuousQFunctionForwarder,
    ContinuousQRQFunction,
    ContinuousQRQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
    DiscreteIQNQFunction,
    DiscreteIQNQFunctionForwarder,
    DiscreteMeanQFunction,
    DiscreteMeanQFunctionForwarder,
    DiscreteQFunctionForwarder,
    DiscreteQRQFunction,
    DiscreteQRQFunctionForwarder,
    ImplicitQuantileTargetOutput,
    QuantileTargetOutput,
    TargetOutput,
)
from d3rlpy.models.torch.q_functions.ensemble_q_function import (
    _reduce_ensemble,
    _reduce_quantile_ensemble,
)
from d3rlpy.types import Shape

from ....testing_utils import create_torch_observations
from ..model_test import DummyEncoder, DummyEncoderWithAction


@pytest.mark.parametrize("n_ensembles", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("reduction", ["min", "max", "mean", "none"])
def test_reduce_ensemble(
    n_ensembles: int, batch_size: int, reduction: str
) -> None:
    y = torch.rand(n_ensembles, batch_size, 1)
    ret = _reduce_ensemble(y, reduction)
    if reduction == "min":
        assert ret.shape == (batch_size, 1)
        assert torch.allclose(ret, y.min(dim=0).values)
    elif reduction == "max":
        assert ret.shape == (batch_size, 1)
        assert torch.allclose(ret, y.max(dim=0).values)
    elif reduction == "mean":
        assert ret.shape == (batch_size, 1)
        assert torch.allclose(ret, y.mean(dim=0))
    elif reduction == "none":
        assert ret.shape == (n_ensembles, batch_size, 1)
        assert (ret == y).all()


@pytest.mark.parametrize("n_ensembles", [2])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("reduction", ["min", "max"])
def test_reduce_quantile_ensemble(
    n_ensembles: int,
    n_quantiles: int,
    batch_size: int,
    reduction: str,
) -> None:
    y = torch.rand(n_ensembles, batch_size, n_quantiles)
    mean = y.mean(dim=2)
    ret = _reduce_quantile_ensemble(y, mean, reduction)
    if reduction == "min":
        assert ret.shape == (batch_size, n_quantiles)
        indices = mean.min(dim=0).indices
        assert torch.allclose(ret, y[indices, torch.arange(batch_size)])
    elif reduction == "max":
        assert ret.shape == (batch_size, n_quantiles)
        indices = mean.max(dim=0).indices
        assert torch.allclose(ret, y[indices, torch.arange(batch_size)])


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("ensemble_size", [5])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn"])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("embed_size", [64])
@pytest.mark.parametrize("use_masks", [False, True])
def test_discrete_ensemble_q_function_forwarder(
    observation_shape: Shape,
    action_size: int,
    batch_size: int,
    gamma: float,
    ensemble_size: int,
    q_func_factory: str,
    n_quantiles: int,
    embed_size: int,
    use_masks: bool,
) -> None:
    forwarders: list[DiscreteQFunctionForwarder] = []
    for _ in range(ensemble_size):
        encoder = DummyEncoder(observation_shape)
        forwarder: DiscreteQFunctionForwarder
        if q_func_factory == "mean":
            q_func = DiscreteMeanQFunction(
                encoder, encoder.get_feature_size(), action_size
            )
            forwarder = DiscreteMeanQFunctionForwarder(q_func, action_size)
        elif q_func_factory == "qr":
            q_func = DiscreteQRQFunction(
                encoder, encoder.get_feature_size(), action_size, n_quantiles
            )
            forwarder = DiscreteQRQFunctionForwarder(q_func, n_quantiles)
        elif q_func_factory == "iqn":
            q_func = DiscreteIQNQFunction(
                encoder,
                encoder.get_feature_size(),
                action_size,
                n_quantiles,
                n_quantiles,
                embed_size,
            )
            forwarder = DiscreteIQNQFunctionForwarder(q_func, n_quantiles)
        else:
            raise ValueError
        forwarders.append(forwarder)
    ensemble_forwarder = DiscreteEnsembleQFunctionForwarder(
        forwarders, action_size
    )

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    values = ensemble_forwarder.compute_expected_q(x, "none")
    assert values.shape == (ensemble_size, batch_size, action_size)

    # check compute_target
    action = torch.randint(high=action_size, size=(batch_size,))
    target = ensemble_forwarder.compute_target(x, action)
    if q_func_factory == "mean":
        assert target.q_value.shape == (batch_size, 1)
        min_values = values.min(dim=0).values
        assert torch.allclose(
            min_values[torch.arange(batch_size), action],
            target.q_value.view(-1),
        )
    else:
        assert isinstance(
            target, (QuantileTargetOutput, ImplicitQuantileTargetOutput)
        )
        assert target.quantile.shape == (batch_size, n_quantiles)

    # check compute_target with action=None
    targets = ensemble_forwarder.compute_target(x)
    if q_func_factory == "mean":
        assert targets.q_value.shape == (batch_size, action_size)
    else:
        assert isinstance(
            target, (QuantileTargetOutput, ImplicitQuantileTargetOutput)
        )
        assert targets.quantile.shape == (batch_size, action_size, n_quantiles)

    # check reductions
    if q_func_factory != "iqn":
        assert torch.allclose(
            values.min(dim=0).values,
            ensemble_forwarder.compute_expected_q(x, "min"),
        )
        assert torch.allclose(
            values.max(dim=0).values,
            ensemble_forwarder.compute_expected_q(x, "max"),
        )
        assert torch.allclose(
            values.mean(dim=0), ensemble_forwarder.compute_expected_q(x, "mean")
        )

    # check td computation
    obs_t = create_torch_observations(observation_shape, batch_size)
    act_t = torch.randint(
        0, action_size, size=(batch_size, 1), dtype=torch.int64
    )
    rew_tp1 = torch.rand(batch_size, 1)
    ter_tp1 = torch.randint(2, size=(batch_size, 1))
    if q_func_factory == "mean":
        target = TargetOutput(torch.rand(batch_size, 1))
    elif q_func_factory == "qr":
        target = QuantileTargetOutput(
            q_value=torch.rand(batch_size, 1),
            quantile=torch.rand(batch_size, n_quantiles),
        )
    elif q_func_factory == "iqn":
        target = ImplicitQuantileTargetOutput(
            q_value=torch.rand(batch_size, 1),
            quantile=torch.rand(batch_size, n_quantiles),
            taus=torch.rand(batch_size, n_quantiles),
        )
    else:
        raise ValueError("invalid type")
    masks = (
        torch.randint(2, size=(batch_size, 1), dtype=torch.float32)
        if use_masks
        else None
    )
    ref_td_sum = 0.0
    for forwarder in forwarders:
        ind_loss = forwarder.compute_error(
            observations=obs_t,
            actions=act_t,
            rewards=rew_tp1,
            target=target,
            terminals=ter_tp1,
            gamma=gamma,
            reduction="none",
        )
        if masks is not None:
            ind_loss *= masks
        ref_td_sum += ind_loss.mean()
    loss = ensemble_forwarder.compute_error(
        observations=obs_t,
        actions=act_t,
        rewards=rew_tp1,
        target=target,
        terminals=ter_tp1,
        gamma=gamma,
        masks=masks,
    )
    if q_func_factory != "iqn":
        assert torch.allclose(ref_td_sum, loss)


@pytest.mark.parametrize("observation_shape", [(100,), ((100,), (200,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("ensemble_size", [5])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn"])
@pytest.mark.parametrize("embed_size", [64])
@pytest.mark.parametrize("use_masks", [False, True])
def test_ensemble_continuous_q_function(
    observation_shape: Shape,
    action_size: int,
    batch_size: int,
    gamma: float,
    ensemble_size: int,
    q_func_factory: str,
    n_quantiles: int,
    embed_size: int,
    use_masks: bool,
) -> None:
    forwarders: list[ContinuousQFunctionForwarder] = []
    for _ in range(ensemble_size):
        forwarder: ContinuousQFunctionForwarder
        encoder = DummyEncoderWithAction(observation_shape, action_size)
        if q_func_factory == "mean":
            q_func = ContinuousMeanQFunction(
                encoder, encoder.get_feature_size()
            )
            forwarder = ContinuousMeanQFunctionForwarder(q_func)
        elif q_func_factory == "qr":
            q_func = ContinuousQRQFunction(
                encoder, encoder.get_feature_size(), n_quantiles
            )
            forwarder = ContinuousQRQFunctionForwarder(q_func, n_quantiles)
        elif q_func_factory == "iqn":
            q_func = ContinuousIQNQFunction(
                encoder,
                encoder.get_feature_size(),
                n_quantiles,
                n_quantiles,
                embed_size,
            )
            forwarder = ContinuousIQNQFunctionForwarder(q_func, n_quantiles)
        else:
            raise ValueError
        forwarders.append(forwarder)

    ensemble_forwarder = ContinuousEnsembleQFunctionForwarder(
        forwarders, action_size
    )

    # check output shape
    x = create_torch_observations(observation_shape, batch_size)
    action = torch.rand(batch_size, action_size)
    values = ensemble_forwarder.compute_expected_q(x, action, "none")
    assert values.shape == (ensemble_size, batch_size, 1)

    # check compute_target
    target = ensemble_forwarder.compute_target(x, action)
    if q_func_factory == "mean":
        assert target.q_value.shape == (batch_size, 1)
        min_values = values.min(dim=0).values
        assert (target.q_value == min_values).all()
    else:
        assert isinstance(
            target, (QuantileTargetOutput, ImplicitQuantileTargetOutput)
        )
        assert target.quantile.shape == (batch_size, n_quantiles)

    # check reductions
    if q_func_factory != "iqn":
        assert torch.allclose(
            values.min(dim=0)[0],
            ensemble_forwarder.compute_expected_q(x, action, "min"),
        )
        assert torch.allclose(
            values.max(dim=0)[0],
            ensemble_forwarder.compute_expected_q(x, action, "max"),
        )
        assert torch.allclose(
            values.mean(dim=0),
            ensemble_forwarder.compute_expected_q(x, action, "mean"),
        )

    # check td computation
    obs_t = create_torch_observations(observation_shape, batch_size)
    act_t = torch.rand(batch_size, action_size)
    rew_tp1 = torch.rand(batch_size, 1)
    ter_tp1 = torch.randint(2, size=(batch_size, 1))
    if q_func_factory == "mean":
        target = TargetOutput(torch.rand(batch_size, 1))
    elif q_func_factory == "qr":
        target = QuantileTargetOutput(
            q_value=torch.rand(batch_size, 1),
            quantile=torch.rand(batch_size, n_quantiles),
        )
    elif q_func_factory == "iqn":
        target = ImplicitQuantileTargetOutput(
            q_value=torch.rand(batch_size, 1),
            quantile=torch.rand(batch_size, n_quantiles),
            taus=torch.rand(batch_size, n_quantiles),
        )
    else:
        raise ValueError("invalid type")
    masks = (
        torch.randint(2, size=(batch_size, 1), dtype=torch.float32)
        if use_masks
        else None
    )
    ref_td_sum = 0.0
    for forwarder in forwarders:
        ind_loss = forwarder.compute_error(
            observations=obs_t,
            actions=act_t,
            rewards=rew_tp1,
            target=target,
            terminals=ter_tp1,
            gamma=gamma,
            reduction="none",
        )
        if masks is not None:
            ind_loss = ind_loss * masks
        ref_td_sum += ind_loss.mean()
    loss = ensemble_forwarder.compute_error(
        observations=obs_t,
        actions=act_t,
        rewards=rew_tp1,
        target=target,
        terminals=ter_tp1,
        gamma=gamma,
        masks=masks,
    )
    if q_func_factory != "iqn":
        assert torch.allclose(ref_td_sum, loss)
