from typing import List

import pytest
import torch

from d3rlpy.models.torch import (
    ContinuousFQFQFunction,
    ContinuousIQNQFunction,
    ContinuousMeanQFunction,
    ContinuousQFunction,
    ContinuousQRQFunction,
    DiscreteFQFQFunction,
    DiscreteIQNQFunction,
    DiscreteMeanQFunction,
    DiscreteQFunction,
    DiscreteQRQFunction,
    EnsembleContinuousQFunction,
    EnsembleDiscreteQFunction,
)
from d3rlpy.models.torch.q_functions.ensemble_q_function import (
    _reduce_ensemble,
    _reduce_quantile_ensemble,
)

from ..model_test import (
    DummyEncoder,
    DummyEncoderWithAction,
    check_parameter_updates,
)


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
    ret = _reduce_quantile_ensemble(y, reduction)
    mean = y.mean(dim=2)
    if reduction == "min":
        assert ret.shape == (batch_size, n_quantiles)
        indices = mean.min(dim=0).indices
        assert torch.allclose(ret, y[indices, torch.arange(batch_size)])
    elif reduction == "max":
        assert ret.shape == (batch_size, n_quantiles)
        indices = mean.max(dim=0).indices
        assert torch.allclose(ret, y[indices, torch.arange(batch_size)])


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("ensemble_size", [5])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("embed_size", [64])
def test_ensemble_discrete_q_function(
    feature_size: int,
    action_size: int,
    batch_size: int,
    gamma: float,
    ensemble_size: int,
    q_func_factory: str,
    n_quantiles: int,
    embed_size: int,
) -> None:
    q_funcs: List[DiscreteQFunction] = []
    for _ in range(ensemble_size):
        encoder = DummyEncoder(feature_size)
        if q_func_factory == "mean":
            q_func = DiscreteMeanQFunction(encoder, action_size)
        elif q_func_factory == "qr":
            q_func = DiscreteQRQFunction(encoder, action_size, n_quantiles)
        elif q_func_factory == "iqn":
            q_func = DiscreteIQNQFunction(
                encoder, action_size, n_quantiles, n_quantiles, embed_size
            )
        elif q_func_factory == "fqf":
            q_func = DiscreteFQFQFunction(
                encoder, action_size, n_quantiles, embed_size
            )
        q_funcs.append(q_func)
    q_func = EnsembleDiscreteQFunction(q_funcs)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    values = q_func(x, "none")
    assert values.shape == (ensemble_size, batch_size, action_size)

    # check compute_target
    action = torch.randint(high=action_size, size=(batch_size,))
    target = q_func.compute_target(x, action)
    if q_func_factory == "mean":
        assert target.shape == (batch_size, 1)
        min_values = values.min(dim=0).values
        assert torch.allclose(
            min_values[torch.arange(batch_size), action], target.view(-1)
        )
    else:
        assert target.shape == (batch_size, n_quantiles)

    # check compute_target with action=None
    targets = q_func.compute_target(x)
    if q_func_factory == "mean":
        assert targets.shape == (batch_size, action_size)
    else:
        assert targets.shape == (batch_size, action_size, n_quantiles)

    # check reductions
    if q_func_factory != "iqn":
        assert torch.allclose(values.min(dim=0).values, q_func(x, "min"))
        assert torch.allclose(values.max(dim=0).values, q_func(x, "max"))
        assert torch.allclose(values.mean(dim=0), q_func(x, "mean"))

    # check td computation
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.randint(
        0, action_size, size=(batch_size, 1), dtype=torch.int64
    )
    rew_tp1 = torch.rand(batch_size, 1)
    ter_tp1 = torch.randint(2, size=(batch_size, 1))
    if q_func_factory == "mean":
        q_tp1 = torch.rand(batch_size, 1)
    else:
        q_tp1 = torch.rand(batch_size, n_quantiles)
    ref_td_sum = 0.0
    for i in range(ensemble_size):
        f = q_func.q_funcs[i]
        ref_td_sum += f.compute_error(
            obs_t, act_t, rew_tp1, q_tp1, ter_tp1, gamma
        )
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1, ter_tp1, gamma)
    if q_func_factory != "iqn":
        assert torch.allclose(ref_td_sum, loss)

    # check layer connection
    check_parameter_updates(
        q_func,
        (obs_t, act_t, rew_tp1, q_tp1, ter_tp1, gamma),
    )


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("ensemble_size", [5])
@pytest.mark.parametrize("n_quantiles", [200])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("embed_size", [64])
def test_ensemble_continuous_q_function(
    feature_size: int,
    action_size: int,
    batch_size: int,
    gamma: float,
    ensemble_size: int,
    q_func_factory: str,
    n_quantiles: int,
    embed_size: int,
) -> None:
    q_funcs: List[ContinuousQFunction] = []
    for _ in range(ensemble_size):
        encoder = DummyEncoderWithAction(feature_size, action_size)
        if q_func_factory == "mean":
            q_func = ContinuousMeanQFunction(encoder)
        elif q_func_factory == "qr":
            q_func = ContinuousQRQFunction(encoder, n_quantiles)
        elif q_func_factory == "iqn":
            q_func = ContinuousIQNQFunction(
                encoder, n_quantiles, n_quantiles, embed_size
            )
        elif q_func_factory == "fqf":
            q_func = ContinuousFQFQFunction(encoder, n_quantiles, embed_size)
        q_funcs.append(q_func)

    q_func = EnsembleContinuousQFunction(q_funcs)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    values = q_func(x, action, "none")
    assert values.shape == (ensemble_size, batch_size, 1)

    # check compute_target
    target = q_func.compute_target(x, action)
    if q_func_factory == "mean":
        assert target.shape == (batch_size, 1)
        min_values = values.min(dim=0).values
        assert (target == min_values).all()
    else:
        assert target.shape == (batch_size, n_quantiles)

    # check reductions
    if q_func_factory != "iqn":
        assert torch.allclose(values.min(dim=0)[0], q_func(x, action, "min"))
        assert torch.allclose(values.max(dim=0)[0], q_func(x, action, "max"))
        assert torch.allclose(values.mean(dim=0), q_func(x, action, "mean"))

    # check td computation
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.rand(batch_size, action_size)
    rew_tp1 = torch.rand(batch_size, 1)
    ter_tp1 = torch.randint(2, size=(batch_size, 1))
    if q_func_factory == "mean":
        q_tp1 = torch.rand(batch_size, 1)
    else:
        q_tp1 = torch.rand(batch_size, n_quantiles)
    ref_td_sum = 0.0
    for i in range(ensemble_size):
        f = q_func.q_funcs[i]
        ref_td_sum += f.compute_error(
            obs_t, act_t, rew_tp1, q_tp1, ter_tp1, gamma
        )
    loss = q_func.compute_error(obs_t, act_t, rew_tp1, q_tp1, ter_tp1, gamma)
    if q_func_factory != "iqn":
        assert torch.allclose(ref_td_sum, loss)

    # check layer connection
    check_parameter_updates(
        q_func,
        (obs_t, act_t, rew_tp1, q_tp1, ter_tp1, gamma),
    )
