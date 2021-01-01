import pytest
import torch

from d3rlpy.encoders import DefaultEncoderFactory
from d3rlpy.models.torch.dynamics import _compute_ensemble_variance
from d3rlpy.models.torch.dynamics import ProbablisticDynamics
from d3rlpy.models.torch.dynamics import EnsembleDynamics
from .model_test import check_parameter_updates, DummyEncoder


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("n_ensembles", [5])
@pytest.mark.parametrize("variance_type", ["max", "data"])
def test_compute_ensemble_variance(
    batch_size, observation_shape, n_ensembles, variance_type
):
    observations = torch.rand((batch_size, n_ensembles) + observation_shape)
    rewards = torch.rand(batch_size, n_ensembles, 1)
    variances = torch.rand(batch_size, n_ensembles, 1)

    if variance_type == "max":
        ref = variances.max(dim=1).values
    elif variance_type == "data":
        data = torch.cat([observations, rewards], dim=2)
        ref = (data.std(dim=1) ** 2).sum(dim=1, keepdims=True)

    variances = _compute_ensemble_variance(
        observations, rewards, variances, variance_type
    )

    assert variances.shape == (batch_size, 1)
    assert torch.allclose(variances, ref)


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
def test_probablistic_dynamics(feature_size, action_size, batch_size):
    encoder = DummyEncoder(feature_size, action_size, True)
    dynamics = ProbablisticDynamics(encoder)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    pred_x, pred_reward = dynamics(x, action)
    assert pred_x.shape == (batch_size, feature_size)
    assert pred_reward.shape == (batch_size, 1)

    # check variance
    _, _, variance = dynamics.predict_with_variance(x, action)
    assert variance.shape == (batch_size, 1)

    # TODO: check error
    reward = torch.rand(batch_size, 1)
    loss = dynamics.compute_error(x, action, reward, x)
    assert loss.shape == (batch_size, 1)

    # check layer connection
    check_parameter_updates(dynamics, (x, action, reward, x))


@pytest.mark.parametrize("feature_size", [100])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n_ensembles", [5])
def test_ensemble_dynamics_dynamics(
    feature_size, action_size, batch_size, n_ensembles
):
    encoder = DummyEncoder(feature_size, action_size, True)
    models = []
    for _ in range(n_ensembles):
        models.append(ProbablisticDynamics(encoder))

    dynamics = EnsembleDynamics(models)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    pred_x, pred_reward = dynamics(x, action)
    assert pred_x.shape == (batch_size, feature_size)
    assert pred_reward.shape == (batch_size, 1)

    # check variance
    _, _, variances = dynamics.predict_with_variance(x, action)
    assert variances.shape == (batch_size, 1)

    # TODO: check error
    reward = torch.rand(batch_size, 1)
    loss = dynamics.compute_error(x, action, reward, x)

    # check layer connection
    check_parameter_updates(dynamics, (x, action, reward, x))
