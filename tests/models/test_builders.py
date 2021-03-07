import pytest
import torch
import numpy as np

from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.models.builders import create_deterministic_policy
from d3rlpy.models.builders import create_deterministic_residual_policy
from d3rlpy.models.builders import create_normal_policy
from d3rlpy.models.builders import create_categorical_policy
from d3rlpy.models.builders import create_discrete_q_function
from d3rlpy.models.builders import create_continuous_q_function
from d3rlpy.models.builders import create_conditional_vae
from d3rlpy.models.builders import create_discrete_imitator
from d3rlpy.models.builders import create_deterministic_regressor
from d3rlpy.models.builders import create_probablistic_regressor
from d3rlpy.models.builders import create_value_function
from d3rlpy.models.builders import create_probablistic_dynamics
from d3rlpy.models.builders import create_parameter
from d3rlpy.models.torch.policies import DeterministicPolicy
from d3rlpy.models.torch.policies import DeterministicResidualPolicy
from d3rlpy.models.torch.policies import NormalPolicy
from d3rlpy.models.torch.policies import CategoricalPolicy
from d3rlpy.models.torch.q_functions import EnsembleDiscreteQFunction
from d3rlpy.models.torch.q_functions import EnsembleContinuousQFunction
from d3rlpy.models.torch.imitators import ConditionalVAE
from d3rlpy.models.torch.imitators import DiscreteImitator
from d3rlpy.models.torch.imitators import DeterministicRegressor
from d3rlpy.models.torch.imitators import ProbablisticRegressor
from d3rlpy.models.torch.v_functions import ValueFunction
from d3rlpy.models.torch.dynamics import EnsembleDynamics
from d3rlpy.models.torch.parameters import Parameter


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_deterministic_policy(
    observation_shape, action_size, batch_size, encoder_factory
):
    policy = create_deterministic_policy(
        observation_shape, action_size, encoder_factory
    )

    assert isinstance(policy, DeterministicPolicy)

    x = torch.rand((batch_size,) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("scale", [0.05])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_deterministic_residual_policy(
    observation_shape, action_size, scale, batch_size, encoder_factory
):
    policy = create_deterministic_residual_policy(
        observation_shape, action_size, scale, encoder_factory
    )

    assert isinstance(policy, DeterministicResidualPolicy)

    x = torch.rand((batch_size,) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = policy(x, action)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_normal_policy(
    observation_shape, action_size, batch_size, encoder_factory
):
    policy = create_normal_policy(
        observation_shape, action_size, encoder_factory
    )

    assert isinstance(policy, NormalPolicy)

    x = torch.rand((batch_size,) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_categorical_policy(
    observation_shape, action_size, batch_size, encoder_factory
):
    policy = create_categorical_policy(
        observation_shape, action_size, encoder_factory
    )

    assert isinstance(policy, CategoricalPolicy)

    x = torch.rand((batch_size,) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size,)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n_ensembles", [1, 5])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("share_encoder", [False, True])
@pytest.mark.parametrize("bootstrap", [False, True])
def test_create_discrete_q_function(
    observation_shape,
    action_size,
    batch_size,
    n_ensembles,
    encoder_factory,
    share_encoder,
    bootstrap,
):
    q_func_factory = MeanQFunctionFactory(
        share_encoder=share_encoder, bootstrap=bootstrap
    )

    q_func = create_discrete_q_function(
        observation_shape,
        action_size,
        encoder_factory,
        q_func_factory,
        n_ensembles,
    )

    assert isinstance(q_func, EnsembleDiscreteQFunction)
    if n_ensembles == 1:
        assert q_func.bootstrap == False
    else:
        assert q_func.bootstrap == bootstrap

    # check share_encoder
    encoder = q_func.q_funcs[0].encoder
    for q_func in q_func.q_funcs[1:]:
        if share_encoder:
            assert encoder is q_func.encoder
        else:
            assert encoder is not q_func.encoder

    x = torch.rand((batch_size,) + observation_shape)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n_ensembles", [1, 2])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("share_encoder", [False, True])
@pytest.mark.parametrize("bootstrap", [False, True])
def test_create_continuous_q_function(
    observation_shape,
    action_size,
    batch_size,
    n_ensembles,
    encoder_factory,
    share_encoder,
    bootstrap,
):
    q_func_factory = MeanQFunctionFactory(
        share_encoder=share_encoder, bootstrap=bootstrap
    )

    q_func = create_continuous_q_function(
        observation_shape,
        action_size,
        encoder_factory,
        q_func_factory,
        n_ensembles,
    )

    assert isinstance(q_func, EnsembleContinuousQFunction)
    if n_ensembles == 1:
        assert q_func.bootstrap == False
    else:
        assert q_func.bootstrap == bootstrap

    # check share_encoder
    encoder = q_func.q_funcs[0].encoder
    for q_func in q_func.q_funcs[1:]:
        if share_encoder:
            assert encoder is q_func.encoder
        else:
            assert encoder is not q_func.encoder

    x = torch.rand((batch_size,) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("beta", [1.0])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_conditional_vae(
    observation_shape,
    action_size,
    latent_size,
    beta,
    batch_size,
    encoder_factory,
):
    vae = create_conditional_vae(
        observation_shape, action_size, latent_size, beta, encoder_factory
    )

    assert isinstance(vae, ConditionalVAE)

    x = torch.rand((batch_size,) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = vae(x, action)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("beta", [1e-2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_discrete_imitator(
    observation_shape, action_size, beta, batch_size, encoder_factory
):
    imitator = create_discrete_imitator(
        observation_shape, action_size, beta, encoder_factory
    )

    assert isinstance(imitator, DiscreteImitator)

    x = torch.rand((batch_size,) + observation_shape)
    y = imitator(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_deterministic_regressor(
    observation_shape, action_size, batch_size, encoder_factory
):
    imitator = create_deterministic_regressor(
        observation_shape, action_size, encoder_factory
    )

    assert isinstance(imitator, DeterministicRegressor)

    x = torch.rand((batch_size,) + observation_shape)
    y = imitator(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_probablistic_regressor(
    observation_shape, action_size, batch_size, encoder_factory
):
    imitator = create_probablistic_regressor(
        observation_shape, action_size, encoder_factory
    )

    assert isinstance(imitator, ProbablisticRegressor)

    x = torch.rand((batch_size,) + observation_shape)
    y = imitator(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("batch_size", [32])
def test_create_value_function(observation_shape, encoder_factory, batch_size):
    v_func = create_value_function(observation_shape, encoder_factory)

    assert isinstance(v_func, ValueFunction)

    x = torch.rand((batch_size,) + observation_shape)
    y = v_func(x)
    assert y.shape == (batch_size, 1)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("n_ensembles", [5])
@pytest.mark.parametrize("discrete_action", [False, True])
@pytest.mark.parametrize("batch_size", [32])
def test_create_probablistic_dynamics(
    observation_shape,
    action_size,
    encoder_factory,
    n_ensembles,
    discrete_action,
    batch_size,
):
    dynamics = create_probablistic_dynamics(
        observation_shape,
        action_size,
        encoder_factory,
        n_ensembles,
        discrete_action,
    )

    assert isinstance(dynamics, EnsembleDynamics)
    assert len(dynamics.models) == n_ensembles

    x = torch.rand((batch_size,) + observation_shape)
    if discrete_action:
        action = torch.randint(0, action_size, size=(batch_size, 1))
    else:
        action = torch.rand(batch_size, action_size)
    observation, reward = dynamics(x, action)
    assert observation.shape == (batch_size,) + observation_shape
    assert reward.shape == (batch_size, 1)


@pytest.mark.parametrize("shape", [(100,)])
def test_create_parameter(shape):
    x = np.random.random()
    parameter = create_parameter(shape, x)

    assert len(list(parameter.parameters())) == 1
    assert np.allclose(parameter().detach().numpy(), x)
