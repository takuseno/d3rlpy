import pytest
import torch

from d3rlpy.encoders import DefaultEncoderFactory
from d3rlpy.models.torch.utility import create_deterministic_policy
from d3rlpy.models.torch.utility import create_deterministic_residual_policy
from d3rlpy.models.torch.utility import create_normal_policy
from d3rlpy.models.torch.utility import create_categorical_policy
from d3rlpy.models.torch.utility import create_discrete_q_function
from d3rlpy.models.torch.utility import create_continuous_q_function
from d3rlpy.models.torch.policies import DeterministicPolicy
from d3rlpy.models.torch.policies import DeterministicResidualPolicy
from d3rlpy.models.torch.policies import NormalPolicy
from d3rlpy.models.torch.policies import CategoricalPolicy
from d3rlpy.models.torch.q_functions import EnsembleDiscreteQFunction
from d3rlpy.models.torch.q_functions import EnsembleContinuousQFunction


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('encoder_factory', [DefaultEncoderFactory()])
def test_create_deterministic_policy(observation_shape, action_size,
                                     batch_size, encoder_factory):
    policy = create_deterministic_policy(observation_shape, action_size,
                                         encoder_factory)

    assert isinstance(policy, DeterministicPolicy)

    x = torch.rand((batch_size, ) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('scale', [0.05])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('encoder_factory', [DefaultEncoderFactory()])
def test_create_deterministic_residual_policy(observation_shape, action_size,
                                              scale, batch_size,
                                              encoder_factory):
    policy = create_deterministic_residual_policy(observation_shape,
                                                  action_size, scale,
                                                  encoder_factory)

    assert isinstance(policy, DeterministicResidualPolicy)

    x = torch.rand((batch_size, ) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = policy(x, action)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('encoder_factory', [DefaultEncoderFactory()])
def test_create_normal_policy(observation_shape, action_size, batch_size,
                              encoder_factory):
    policy = create_normal_policy(observation_shape, action_size,
                                  encoder_factory)

    assert isinstance(policy, NormalPolicy)

    x = torch.rand((batch_size, ) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('encoder_factory', [DefaultEncoderFactory()])
def test_create_categorical_policy(observation_shape, action_size, batch_size,
                                   encoder_factory):
    policy = create_categorical_policy(observation_shape, action_size,
                                       encoder_factory)

    assert isinstance(policy, CategoricalPolicy)

    x = torch.rand((batch_size, ) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size, )


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('n_ensembles', [1, 5])
@pytest.mark.parametrize('encoder_factory', [DefaultEncoderFactory()])
@pytest.mark.parametrize('q_func_factory', [MeanQFunctionFactory()])
@pytest.mark.parametrize('share_encoder', [False, True])
def test_create_discrete_q_function(observation_shape, action_size, batch_size,
                                    n_ensembles, encoder_factory,
                                    q_func_factory, share_encoder):
    q_func = create_discrete_q_function(observation_shape,
                                        action_size,
                                        encoder_factory,
                                        q_func_factory,
                                        n_ensembles,
                                        share_encoder=share_encoder)

    assert isinstance(q_func, EnsembleDiscreteQFunction)

    # check share_encoder
    encoder = q_func.q_funcs[0].encoder
    for q_func in q_func.q_funcs[1:]:
        if share_encoder:
            assert encoder is q_func.encoder
        else:
            assert encoder is not q_func.encoder

    x = torch.rand((batch_size, ) + observation_shape)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('n_ensembles', [1, 2])
@pytest.mark.parametrize('encoder_factory', [DefaultEncoderFactory()])
@pytest.mark.parametrize('q_func_factory', [MeanQFunctionFactory()])
@pytest.mark.parametrize('share_encoder', [False, True])
def test_create_continuous_q_function(observation_shape, action_size,
                                      batch_size, n_ensembles, encoder_factory,
                                      q_func_factory, share_encoder):
    q_func = create_continuous_q_function(observation_shape,
                                          action_size,
                                          encoder_factory,
                                          q_func_factory,
                                          n_ensembles,
                                          share_encoder=share_encoder)

    assert isinstance(q_func, EnsembleContinuousQFunction)

    # check share_encoder
    encoder = q_func.q_funcs[0].encoder
    for q_func in q_func.q_funcs[1:]:
        if share_encoder:
            assert encoder is q_func.encoder
        else:
            assert encoder is not q_func.encoder

    x = torch.rand((batch_size, ) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)
