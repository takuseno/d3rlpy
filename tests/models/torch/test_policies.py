import pytest
import torch
import copy

from d3rlpy.models.torch.policies import create_deterministic_policy
from d3rlpy.models.torch.policies import create_deterministic_residual_policy
from d3rlpy.models.torch.policies import create_normal_policy
from d3rlpy.models.torch.policies import create_categorical_policy
from d3rlpy.models.torch.policies import DeterministicPolicy
from d3rlpy.models.torch.policies import DeterministicResidualPolicy
from d3rlpy.models.torch.policies import NormalPolicy
from d3rlpy.models.torch.policies import CategoricalPolicy
from .model_test import create_encoder_factory
from .model_test import check_parameter_updates, DummyEncoder


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_encoder_factory', [False, True])
def test_create_deterministic_policy(observation_shape, action_size,
                                     batch_size, use_encoder_factory):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)

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
@pytest.mark.parametrize('use_encoder_factory', [False, True])
def test_create_deterministic_residual_policy(observation_shape, action_size,
                                              scale, batch_size,
                                              use_encoder_factory):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)

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
@pytest.mark.parametrize('use_encoder_factory', [False, True])
def test_create_normal_policy(observation_shape, action_size, batch_size,
                              use_encoder_factory):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)

    policy = create_normal_policy(observation_shape, action_size,
                                  encoder_factory)

    assert isinstance(policy, NormalPolicy)

    x = torch.rand((batch_size, ) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_encoder_factory', [False, True])
def test_create_categorical_policy(observation_shape, action_size, batch_size,
                                   use_encoder_factory):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)

    policy = create_categorical_policy(observation_shape, action_size,
                                       encoder_factory)

    assert isinstance(policy, CategoricalPolicy)

    x = torch.rand((batch_size, ) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size, )


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
def test_deterministic_policy(feature_size, action_size, batch_size):
    encoder = DummyEncoder(feature_size)
    policy = DeterministicPolicy(encoder, action_size)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = policy(x)
    assert y.shape == (batch_size, action_size)

    # check best action
    best_action = policy.best_action(x)
    assert torch.allclose(best_action, y)

    # check layer connection
    check_parameter_updates(policy, (x, ))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('scale', [0.05])
@pytest.mark.parametrize('batch_size', [32])
def test_deterministic_residual_policy(feature_size, action_size, scale,
                                       batch_size):
    encoder = DummyEncoder(feature_size, action_size)
    policy = DeterministicResidualPolicy(encoder, scale)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = policy(x, action)
    assert y.shape == (batch_size, action_size)

    # check residual
    assert not (y == action).any()
    assert ((y - action).abs() <= scale).all()

    # check best action
    best_action = policy.best_action(x, action)
    assert torch.allclose(best_action, y)

    # check layer connection
    check_parameter_updates(policy, (x, action))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('min_logstd', [-20.0])
@pytest.mark.parametrize('max_logstd', [2.0])
@pytest.mark.parametrize('use_std_parameter', [True, False])
@pytest.mark.parametrize('n', [10])
def test_normal_policy(feature_size, action_size, batch_size, min_logstd,
                       max_logstd, use_std_parameter, n):
    encoder = DummyEncoder(feature_size)
    policy = NormalPolicy(encoder, action_size, min_logstd, max_logstd,
                          use_std_parameter)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = policy(x)
    assert y.shape == (batch_size, action_size)

    # check distribution type
    assert isinstance(policy.dist(x), torch.distributions.Normal)

    # check if sampled action is not identical to the best action
    assert not torch.allclose(policy.sample(x), policy.best_action(x))

    # check sample_n
    y_n, log_prob_n = policy.sample_n(x, n, True)
    assert y_n.shape == (batch_size, n, action_size)
    assert log_prob_n.shape == (batch_size, n, 1)

    # check layer connection
    check_parameter_updates(policy, (x, ))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('n', [10])
def test_categorical_policy(feature_size, action_size, batch_size, n):
    encoder = DummyEncoder(feature_size)
    policy = CategoricalPolicy(encoder, action_size)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = policy(x)
    assert y.shape == (batch_size, )

    # check log_prob shape
    _, log_prob = policy(x, with_log_prob=True)
    assert log_prob.shape == (batch_size, )

    # check distribution type
    assert isinstance(policy.dist(x), torch.distributions.Categorical)

    # check if sampled action is not identical to the bset action
    assert not torch.all(policy.sample(x) == policy.best_action(x))

    # check sample_n
    y_n, log_prob_n = policy.sample_n(x, n, True)
    assert y_n.shape == (batch_size, n)
    assert log_prob_n.shape == (batch_size, n)

    # check log_probs
    log_probs = policy.log_probs(x)
    assert log_probs.shape == (batch_size, action_size)
    assert torch.allclose(log_probs.exp().sum(dim=1), torch.ones(batch_size))

    # check layer connection
    check_parameter_updates(policy, output=log_probs)
