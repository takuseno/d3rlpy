import pytest
import torch

from skbrl.models.torch.policies import create_deterministic_policy
from skbrl.models.torch.policies import create_deterministic_residual_policy
from skbrl.models.torch.policies import create_normal_policy
from skbrl.models.torch.policies import DeterministicPolicy
from skbrl.models.torch.policies import DeterministicResidualPolicy
from skbrl.models.torch.policies import NormalPolicy
from skbrl.tests.models.torch.model_test import check_parameter_updates
from skbrl.tests.models.torch.model_test import DummyHead


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_create_deterministic_policy(observation_shape, action_size,
                                     batch_size, use_batch_norm):
    policy = create_deterministic_policy(observation_shape, action_size,
                                         use_batch_norm)

    assert isinstance(policy, DeterministicPolicy)

    x = torch.rand((batch_size, ) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('scale', [0.05])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_create_deterministic_residual_policy(observation_shape, action_size,
                                              scale, batch_size,
                                              use_batch_norm):
    policy = create_deterministic_residual_policy(observation_shape,
                                                  action_size, scale,
                                                  use_batch_norm)

    assert isinstance(policy, DeterministicResidualPolicy)

    x = torch.rand((batch_size, ) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = policy(x, action)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_create_normal_policy(observation_shape, action_size, batch_size,
                              use_batch_norm):
    policy = create_normal_policy(observation_shape, action_size,
                                  use_batch_norm)

    assert isinstance(policy, NormalPolicy)

    x = torch.rand((batch_size, ) + observation_shape)
    y = policy(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
def test_deterministic_policy(feature_size, action_size, batch_size):
    head = DummyHead(feature_size)
    policy = DeterministicPolicy(head, action_size)

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
    head = DummyHead(feature_size, action_size)
    policy = DeterministicResidualPolicy(head, scale)

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
def test_normal_policy(feature_size, action_size, batch_size):
    head = DummyHead(feature_size)
    policy = NormalPolicy(head, action_size)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = policy(x)
    assert y.shape == (batch_size, action_size)

    # check distribution type
    assert isinstance(policy.dist(x), torch.distributions.Normal)

    # check if sampled action is not identical to the best action
    assert not torch.allclose(policy.sample(x), policy.best_action(x))

    # check layer connection
    check_parameter_updates(policy, (x, ))
