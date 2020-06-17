import numpy as np
import pytest
import torch

from skbrl.models.torch.q_functions import DiscreteQFunction
from skbrl.models.torch.q_functions import EnsembleDiscreteQFunction
from skbrl.models.torch.q_functions import ContinuousQFunction
from skbrl.models.torch.q_functions import EnsembleContinuousQFunction
from skbrl.tests.models.torch.model_test import check_parameter_updates
from skbrl.tests.models.torch.model_test import DummyHead


def ref_huber_loss(a, b):
    abs_diff = np.abs(a - b).reshape((-1, ))
    l2_diff = ((a - b)**2).reshape((-1, ))
    huber_diff = np.zeros_like(abs_diff)
    huber_diff[abs_diff < 1.0] = 0.5 * l2_diff[abs_diff < 1.0]
    huber_diff[abs_diff >= 1.0] = abs_diff[abs_diff >= 1.0] - 0.5
    return np.mean(huber_diff)


def filter_by_action(value, action, action_size):
    act_one_hot = np.identity(action_size)[np.reshape(action, (-1, ))]
    return (value * act_one_hot).sum(axis=1)


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
def test_discrete_q_function(feature_size, action_size, batch_size, gamma):
    head = DummyHead(feature_size)
    q_func = DiscreteQFunction(head, action_size)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)

    # check td calculation
    q_tp1 = np.random.random((batch_size, 1))
    rew_tp1 = np.random.random((batch_size, 1))
    target = rew_tp1 + gamma * q_tp1

    obs_t = torch.rand(batch_size, feature_size)
    act_t = np.random.randint(action_size, size=(batch_size, 1))
    q_t = filter_by_action(q_func(obs_t).detach().numpy(), act_t, action_size)
    ref_loss = ref_huber_loss(q_t.reshape((-1, 1)), target)

    loss = q_func.compute_td(obs_t,
                             torch.tensor(act_t, dtype=torch.int64),
                             torch.tensor(rew_tp1, dtype=torch.float32),
                             torch.tensor(q_tp1, dtype=torch.float32),
                             gamma=gamma)

    assert np.allclose(loss.detach().numpy(), ref_loss)

    # check layer connection
    check_parameter_updates(q_func, (x, ))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
def test_continuous_q_function(feature_size, action_size, batch_size, gamma):
    head = DummyHead(feature_size + action_size)
    q_func = ContinuousQFunction(head)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)

    # check td calculation
    q_tp1 = np.random.random((batch_size, 1))
    rew_tp1 = np.random.random((batch_size, 1))
    target = rew_tp1 + gamma * q_tp1

    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.rand(batch_size, action_size)
    q_t = q_func(obs_t, act_t).detach().numpy()
    ref_loss = ((q_t - target)**2).mean()

    loss = q_func.compute_td(obs_t, act_t,
                             torch.tensor(rew_tp1, dtype=torch.float32),
                             torch.tensor(q_tp1, dtype=torch.float32), gamma)

    assert np.allclose(loss.detach().numpy(), ref_loss)

    # check layer connection
    check_parameter_updates(q_func, (x, action))
