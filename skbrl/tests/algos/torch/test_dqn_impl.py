import pytest

from skbrl.algos.torch.dqn_impl import DQNImpl, DoubleDQNImpl
from skbrl.tests.algos.algo_test import torch_impl_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [2.5e-4])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('eps', [0.95])
@pytest.mark.parametrize('use_batch_norm', [True, False])
def test_dqn_impl(observation_shape, action_size, learning_rate, gamma, eps,
                  use_batch_norm):
    impl = DQNImpl(observation_shape,
                   action_size,
                   learning_rate,
                   gamma,
                   eps,
                   use_batch_norm,
                   use_gpu=False)
    torch_impl_tester(impl, discrete=True)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [2.5e-4])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('eps', [0.95])
@pytest.mark.parametrize('use_batch_norm', [True, False])
def test_double_dqn_impl(observation_shape, action_size, learning_rate, gamma,
                         eps, use_batch_norm):
    impl = DoubleDQNImpl(observation_shape,
                         action_size,
                         learning_rate,
                         gamma,
                         eps,
                         use_batch_norm,
                         use_gpu=False)
    torch_impl_tester(impl, discrete=True)
