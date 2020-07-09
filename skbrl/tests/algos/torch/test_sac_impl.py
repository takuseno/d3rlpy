import pytest

from skbrl.algos.torch.sac_impl import SACImpl
from skbrl.tests.algos.algo_test import torch_impl_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('actor_learning_rate', [1e-3])
@pytest.mark.parametrize('critic_learning_rate', [1e-3])
@pytest.mark.parametrize('temp_learning_rate', [1e-3])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('tau', [0.05])
@pytest.mark.parametrize('n_critics', [2])
@pytest.mark.parametrize('initial_temperature', [1.0])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn'])
def test_sac_impl(observation_shape, action_size, actor_learning_rate,
                  critic_learning_rate, temp_learning_rate, gamma, tau,
                  n_critics, initial_temperature, eps, use_batch_norm,
                  q_func_type):
    impl = SACImpl(observation_shape,
                   action_size,
                   actor_learning_rate,
                   critic_learning_rate,
                   temp_learning_rate,
                   gamma,
                   tau,
                   n_critics,
                   initial_temperature,
                   eps,
                   use_batch_norm,
                   q_func_type,
                   use_gpu=False)
    torch_impl_tester(impl,
                      discrete=False,
                      deterministic_best_action=q_func_type != 'iqn')
