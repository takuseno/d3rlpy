import pytest

from d3rlpy.algos.torch.cql_impl import CQLImpl, DiscreteCQLImpl
from tests.algos.algo_test import torch_impl_tester, DummyScaler


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('actor_learning_rate', [1e-3])
@pytest.mark.parametrize('critic_learning_rate', [1e-3])
@pytest.mark.parametrize('temp_learning_rate', [1e-3])
@pytest.mark.parametrize('alpha_learning_rate', [1e-3])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('tau', [0.05])
@pytest.mark.parametrize('n_critics', [2])
@pytest.mark.parametrize('initial_temperature', [1.0])
@pytest.mark.parametrize('initial_alpha', [5.0])
@pytest.mark.parametrize('alpha_threshold', [10.0])
@pytest.mark.parametrize('n_action_samples', [10])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
def test_cql_impl(observation_shape, action_size, actor_learning_rate,
                  critic_learning_rate, temp_learning_rate,
                  alpha_learning_rate, gamma, tau, n_critics,
                  initial_temperature, initial_alpha, alpha_threshold,
                  n_action_samples, eps, use_batch_norm, q_func_type, scaler):
    impl = CQLImpl(observation_shape,
                   action_size,
                   actor_learning_rate,
                   critic_learning_rate,
                   temp_learning_rate,
                   alpha_learning_rate,
                   gamma,
                   tau,
                   n_critics,
                   initial_temperature,
                   initial_alpha,
                   alpha_threshold,
                   n_action_samples,
                   eps,
                   use_batch_norm,
                   q_func_type,
                   use_gpu=False,
                   scaler=scaler)
    torch_impl_tester(impl,
                      discrete=False,
                      deterministic_best_action=q_func_type != 'iqn')


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [2.5e-4])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('n_critics', [1])
@pytest.mark.parametrize('eps', [0.95])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
def test_double_dqn_impl(observation_shape, action_size, learning_rate, gamma,
                         n_critics, eps, use_batch_norm, q_func_type, scaler):
    impl = DiscreteCQLImpl(observation_shape,
                           action_size,
                           learning_rate,
                           gamma,
                           n_critics,
                           eps,
                           use_batch_norm,
                           q_func_type=q_func_type,
                           use_gpu=False,
                           scaler=scaler)
    torch_impl_tester(impl,
                      discrete=True,
                      deterministic_best_action=q_func_type != 'iqn')
