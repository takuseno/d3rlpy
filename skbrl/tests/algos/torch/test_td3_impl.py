import pytest

from skbrl.algos.torch.td3_impl import TD3Impl
from skbrl.tests.algos.algo_test import torch_impl_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('actor_learning_rate', [1e-3])
@pytest.mark.parametrize('critic_learning_rate', [1e-3])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('tau', [0.05])
@pytest.mark.parametrize('reguralizing_rate', [0.0])
@pytest.mark.parametrize('n_critics', [2])
@pytest.mark.parametrize('target_smoothing_sigma', [0.2])
@pytest.mark.parametrize('target_smoothing_clip', [0.5])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('distribution_type', [None, 'qr', 'iqn'])
def test_ddpg_impl(observation_shape, action_size, actor_learning_rate,
                   critic_learning_rate, gamma, tau, reguralizing_rate,
                   n_critics, target_smoothing_sigma, target_smoothing_clip,
                   eps, use_batch_norm, distribution_type):
    impl = TD3Impl(observation_shape,
                   action_size,
                   actor_learning_rate,
                   critic_learning_rate,
                   gamma,
                   tau,
                   reguralizing_rate,
                   n_critics,
                   target_smoothing_sigma,
                   target_smoothing_clip,
                   eps,
                   use_batch_norm,
                   distribution_type=distribution_type,
                   use_gpu=False)
    torch_impl_tester(impl,
                      discrete=False,
                      deterministic_best_action=distribution_type != 'iqn')
