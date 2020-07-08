import pytest

from skbrl.algos.torch.ddpg_impl import DDPGImpl
from skbrl.tests.algos.algo_test import torch_impl_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('actor_learning_rate', [1e-3])
@pytest.mark.parametrize('critic_learning_rate', [1e-3])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('tau', [0.05])
@pytest.mark.parametrize('reguralizing_rate', [1e-8])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('distribution_type', [None, 'qr', 'iqn'])
def test_ddpg_impl(observation_shape, action_size, actor_learning_rate,
                   critic_learning_rate, gamma, tau, reguralizing_rate, eps,
                   use_batch_norm, distribution_type):
    impl = DDPGImpl(observation_shape,
                    action_size,
                    actor_learning_rate,
                    critic_learning_rate,
                    gamma,
                    tau,
                    reguralizing_rate,
                    eps,
                    use_batch_norm,
                    distribution_type=distribution_type,
                    use_gpu=False)
    torch_impl_tester(impl,
                      discrete=False,
                      deterministic_best_action=distribution_type != 'iqn')
