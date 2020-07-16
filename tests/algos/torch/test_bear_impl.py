import pytest
import torch

from d3rlpy.algos.torch.bear_impl import BEARImpl
from tests.algos.algo_test import torch_impl_tester, DummyScaler


@pytest.mark.parametrize('observation_shape', [(100, ), (1, 48, 48)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('actor_learning_rate', [1e-3])
@pytest.mark.parametrize('critic_learning_rate', [1e-3])
@pytest.mark.parametrize('imitator_learning_rate', [1e-3])
@pytest.mark.parametrize('temp_learning_rate', [1e-3])
@pytest.mark.parametrize('alpha_learning_rate', [1e-3])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('tau', [0.05])
@pytest.mark.parametrize('n_critics', [2])
@pytest.mark.parametrize('initial_temperature', [1.0])
@pytest.mark.parametrize('initial_alpha', [1.0])
@pytest.mark.parametrize('alpha_threshold', [0.05])
@pytest.mark.parametrize('lam', [0.75])
@pytest.mark.parametrize('n_action_samples', [4])
@pytest.mark.parametrize('mmd_sigma', [20.0])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
def test_bcq_impl(observation_shape, action_size, actor_learning_rate,
                  critic_learning_rate, imitator_learning_rate,
                  temp_learning_rate, alpha_learning_rate, gamma, tau,
                  n_critics, initial_temperature, initial_alpha,
                  alpha_threshold, lam, n_action_samples, mmd_sigma, eps,
                  use_batch_norm, q_func_type, scaler):
    impl = BEARImpl(observation_shape,
                    action_size,
                    actor_learning_rate,
                    critic_learning_rate,
                    imitator_learning_rate,
                    temp_learning_rate,
                    alpha_learning_rate,
                    gamma,
                    tau,
                    n_critics,
                    initial_temperature,
                    initial_alpha,
                    alpha_threshold,
                    lam,
                    n_action_samples,
                    mmd_sigma,
                    eps,
                    use_batch_norm,
                    q_func_type,
                    use_gpu=False,
                    scaler=scaler)

    x = torch.rand(32, *observation_shape)
    target = impl.compute_target(x)
    if q_func_type == 'mean':
        assert target.shape == (32, 1)
    else:
        n_quantiles = impl.q_func.q_funcs[0].n_quantiles
        assert target.shape == (32, n_quantiles)

    torch_impl_tester(impl,
                      discrete=False,
                      deterministic_best_action=q_func_type != 'iqn')
