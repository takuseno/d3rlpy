import pytest
import torch

from skbrl.algos.torch.bcq_impl import BCQImpl, DiscreteBCQImpl
from skbrl.tests.algos.algo_test import torch_impl_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (1, 48, 48)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('actor_learning_rate', [1e-3])
@pytest.mark.parametrize('critic_learning_rate', [1e-3])
@pytest.mark.parametrize('imitator_learning_rate', [1e-3])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('tau', [0.05])
@pytest.mark.parametrize('n_critics', [2])
@pytest.mark.parametrize('lam', [0.75])
@pytest.mark.parametrize('n_action_samples', [100])
@pytest.mark.parametrize('action_flexibility', [0.05])
@pytest.mark.parametrize('latent_size', [32])
@pytest.mark.parametrize('beta', [0.5])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('use_quantile_regression', [True, False])
def test_bcq_impl(observation_shape, action_size, actor_learning_rate,
                  critic_learning_rate, imitator_learning_rate, gamma, tau,
                  n_critics, lam, n_action_samples, action_flexibility,
                  latent_size, beta, eps, use_batch_norm,
                  use_quantile_regression):
    impl = BCQImpl(observation_shape,
                   action_size,
                   actor_learning_rate,
                   critic_learning_rate,
                   imitator_learning_rate,
                   gamma,
                   tau,
                   n_critics,
                   lam,
                   n_action_samples,
                   action_flexibility,
                   latent_size,
                   beta,
                   eps,
                   use_batch_norm,
                   use_quantile_regression,
                   use_gpu=False)

    # test internal methods
    x = torch.rand(32, *observation_shape)

    repeated_x = impl._repeat_observation(x)
    assert repeated_x.shape == (32, n_action_samples) + observation_shape

    action = impl._sample_action(repeated_x)
    assert action.shape == (32, n_action_samples, action_size)

    value = impl._predict_value(repeated_x, action)
    assert value.shape == (n_critics, 32 * n_action_samples, 1)

    target_values = impl._predict_value(repeated_x, action, target=True)
    if use_quantile_regression:
        assert target_values.shape == (n_critics, 32 * n_action_samples, 200)
    else:
        assert target_values.shape == (n_critics, 32 * n_action_samples, 1)

    target = impl.compute_target(x)
    if use_quantile_regression:
        assert target.shape == (32, 200)
    else:
        assert target.shape == (32, 1)

    best_action = impl._predict_best_action(x)
    assert best_action.shape == (32, action_size)

    torch_impl_tester(impl, discrete=False, deterministic_best_action=False)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [2.5e-4])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('action_flexibility', [0.3])
@pytest.mark.parametrize('beta', [1e-2])
@pytest.mark.parametrize('eps', [0.95])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('use_quantile_regression', [True, False])
def test_discrete_bcq_impl(observation_shape, action_size, learning_rate,
                           gamma, action_flexibility, beta, eps,
                           use_batch_norm, use_quantile_regression):
    impl = DiscreteBCQImpl(observation_shape,
                           action_size,
                           learning_rate,
                           gamma,
                           action_flexibility,
                           beta,
                           eps,
                           use_batch_norm,
                           use_quantile_regression,
                           use_gpu=False)
    torch_impl_tester(impl, discrete=True)
