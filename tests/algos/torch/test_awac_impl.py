import pytest

from d3rlpy.algos.torch.awac_impl import AWACImpl
from d3rlpy.augmentation import AugmentationPipeline
from d3rlpy.optimizers import AdamFactory
from tests.algos.algo_test import torch_impl_tester, DummyScaler


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('actor_learning_rate', [1e-3])
@pytest.mark.parametrize('critic_learning_rate', [1e-3])
@pytest.mark.parametrize('actor_optim_factory', [AdamFactory()])
@pytest.mark.parametrize('critic_optim_factory', [AdamFactory()])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('tau', [0.05])
@pytest.mark.parametrize('lam', [1.0])
@pytest.mark.parametrize('n_action_samples', [10])
@pytest.mark.parametrize('max_weight', [20.0])
@pytest.mark.parametrize('n_critics', [1])
@pytest.mark.parametrize('bootstrap', [False])
@pytest.mark.parametrize('share_encoder', [True])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
@pytest.mark.parametrize('encoder_params', [{}])
def test_awac_impl(observation_shape, action_size, actor_learning_rate,
                   critic_learning_rate, actor_optim_factory,
                   critic_optim_factory, gamma, tau, lam, n_action_samples,
                   max_weight, n_critics, bootstrap, share_encoder,
                   use_batch_norm, q_func_type, scaler, augmentation,
                   n_augmentations, encoder_params):
    impl = AWACImpl(observation_shape,
                    action_size,
                    actor_learning_rate,
                    critic_learning_rate,
                    actor_optim_factory,
                    critic_optim_factory,
                    gamma,
                    tau,
                    lam,
                    n_action_samples,
                    max_weight,
                    n_critics,
                    bootstrap,
                    share_encoder,
                    use_batch_norm,
                    q_func_type=q_func_type,
                    use_gpu=False,
                    scaler=scaler,
                    augmentation=augmentation,
                    n_augmentations=n_augmentations,
                    encoder_params=encoder_params)
    torch_impl_tester(impl,
                      discrete=False,
                      deterministic_best_action=q_func_type != 'iqn')
