import pytest

from d3rlpy.algos.torch.ddpg_impl import DDPGImpl
from d3rlpy.augmentation import AugmentationPipeline
from d3rlpy.optimizers import AdamFactory
from d3rlpy.encoders import DefaultEncoderFactory
from d3rlpy.q_functions import create_q_func_factory
from tests.algos.algo_test import torch_impl_tester, DummyScaler


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('actor_learning_rate', [1e-3])
@pytest.mark.parametrize('critic_learning_rate', [1e-3])
@pytest.mark.parametrize('actor_optim_factory', [AdamFactory()])
@pytest.mark.parametrize('critic_optim_factory', [AdamFactory()])
@pytest.mark.parametrize('encoder_factory', [DefaultEncoderFactory()])
@pytest.mark.parametrize('q_func_factory', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('tau', [0.05])
@pytest.mark.parametrize('n_critics', [1])
@pytest.mark.parametrize('bootstrap', [False])
@pytest.mark.parametrize('share_encoder', [True])
@pytest.mark.parametrize('reguralizing_rate', [1e-8])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
def test_ddpg_impl(observation_shape, action_size, actor_learning_rate,
                   critic_learning_rate, actor_optim_factory,
                   critic_optim_factory, encoder_factory, q_func_factory,
                   gamma, tau, n_critics, bootstrap, share_encoder,
                   reguralizing_rate, scaler, augmentation, n_augmentations):
    impl = DDPGImpl(observation_shape,
                    action_size,
                    actor_learning_rate,
                    critic_learning_rate,
                    actor_optim_factory,
                    critic_optim_factory,
                    encoder_factory,
                    encoder_factory,
                    create_q_func_factory(q_func_factory),
                    gamma,
                    tau,
                    n_critics,
                    bootstrap,
                    share_encoder,
                    reguralizing_rate,
                    use_gpu=False,
                    scaler=scaler,
                    augmentation=augmentation,
                    n_augmentations=n_augmentations)
    torch_impl_tester(impl,
                      discrete=False,
                      deterministic_best_action=q_func_factory != 'iqn')
