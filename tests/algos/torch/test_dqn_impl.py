import pytest

from d3rlpy.augmentation import AugmentationPipeline
from d3rlpy.algos.torch.dqn_impl import DQNImpl, DoubleDQNImpl
from d3rlpy.optimizers import AdamFactory
from tests import create_encoder_factory
from tests.algos.algo_test import torch_impl_tester, DummyScaler


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [2.5e-4])
@pytest.mark.parametrize('optim_factory', [AdamFactory()])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('n_critics', [1])
@pytest.mark.parametrize('bootstrap', [False])
@pytest.mark.parametrize('share_encoder', [False, True])
@pytest.mark.parametrize('use_encoder_factory', [True, False])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
def test_dqn_impl(observation_shape, action_size, learning_rate, optim_factory,
                  gamma, n_critics, bootstrap, share_encoder,
                  use_encoder_factory, q_func_type, scaler, augmentation,
                  n_augmentations):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)
    impl = DQNImpl(observation_shape,
                   action_size,
                   learning_rate,
                   optim_factory,
                   encoder_factory,
                   gamma,
                   n_critics,
                   bootstrap,
                   share_encoder,
                   q_func_type,
                   use_gpu=False,
                   scaler=scaler,
                   augmentation=augmentation,
                   n_augmentations=n_augmentations)
    torch_impl_tester(impl,
                      discrete=True,
                      deterministic_best_action=q_func_type != 'iqn')


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [2.5e-4])
@pytest.mark.parametrize('optim_factory', [AdamFactory()])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('n_critics', [1])
@pytest.mark.parametrize('bootstrap', [False])
@pytest.mark.parametrize('share_encoder', [False, True])
@pytest.mark.parametrize('use_encoder_factory', [True, False])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
def test_double_dqn_impl(observation_shape, action_size, learning_rate,
                         optim_factory, gamma, n_critics, bootstrap,
                         share_encoder, use_encoder_factory, q_func_type,
                         scaler, augmentation, n_augmentations):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)
    impl = DoubleDQNImpl(observation_shape,
                         action_size,
                         learning_rate,
                         optim_factory,
                         encoder_factory,
                         gamma,
                         n_critics,
                         bootstrap,
                         share_encoder,
                         q_func_type=q_func_type,
                         use_gpu=False,
                         scaler=scaler,
                         augmentation=augmentation,
                         n_augmentations=n_augmentations)
    torch_impl_tester(impl,
                      discrete=True,
                      deterministic_best_action=q_func_type != 'iqn')
