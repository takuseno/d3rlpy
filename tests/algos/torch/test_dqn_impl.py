import pytest

from d3rlpy.augmentation import AugmentationPipeline
from d3rlpy.algos.torch.dqn_impl import DQNImpl, DoubleDQNImpl
from tests.algos.algo_test import torch_impl_tester, DummyScaler


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [2.5e-4])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('n_critics', [1])
@pytest.mark.parametrize('bootstrap', [False])
@pytest.mark.parametrize('share_encoder', [False, True])
@pytest.mark.parametrize('eps', [0.95])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
@pytest.mark.parametrize('encoder_params', [{}])
def test_dqn_impl(observation_shape, action_size, learning_rate, gamma,
                  n_critics, bootstrap, share_encoder, eps, use_batch_norm,
                  q_func_type, scaler, augmentation, n_augmentations,
                  encoder_params):
    impl = DQNImpl(observation_shape,
                   action_size,
                   learning_rate,
                   gamma,
                   n_critics,
                   bootstrap,
                   share_encoder,
                   eps,
                   use_batch_norm,
                   q_func_type,
                   use_gpu=False,
                   scaler=scaler,
                   augmentation=augmentation,
                   n_augmentations=n_augmentations,
                   encoder_params=encoder_params)
    torch_impl_tester(impl,
                      discrete=True,
                      deterministic_best_action=q_func_type != 'iqn')


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [2.5e-4])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('n_critics', [1])
@pytest.mark.parametrize('bootstrap', [False])
@pytest.mark.parametrize('share_encoder', [False, True])
@pytest.mark.parametrize('eps', [0.95])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
@pytest.mark.parametrize('encoder_params', [{}])
def test_double_dqn_impl(observation_shape, action_size, learning_rate, gamma,
                         n_critics, bootstrap, share_encoder, eps,
                         use_batch_norm, q_func_type, scaler, augmentation,
                         n_augmentations, encoder_params):
    impl = DoubleDQNImpl(observation_shape,
                         action_size,
                         learning_rate,
                         gamma,
                         n_critics,
                         bootstrap,
                         share_encoder,
                         eps,
                         use_batch_norm,
                         q_func_type=q_func_type,
                         use_gpu=False,
                         scaler=scaler,
                         augmentation=augmentation,
                         n_augmentations=n_augmentations,
                         encoder_params=encoder_params)
    torch_impl_tester(impl,
                      discrete=True,
                      deterministic_best_action=q_func_type != 'iqn')
