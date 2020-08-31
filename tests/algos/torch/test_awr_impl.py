import pytest

from d3rlpy.algos.torch.awr_impl import AWRImpl
from d3rlpy.augmentation import AugmentationPipeline
from tests.algos.algo_test import torch_impl_tester, DummyScaler


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('actor_learning_rate', [5e-5])
@pytest.mark.parametrize('critic_learning_rate', [1e-4])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
@pytest.mark.parametrize('encoder_params', [{}])
def test_awr_impl(observation_shape, action_size, actor_learning_rate,
                  critic_learning_rate, eps, use_batch_norm, scaler,
                  augmentation, n_augmentations, encoder_params):
    impl = AWRImpl(observation_shape,
                   action_size,
                   actor_learning_rate,
                   critic_learning_rate,
                   eps,
                   use_batch_norm,
                   use_gpu=False,
                   scaler=scaler,
                   augmentation=augmentation,
                   n_augmentations=n_augmentations,
                   encoder_params=encoder_params)
    torch_impl_tester(impl, discrete=False, test_with_std=False)
