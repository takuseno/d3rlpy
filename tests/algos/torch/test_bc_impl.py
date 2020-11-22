import pytest

from d3rlpy.algos.torch.bc_impl import BCImpl, DiscreteBCImpl
from d3rlpy.augmentation import AugmentationPipeline
from d3rlpy.optimizers import AdamFactory
from tests.algos.algo_test import torch_impl_tester, DummyScaler


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [1e-3])
@pytest.mark.parametrize('optim_factory', [AdamFactory()])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
@pytest.mark.parametrize('encoder_params', [{}])
def test_bc_impl(observation_shape, action_size, learning_rate, optim_factory,
                 use_batch_norm, scaler, augmentation, n_augmentations,
                 encoder_params):
    impl = BCImpl(observation_shape,
                  action_size,
                  learning_rate,
                  optim_factory,
                  use_batch_norm,
                  use_gpu=False,
                  scaler=scaler,
                  augmentation=augmentation,
                  n_augmentations=n_augmentations,
                  encoder_params=encoder_params)
    torch_impl_tester(impl, discrete=False, imitator=True)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [1e-3])
@pytest.mark.parametrize('optim_factory', [AdamFactory()])
@pytest.mark.parametrize('beta', [0.5])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
@pytest.mark.parametrize('encoder_params', [{}])
def test_discrete_bc_impl(observation_shape, action_size, learning_rate,
                          optim_factory, beta, use_batch_norm, scaler,
                          augmentation, n_augmentations, encoder_params):
    impl = DiscreteBCImpl(observation_shape,
                          action_size,
                          learning_rate,
                          optim_factory,
                          beta,
                          use_batch_norm,
                          use_gpu=False,
                          scaler=scaler,
                          augmentation=augmentation,
                          n_augmentations=n_augmentations,
                          encoder_params=encoder_params)
    torch_impl_tester(impl, discrete=True, imitator=True)
