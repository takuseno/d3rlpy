import pytest

from d3rlpy.algos.torch.awr_impl import AWRImpl, DiscreteAWRImpl
from d3rlpy.augmentation import AugmentationPipeline
from d3rlpy.optimizers import AdamFactory
from d3rlpy.encoders import DefaultEncoderFactory
from tests.algos.algo_test import torch_impl_tester, DummyScaler


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('actor_learning_rate', [5e-5])
@pytest.mark.parametrize('critic_learning_rate', [1e-4])
@pytest.mark.parametrize('actor_optim_factory', [AdamFactory()])
@pytest.mark.parametrize('critic_optim_factory', [AdamFactory()])
@pytest.mark.parametrize('encoder_factory', [DefaultEncoderFactory()])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
def test_awr_impl(observation_shape, action_size, actor_learning_rate,
                  critic_learning_rate, actor_optim_factory,
                  critic_optim_factory, encoder_factory, scaler, augmentation,
                  n_augmentations):
    impl = AWRImpl(observation_shape,
                   action_size,
                   actor_learning_rate,
                   critic_learning_rate,
                   actor_optim_factory,
                   critic_optim_factory,
                   encoder_factory,
                   encoder_factory,
                   use_gpu=False,
                   scaler=scaler,
                   augmentation=augmentation,
                   n_augmentations=n_augmentations)
    torch_impl_tester(impl, discrete=False, test_with_std=False)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('actor_learning_rate', [5e-5])
@pytest.mark.parametrize('critic_learning_rate', [1e-4])
@pytest.mark.parametrize('actor_optim_factory', [AdamFactory()])
@pytest.mark.parametrize('critic_optim_factory', [AdamFactory()])
@pytest.mark.parametrize('encoder_factory', [DefaultEncoderFactory()])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
def test_discrete_awr_impl(observation_shape, action_size, actor_learning_rate,
                           critic_learning_rate, actor_optim_factory,
                           critic_optim_factory, encoder_factory, scaler,
                           augmentation, n_augmentations):
    impl = DiscreteAWRImpl(observation_shape,
                           action_size,
                           actor_learning_rate,
                           critic_learning_rate,
                           actor_optim_factory,
                           critic_optim_factory,
                           encoder_factory,
                           encoder_factory,
                           use_gpu=False,
                           scaler=scaler,
                           augmentation=augmentation,
                           n_augmentations=n_augmentations)
    torch_impl_tester(impl, discrete=True, test_with_std=False)
