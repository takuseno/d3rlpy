import pytest
import numpy as np
import os

from d3rlpy.metrics.ope.torch.fqe_impl import FQEImpl, DiscreteFQEImpl
from d3rlpy.augmentation import AugmentationPipeline
from tests.algos.algo_test import DummyScaler


def torch_impl_tester(impl, discrete):
    # setup implementation
    impl.build()

    observations = np.random.random((100, ) + impl.observation_shape)
    if discrete:
        actions = np.random.randint(impl.action_size, size=100)
    else:
        actions = np.random.random((100, impl.action_size))

    # check predict_value
    value = impl.predict_value(observations, actions, with_std=False)
    assert value.shape == (100, )

    # check predict_value with standard deviation
    value, std = impl.predict_value(observations, actions, with_std=True)
    assert value.shape == (100, )
    assert std.shape == (100, )

    # check save_model and load_model
    impl.save_model(os.path.join('test_data', 'model.pt'))
    impl.load_model(os.path.join('test_data', 'model.pt'))


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [1e-3])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('n_critics', [1])
@pytest.mark.parametrize('bootstrap', [False])
@pytest.mark.parametrize('share_encoder', [True])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
@pytest.mark.parametrize('encoder_params', [{}])
def test_fqe_impl(observation_shape, action_size, learning_rate, gamma,
                  n_critics, bootstrap, share_encoder, eps, use_batch_norm,
                  q_func_type, scaler, augmentation, n_augmentations,
                  encoder_params):
    fqe = FQEImpl(observation_shape,
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

    torch_impl_tester(fqe, False)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [1e-3])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('n_critics', [1])
@pytest.mark.parametrize('bootstrap', [False])
@pytest.mark.parametrize('share_encoder', [True])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('n_augmentations', [1])
@pytest.mark.parametrize('encoder_params', [{}])
def test_discrete_fqe_impl(observation_shape, action_size, learning_rate,
                           gamma, n_critics, bootstrap, share_encoder, eps,
                           use_batch_norm, q_func_type, scaler, augmentation,
                           n_augmentations, encoder_params):
    fqe = DiscreteFQEImpl(observation_shape,
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

    torch_impl_tester(fqe, True)
