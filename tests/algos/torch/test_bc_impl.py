import pytest

from d3rlpy.algos.torch.bc_impl import BCImpl, DiscreteBCImpl
from d3rlpy.augmentation import DrQPipeline
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.encoders import DefaultEncoderFactory
from tests.algos.algo_test import (
    torch_impl_tester,
    DummyScaler,
    DummyActionScaler,
)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [1e-3])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
@pytest.mark.parametrize("augmentation", [DrQPipeline()])
def test_bc_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    scaler,
    action_scaler,
    augmentation,
):
    impl = BCImpl(
        observation_shape,
        action_size,
        learning_rate,
        optim_factory,
        encoder_factory,
        use_gpu=False,
        scaler=scaler,
        action_scaler=action_scaler,
        augmentation=augmentation,
    )
    torch_impl_tester(impl, discrete=False, imitator=True)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [1e-3])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("beta", [0.5])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("augmentation", [DrQPipeline()])
def test_discrete_bc_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    beta,
    scaler,
    augmentation,
):
    impl = DiscreteBCImpl(
        observation_shape,
        action_size,
        learning_rate,
        optim_factory,
        encoder_factory,
        beta,
        use_gpu=False,
        scaler=scaler,
        augmentation=augmentation,
    )
    torch_impl_tester(impl, discrete=True, imitator=True)
