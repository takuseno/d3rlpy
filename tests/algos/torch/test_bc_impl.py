import pytest

from d3rlpy.algos.torch.bc_impl import BCImpl, DiscreteBCImpl
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from tests.algos.algo_test import (
    DummyActionScaler,
    DummyScaler,
    torch_impl_tester,
)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [1e-3])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("policy_type", ["deterministic", "stochastic"])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
def test_bc_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    policy_type,
    scaler,
    action_scaler,
):
    impl = BCImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        learning_rate=learning_rate,
        optim_factory=optim_factory,
        encoder_factory=encoder_factory,
        policy_type=policy_type,
        use_gpu=None,
        scaler=scaler,
        action_scaler=action_scaler,
    )
    torch_impl_tester(impl, discrete=False, imitator=True)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [1e-3])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("beta", [0.5])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
def test_discrete_bc_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    beta,
    scaler,
):
    impl = DiscreteBCImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        learning_rate=learning_rate,
        optim_factory=optim_factory,
        encoder_factory=encoder_factory,
        beta=beta,
        use_gpu=None,
        scaler=scaler,
    )
    torch_impl_tester(impl, discrete=True, imitator=True)
