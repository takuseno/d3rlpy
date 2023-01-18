import pytest

from d3rlpy.algos.qlearning.torch.bc_impl import BCImpl, DiscreteBCImpl
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from tests.algos.qlearning.algo_impl_test import impl_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [1e-3])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("policy_type", ["deterministic", "stochastic"])
def test_bc_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    policy_type,
):
    impl = BCImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        learning_rate=learning_rate,
        optim_factory=optim_factory,
        encoder_factory=encoder_factory,
        policy_type=policy_type,
        device="cpu:0",
    )
    impl_tester(impl, discrete=False, test_predict_value=False)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [1e-3])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("beta", [0.5])
def test_discrete_bc_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    beta,
):
    impl = DiscreteBCImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        learning_rate=learning_rate,
        optim_factory=optim_factory,
        encoder_factory=encoder_factory,
        beta=beta,
        device="cpu:0",
    )
    impl_tester(impl, discrete=True, test_predict_value=False)
