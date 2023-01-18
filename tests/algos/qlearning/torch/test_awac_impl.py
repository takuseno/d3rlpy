import pytest

from d3rlpy.algos.qlearning.torch.awac_impl import AWACImpl
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory, QRQFunctionFactory
from tests.algos.qlearning.algo_impl_test import impl_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("actor_learning_rate", [1e-3])
@pytest.mark.parametrize("critic_learning_rate", [1e-3])
@pytest.mark.parametrize("actor_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("critic_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("lam", [1.0])
@pytest.mark.parametrize("n_action_samples", [10])
@pytest.mark.parametrize("n_critics", [1])
def test_awac_impl(
    observation_shape,
    action_size,
    actor_learning_rate,
    critic_learning_rate,
    actor_optim_factory,
    critic_optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    tau,
    lam,
    n_action_samples,
    n_critics,
):
    impl = AWACImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        actor_optim_factory=actor_optim_factory,
        critic_optim_factory=critic_optim_factory,
        actor_encoder_factory=encoder_factory,
        critic_encoder_factory=encoder_factory,
        q_func_factory=q_func_factory,
        gamma=gamma,
        tau=tau,
        lam=lam,
        n_action_samples=n_action_samples,
        n_critics=n_critics,
        device="cpu:0",
    )
    impl_tester(impl, discrete=False)
