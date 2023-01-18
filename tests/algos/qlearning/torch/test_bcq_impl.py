import pytest
import torch

from d3rlpy.algos.qlearning.torch.bcq_impl import BCQImpl, DiscreteBCQImpl
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory, QRQFunctionFactory
from tests.algos.qlearning.algo_impl_test import impl_tester


@pytest.mark.parametrize("observation_shape", [(100,), (1, 48, 48)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("actor_learning_rate", [1e-3])
@pytest.mark.parametrize("critic_learning_rate", [1e-3])
@pytest.mark.parametrize("imitator_learning_rate", [1e-3])
@pytest.mark.parametrize("actor_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("critic_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("imitator_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("lam", [0.75])
@pytest.mark.parametrize("n_action_samples", [10])  # small for test
@pytest.mark.parametrize("action_flexibility", [0.05])
@pytest.mark.parametrize("beta", [0.5])
def test_bcq_impl(
    observation_shape,
    action_size,
    actor_learning_rate,
    critic_learning_rate,
    imitator_learning_rate,
    actor_optim_factory,
    critic_optim_factory,
    imitator_optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    tau,
    n_critics,
    lam,
    n_action_samples,
    action_flexibility,
    beta,
):
    impl = BCQImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        imitator_learning_rate=imitator_learning_rate,
        actor_optim_factory=actor_optim_factory,
        critic_optim_factory=critic_optim_factory,
        imitator_optim_factory=imitator_optim_factory,
        actor_encoder_factory=encoder_factory,
        critic_encoder_factory=encoder_factory,
        imitator_encoder_factory=encoder_factory,
        q_func_factory=q_func_factory,
        gamma=gamma,
        tau=tau,
        n_critics=n_critics,
        lam=lam,
        n_action_samples=n_action_samples,
        action_flexibility=action_flexibility,
        beta=beta,
        device="cpu:0",
    )
    impl.build()

    # test internal methods
    x = torch.rand(32, *observation_shape)

    repeated_x = impl._repeat_observation(x)
    assert repeated_x.shape == (32, n_action_samples) + observation_shape

    action = impl._sample_repeated_action(repeated_x)
    assert action.shape == (32, n_action_samples, action_size)

    value = impl._predict_value(repeated_x, action)
    assert value.shape == (n_critics, 32 * n_action_samples, 1)

    impl_tester(impl, discrete=False)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [2.5e-4])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("action_flexibility", [0.3])
@pytest.mark.parametrize("beta", [1e-2])
def test_discrete_bcq_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    n_critics,
    action_flexibility,
    beta,
):
    impl = DiscreteBCQImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        learning_rate=learning_rate,
        optim_factory=optim_factory,
        encoder_factory=encoder_factory,
        q_func_factory=q_func_factory,
        gamma=gamma,
        n_critics=n_critics,
        action_flexibility=action_flexibility,
        beta=beta,
        device="cpu:0",
    )
    impl_tester(impl, discrete=True)
