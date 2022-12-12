import pytest

from d3rlpy.algos.torch.td3_plus_bc_impl import TD3PlusBCImpl
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory, QRQFunctionFactory
from tests.algos.algo_test import (
    DummyActionScaler,
    DummyObservationScaler,
    DummyRewardScaler,
    torch_impl_tester,
)


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
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("target_smoothing_sigma", [0.2])
@pytest.mark.parametrize("target_smoothing_clip", [0.5])
@pytest.mark.parametrize("alpha", [0.25])
@pytest.mark.parametrize("observation_scaler", [None, DummyObservationScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
@pytest.mark.parametrize("reward_scaler", [None, DummyRewardScaler()])
def test_td3_plus_bc_impl(
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
    n_critics,
    target_smoothing_sigma,
    target_smoothing_clip,
    alpha,
    observation_scaler,
    action_scaler,
    reward_scaler,
):
    impl = TD3PlusBCImpl(
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
        n_critics=n_critics,
        target_smoothing_sigma=target_smoothing_sigma,
        target_smoothing_clip=target_smoothing_clip,
        alpha=alpha,
        use_gpu=None,
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    torch_impl_tester(impl, discrete=False)
