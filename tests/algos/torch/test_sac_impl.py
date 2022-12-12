import pytest

from d3rlpy.algos.torch.sac_impl import DiscreteSACImpl, SACImpl
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
@pytest.mark.parametrize("temp_learning_rate", [1e-3])
@pytest.mark.parametrize("actor_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("critic_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("temp_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("initial_temperature", [1.0])
@pytest.mark.parametrize("observation_scaler", [None, DummyObservationScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
@pytest.mark.parametrize("reward_scaler", [None, DummyRewardScaler()])
def test_sac_impl(
    observation_shape,
    action_size,
    actor_learning_rate,
    critic_learning_rate,
    temp_learning_rate,
    actor_optim_factory,
    critic_optim_factory,
    temp_optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    tau,
    n_critics,
    initial_temperature,
    observation_scaler,
    action_scaler,
    reward_scaler,
):
    impl = SACImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        temp_learning_rate=temp_learning_rate,
        actor_optim_factory=actor_optim_factory,
        critic_optim_factory=critic_optim_factory,
        temp_optim_factory=temp_optim_factory,
        actor_encoder_factory=encoder_factory,
        critic_encoder_factory=encoder_factory,
        q_func_factory=q_func_factory,
        gamma=gamma,
        tau=tau,
        n_critics=n_critics,
        initial_temperature=initial_temperature,
        use_gpu=None,
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    torch_impl_tester(impl, discrete=False)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("actor_learning_rate", [1e-3])
@pytest.mark.parametrize("critic_learning_rate", [1e-3])
@pytest.mark.parametrize("temp_learning_rate", [1e-3])
@pytest.mark.parametrize("actor_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("critic_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("temp_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("initial_temperature", [1.0])
@pytest.mark.parametrize("observation_scaler", [None, DummyObservationScaler()])
@pytest.mark.parametrize("reward_scaler", [None, DummyRewardScaler()])
def test_discrete_sac_impl(
    observation_shape,
    action_size,
    actor_learning_rate,
    critic_learning_rate,
    temp_learning_rate,
    actor_optim_factory,
    critic_optim_factory,
    temp_optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    n_critics,
    initial_temperature,
    observation_scaler,
    reward_scaler,
):
    impl = DiscreteSACImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        temp_learning_rate=temp_learning_rate,
        actor_optim_factory=actor_optim_factory,
        critic_optim_factory=critic_optim_factory,
        temp_optim_factory=temp_optim_factory,
        actor_encoder_factory=encoder_factory,
        critic_encoder_factory=encoder_factory,
        q_func_factory=q_func_factory,
        gamma=gamma,
        n_critics=n_critics,
        initial_temperature=initial_temperature,
        use_gpu=None,
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    torch_impl_tester(impl, discrete=True)
