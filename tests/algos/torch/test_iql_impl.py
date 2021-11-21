import pytest

from d3rlpy.algos.torch.iql_impl import IQLImpl
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from tests.algos.algo_test import (
    DummyActionScaler,
    DummyRewardScaler,
    DummyScaler,
    torch_impl_tester,
)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("actor_learning_rate", [1e-3])
@pytest.mark.parametrize("critic_learning_rate", [1e-3])
@pytest.mark.parametrize("value_learning_rate", [1e-3])
@pytest.mark.parametrize("actor_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("critic_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("value_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("expectile", [0.7])
@pytest.mark.parametrize("weight_temp", [3.0])
@pytest.mark.parametrize("max_weight", [100.0])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
@pytest.mark.parametrize("reward_scaler", [None, DummyRewardScaler()])
def test_iql_impl(
    observation_shape,
    action_size,
    actor_learning_rate,
    critic_learning_rate,
    value_learning_rate,
    actor_optim_factory,
    critic_optim_factory,
    value_optim_factory,
    encoder_factory,
    gamma,
    tau,
    n_critics,
    expectile,
    weight_temp,
    max_weight,
    scaler,
    action_scaler,
    reward_scaler,
):
    impl = IQLImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        value_learning_rate=value_learning_rate,
        actor_optim_factory=actor_optim_factory,
        critic_optim_factory=critic_optim_factory,
        value_optim_factory=value_optim_factory,
        actor_encoder_factory=encoder_factory,
        critic_encoder_factory=encoder_factory,
        value_encoder_factory=encoder_factory,
        gamma=gamma,
        tau=tau,
        n_critics=n_critics,
        expectile=expectile,
        weight_temp=weight_temp,
        max_weight=max_weight,
        use_gpu=None,
        scaler=scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    torch_impl_tester(impl, discrete=False, deterministic_best_action=True)
