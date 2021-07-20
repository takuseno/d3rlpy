import pytest

from d3rlpy.algos.torch.awac_impl import AWACImpl
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.q_functions import create_q_func_factory
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
@pytest.mark.parametrize("actor_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("critic_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("lam", [1.0])
@pytest.mark.parametrize("n_action_samples", [10])
@pytest.mark.parametrize("max_weight", [20.0])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("target_reduction_type", ["min"])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
@pytest.mark.parametrize("reward_scaler", [None, DummyRewardScaler()])
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
    max_weight,
    n_critics,
    target_reduction_type,
    scaler,
    action_scaler,
    reward_scaler,
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
        q_func_factory=create_q_func_factory(q_func_factory),
        gamma=gamma,
        tau=tau,
        lam=lam,
        n_action_samples=n_action_samples,
        max_weight=max_weight,
        n_critics=n_critics,
        target_reduction_type=target_reduction_type,
        use_gpu=None,
        scaler=scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    torch_impl_tester(
        impl, discrete=False, deterministic_best_action=q_func_factory != "iqn"
    )
