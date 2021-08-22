import pytest

from d3rlpy.algos.torch.crr_impl import CRRImpl
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
@pytest.mark.parametrize("beta", [1.0])
@pytest.mark.parametrize("n_action_samples", [10])
@pytest.mark.parametrize("advantage_type", ["mean"])
@pytest.mark.parametrize("weight_type", ["exp"])
@pytest.mark.parametrize("max_weight", [20.0])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("tau", [5e-3])
@pytest.mark.parametrize("target_reduction_type", ["min"])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
@pytest.mark.parametrize("reward_scaler", [None, DummyRewardScaler()])
def test_crr_impl(
    observation_shape,
    action_size,
    actor_learning_rate,
    critic_learning_rate,
    actor_optim_factory,
    critic_optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    beta,
    n_action_samples,
    advantage_type,
    weight_type,
    max_weight,
    n_critics,
    tau,
    target_reduction_type,
    scaler,
    action_scaler,
    reward_scaler,
):
    impl = CRRImpl(
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
        beta=beta,
        n_action_samples=n_action_samples,
        advantage_type=advantage_type,
        weight_type=weight_type,
        max_weight=max_weight,
        n_critics=n_critics,
        tau=tau,
        target_reduction_type=target_reduction_type,
        use_gpu=None,
        scaler=scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    torch_impl_tester(impl, discrete=False, deterministic_best_action=False)
