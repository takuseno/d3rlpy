import pytest

from d3rlpy.algos.torch.sac_impl import SACImpl, DiscreteSACImpl
from d3rlpy.augmentation import DrQPipeline
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.q_functions import create_q_func_factory
from tests.algos.algo_test import (
    torch_impl_tester,
    DummyScaler,
    DummyActionScaler,
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
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("target_reduction_type", ["min"])
@pytest.mark.parametrize("initial_temperature", [1.0])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
@pytest.mark.parametrize("augmentation", [DrQPipeline()])
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
    target_reduction_type,
    initial_temperature,
    scaler,
    action_scaler,
    augmentation,
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
        q_func_factory=create_q_func_factory(q_func_factory),
        gamma=gamma,
        tau=tau,
        n_critics=n_critics,
        target_reduction_type=target_reduction_type,
        initial_temperature=initial_temperature,
        use_gpu=None,
        scaler=scaler,
        action_scaler=action_scaler,
        augmentation=augmentation,
    )
    torch_impl_tester(
        impl, discrete=False, deterministic_best_action=q_func_factory != "iqn"
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
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("initial_temperature", [1.0])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("augmentation", [DrQPipeline()])
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
    scaler,
    augmentation,
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
        q_func_factory=create_q_func_factory(q_func_factory),
        gamma=gamma,
        n_critics=n_critics,
        initial_temperature=initial_temperature,
        use_gpu=None,
        scaler=scaler,
        augmentation=augmentation,
    )
    torch_impl_tester(
        impl, discrete=True, deterministic_best_action=q_func_factory != "iqn"
    )
