import pytest

from d3rlpy.algos.torch.plas_impl import PLASImpl, PLASWithPerturbationImpl
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
@pytest.mark.parametrize("imitator_learning_rate", [1e-3])
@pytest.mark.parametrize("actor_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("critic_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("imitator_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("target_reduction_type", ["min"])
@pytest.mark.parametrize("lam", [0.75])
@pytest.mark.parametrize("beta", [0.5])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
@pytest.mark.parametrize("augmentation", [DrQPipeline()])
def test_plas_impl(
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
    target_reduction_type,
    lam,
    beta,
    scaler,
    action_scaler,
    augmentation,
):
    impl = PLASImpl(
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
        q_func_factory=create_q_func_factory(q_func_factory),
        gamma=gamma,
        tau=tau,
        n_critics=n_critics,
        target_reduction_type=target_reduction_type,
        lam=lam,
        beta=beta,
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
@pytest.mark.parametrize("imitator_learning_rate", [1e-3])
@pytest.mark.parametrize("actor_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("critic_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("imitator_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("target_reduction_type", ["min"])
@pytest.mark.parametrize("lam", [0.75])
@pytest.mark.parametrize("beta", [0.5])
@pytest.mark.parametrize("action_flexibility", [0.05])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
@pytest.mark.parametrize("augmentation", [DrQPipeline()])
def test_plas_with_perturbation_impl(
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
    target_reduction_type,
    lam,
    beta,
    action_flexibility,
    scaler,
    action_scaler,
    augmentation,
):
    impl = PLASWithPerturbationImpl(
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
        q_func_factory=create_q_func_factory(q_func_factory),
        gamma=gamma,
        tau=tau,
        n_critics=n_critics,
        target_reduction_type=target_reduction_type,
        lam=lam,
        beta=beta,
        action_flexibility=action_flexibility,
        use_gpu=None,
        scaler=scaler,
        action_scaler=action_scaler,
        augmentation=augmentation,
    )
    torch_impl_tester(
        impl, discrete=False, deterministic_best_action=q_func_factory != "iqn"
    )
