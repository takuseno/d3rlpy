import pytest

from d3rlpy.algos.torch.cql_impl import CQLImpl, DiscreteCQLImpl
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
@pytest.mark.parametrize("alpha_learning_rate", [1e-3])
@pytest.mark.parametrize("actor_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("critic_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("temp_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("alpha_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("bootstrap", [False])
@pytest.mark.parametrize("share_encoder", [False, True])
@pytest.mark.parametrize("initial_temperature", [1.0])
@pytest.mark.parametrize("initial_alpha", [5.0])
@pytest.mark.parametrize("alpha_threshold", [10.0])
@pytest.mark.parametrize("n_action_samples", [10])
@pytest.mark.parametrize("soft_q_backup", [True])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
@pytest.mark.parametrize("augmentation", [DrQPipeline()])
def test_cql_impl(
    observation_shape,
    action_size,
    actor_learning_rate,
    critic_learning_rate,
    temp_learning_rate,
    alpha_learning_rate,
    actor_optim_factory,
    critic_optim_factory,
    temp_optim_factory,
    alpha_optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    tau,
    n_critics,
    bootstrap,
    share_encoder,
    initial_temperature,
    initial_alpha,
    alpha_threshold,
    n_action_samples,
    soft_q_backup,
    scaler,
    action_scaler,
    augmentation,
):
    impl = CQLImpl(
        observation_shape,
        action_size,
        actor_learning_rate,
        critic_learning_rate,
        temp_learning_rate,
        alpha_learning_rate,
        actor_optim_factory,
        critic_optim_factory,
        temp_optim_factory,
        alpha_optim_factory,
        encoder_factory,
        encoder_factory,
        create_q_func_factory(q_func_factory),
        gamma,
        tau,
        n_critics,
        bootstrap,
        share_encoder,
        initial_temperature,
        initial_alpha,
        alpha_threshold,
        n_action_samples,
        soft_q_backup,
        use_gpu=False,
        scaler=scaler,
        action_scaler=action_scaler,
        augmentation=augmentation,
    )
    torch_impl_tester(
        impl, discrete=False, deterministic_best_action=q_func_factory != "iqn"
    )


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [2.5e-4])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("bootstrap", [False])
@pytest.mark.parametrize("share_encoder", [False, True])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("augmentation", [DrQPipeline()])
def test_discrete_cql_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    n_critics,
    bootstrap,
    share_encoder,
    scaler,
    augmentation,
):
    impl = DiscreteCQLImpl(
        observation_shape,
        action_size,
        learning_rate,
        optim_factory,
        encoder_factory,
        create_q_func_factory(q_func_factory),
        gamma,
        n_critics,
        bootstrap,
        share_encoder,
        use_gpu=False,
        scaler=scaler,
        augmentation=augmentation,
    )
    torch_impl_tester(
        impl, discrete=True, deterministic_best_action=q_func_factory != "iqn"
    )
