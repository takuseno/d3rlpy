import pytest
import torch

from d3rlpy.algos.torch.bear_impl import BEARImpl
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.q_functions import create_q_func_factory
from tests.algos.algo_test import (
    DummyActionScaler,
    DummyScaler,
    torch_impl_tester,
)


@pytest.mark.parametrize("observation_shape", [(100,), (1, 48, 48)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("actor_learning_rate", [1e-3])
@pytest.mark.parametrize("critic_learning_rate", [1e-3])
@pytest.mark.parametrize("imitator_learning_rate", [1e-3])
@pytest.mark.parametrize("temp_learning_rate", [1e-3])
@pytest.mark.parametrize("alpha_learning_rate", [1e-3])
@pytest.mark.parametrize("actor_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("critic_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("imitator_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("temp_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("alpha_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("initial_temperature", [1.0])
@pytest.mark.parametrize("initial_alpha", [1.0])
@pytest.mark.parametrize("alpha_threshold", [0.05])
@pytest.mark.parametrize("lam", [0.75])
@pytest.mark.parametrize("n_action_samples", [4])
@pytest.mark.parametrize("mmd_kernel", ["laplacian"])
@pytest.mark.parametrize("mmd_sigma", [20.0])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
def test_bear_impl(
    observation_shape,
    action_size,
    actor_learning_rate,
    critic_learning_rate,
    imitator_learning_rate,
    temp_learning_rate,
    alpha_learning_rate,
    actor_optim_factory,
    critic_optim_factory,
    imitator_optim_factory,
    temp_optim_factory,
    alpha_optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    tau,
    n_critics,
    initial_temperature,
    initial_alpha,
    alpha_threshold,
    lam,
    n_action_samples,
    mmd_kernel,
    mmd_sigma,
    scaler,
    action_scaler,
):
    impl = BEARImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        imitator_learning_rate=imitator_learning_rate,
        temp_learning_rate=temp_learning_rate,
        alpha_learning_rate=alpha_learning_rate,
        actor_optim_factory=actor_optim_factory,
        critic_optim_factory=critic_optim_factory,
        imitator_optim_factory=imitator_optim_factory,
        temp_optim_factory=temp_optim_factory,
        alpha_optim_factory=alpha_optim_factory,
        actor_encoder_factory=encoder_factory,
        critic_encoder_factory=encoder_factory,
        imitator_encoder_factory=encoder_factory,
        q_func_factory=create_q_func_factory(q_func_factory),
        gamma=gamma,
        tau=tau,
        n_critics=n_critics,
        initial_temperature=initial_temperature,
        initial_alpha=initial_alpha,
        alpha_threshold=alpha_threshold,
        lam=lam,
        n_action_samples=n_action_samples,
        mmd_kernel=mmd_kernel,
        mmd_sigma=mmd_sigma,
        use_gpu=None,
        scaler=scaler,
        action_scaler=action_scaler,
    )
    impl.build()

    x = torch.rand(32, *observation_shape)
    target = impl.compute_target(x)
    if q_func_factory == "mean":
        assert target.shape == (32, 1)
    else:
        n_quantiles = impl._q_func.q_funcs[0]._n_quantiles
        assert target.shape == (32, n_quantiles)

    torch_impl_tester(
        impl, discrete=False, deterministic_best_action=q_func_factory != "iqn"
    )
