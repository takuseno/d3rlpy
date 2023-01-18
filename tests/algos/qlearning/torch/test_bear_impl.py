import pytest

from d3rlpy.algos.qlearning.torch.bear_impl import BEARImpl
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory, QRQFunctionFactory
from tests.algos.qlearning.algo_impl_test import impl_tester


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
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("tau", [0.05])
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("initial_temperature", [1.0])
@pytest.mark.parametrize("initial_alpha", [1.0])
@pytest.mark.parametrize("alpha_threshold", [0.05])
@pytest.mark.parametrize("lam", [0.75])
@pytest.mark.parametrize("n_action_samples", [100])
@pytest.mark.parametrize("n_target_samples", [10])
@pytest.mark.parametrize("n_mmd_action_samples", [4])
@pytest.mark.parametrize("mmd_kernel", ["laplacian"])
@pytest.mark.parametrize("mmd_sigma", [20.0])
@pytest.mark.parametrize("vae_kl_weight", [0.5])
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
    n_target_samples,
    n_mmd_action_samples,
    mmd_kernel,
    mmd_sigma,
    vae_kl_weight,
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
        q_func_factory=q_func_factory,
        gamma=gamma,
        tau=tau,
        n_critics=n_critics,
        initial_temperature=initial_temperature,
        initial_alpha=initial_alpha,
        alpha_threshold=alpha_threshold,
        lam=lam,
        n_action_samples=n_action_samples,
        n_target_samples=n_target_samples,
        n_mmd_action_samples=n_mmd_action_samples,
        mmd_kernel=mmd_kernel,
        mmd_sigma=mmd_sigma,
        vae_kl_weight=vae_kl_weight,
        device="cpu:0",
    )
    impl.build()

    impl_tester(impl, discrete=False)
