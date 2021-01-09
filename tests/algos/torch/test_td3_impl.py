import pytest

from d3rlpy.algos.torch.td3_impl import TD3Impl
from d3rlpy.augmentation import DrQPipeline
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.q_functions import create_q_func_factory
from tests.algos.algo_test import torch_impl_tester, DummyScaler


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
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("bootstrap", [False])
@pytest.mark.parametrize("share_encoder", [False, True])
@pytest.mark.parametrize("target_smoothing_sigma", [0.2])
@pytest.mark.parametrize("target_smoothing_clip", [0.5])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("augmentation", [DrQPipeline()])
def test_td3_impl(
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
    bootstrap,
    share_encoder,
    target_smoothing_sigma,
    target_smoothing_clip,
    scaler,
    augmentation,
):
    impl = TD3Impl(
        observation_shape,
        action_size,
        actor_learning_rate,
        critic_learning_rate,
        actor_optim_factory,
        critic_optim_factory,
        encoder_factory,
        encoder_factory,
        create_q_func_factory(q_func_factory),
        gamma,
        tau,
        n_critics,
        bootstrap,
        share_encoder,
        target_smoothing_sigma,
        target_smoothing_clip,
        use_gpu=False,
        scaler=scaler,
        augmentation=augmentation,
    )
    torch_impl_tester(
        impl, discrete=False, deterministic_best_action=q_func_factory != "iqn"
    )
