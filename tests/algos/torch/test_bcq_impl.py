import pytest
import torch

from d3rlpy.algos.torch.bcq_impl import BCQImpl, DiscreteBCQImpl
from d3rlpy.augmentation import DrQPipeline
from d3rlpy.optimizers import AdamFactory
from d3rlpy.encoders import DefaultEncoderFactory
from d3rlpy.q_functions import create_q_func_factory
from tests.algos.algo_test import torch_impl_tester, DummyScaler


@pytest.mark.parametrize("observation_shape", [(100,), (1, 48, 48)])
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
@pytest.mark.parametrize("bootstrap", [False])
@pytest.mark.parametrize("share_encoder", [False, True])
@pytest.mark.parametrize("lam", [0.75])
@pytest.mark.parametrize("n_action_samples", [10])  # small for test
@pytest.mark.parametrize("action_flexibility", [0.05])
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("beta", [0.5])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("augmentation", [DrQPipeline()])
def test_bcq_impl(
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
    bootstrap,
    share_encoder,
    lam,
    n_action_samples,
    action_flexibility,
    latent_size,
    beta,
    scaler,
    augmentation,
):
    impl = BCQImpl(
        observation_shape,
        action_size,
        actor_learning_rate,
        critic_learning_rate,
        imitator_learning_rate,
        actor_optim_factory,
        critic_optim_factory,
        imitator_optim_factory,
        encoder_factory,
        encoder_factory,
        encoder_factory,
        create_q_func_factory(q_func_factory),
        gamma,
        tau,
        n_critics,
        bootstrap,
        share_encoder,
        lam,
        n_action_samples,
        action_flexibility,
        latent_size,
        beta,
        use_gpu=False,
        scaler=scaler,
        augmentation=augmentation,
    )
    impl.build()

    # test internal methods
    x = torch.rand(32, *observation_shape)

    repeated_x = impl._repeat_observation(x)
    assert repeated_x.shape == (32, n_action_samples) + observation_shape

    action = impl._sample_action(repeated_x)
    assert action.shape == (32, n_action_samples, action_size)

    value = impl._predict_value(repeated_x, action)
    assert value.shape == (n_critics, 32 * n_action_samples, 1)

    target_values = impl._predict_value(repeated_x, action, target=True)
    if q_func_factory == "mean":
        assert target_values.shape == (n_critics, 32 * n_action_samples, 1)
    else:
        n_quantiles = impl.q_func.q_funcs[0].n_quantiles
        assert target_values.shape == (
            n_critics,
            32 * n_action_samples,
            n_quantiles,
        )

    target = impl.compute_target(x)
    if q_func_factory == "mean":
        assert target.shape == (32, 1)
    else:
        assert target.shape == (32, n_quantiles)

    best_action = impl._predict_best_action(x)
    assert best_action.shape == (32, action_size)

    torch_impl_tester(impl, discrete=False, deterministic_best_action=False)


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
@pytest.mark.parametrize("action_flexibility", [0.3])
@pytest.mark.parametrize("beta", [1e-2])
@pytest.mark.parametrize("scaler", [None])
@pytest.mark.parametrize("augmentation", [DrQPipeline()])
def test_discrete_bcq_impl(
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
    action_flexibility,
    beta,
    scaler,
    augmentation,
):
    impl = DiscreteBCQImpl(
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
        action_flexibility,
        beta,
        use_gpu=False,
        scaler=scaler,
        augmentation=augmentation,
    )
    torch_impl_tester(
        impl, discrete=True, deterministic_best_action=q_func_factory != "iqn"
    )
