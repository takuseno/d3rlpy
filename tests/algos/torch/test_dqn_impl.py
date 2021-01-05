import pytest

from d3rlpy.augmentation import DrQPipeline
from d3rlpy.algos.torch.dqn_impl import DQNImpl, DoubleDQNImpl
from d3rlpy.optimizers import AdamFactory
from d3rlpy.encoders import DefaultEncoderFactory
from d3rlpy.q_functions import create_q_func_factory
from tests.algos.algo_test import torch_impl_tester, DummyScaler


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
def test_dqn_impl(
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
    impl = DQNImpl(
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
def test_double_dqn_impl(
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
    impl = DoubleDQNImpl(
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
