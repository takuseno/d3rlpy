import pytest

from d3rlpy.algos.torch.dqn_impl import DoubleDQNImpl, DQNImpl
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.q_functions import create_q_func_factory
from tests.algos.algo_test import (
    DummyRewardScaler,
    DummyScaler,
    torch_impl_tester,
)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [2.5e-4])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("target_reduction_type", ["min"])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("reward_scaler", [None, DummyRewardScaler()])
def test_dqn_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    n_critics,
    target_reduction_type,
    scaler,
    reward_scaler,
):
    impl = DQNImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        learning_rate=learning_rate,
        optim_factory=optim_factory,
        encoder_factory=encoder_factory,
        q_func_factory=create_q_func_factory(q_func_factory),
        gamma=gamma,
        n_critics=n_critics,
        target_reduction_type=target_reduction_type,
        use_gpu=None,
        scaler=scaler,
        reward_scaler=reward_scaler,
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
@pytest.mark.parametrize("target_reduction_type", ["min"])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("reward_scaler", [None, DummyRewardScaler()])
def test_double_dqn_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    n_critics,
    target_reduction_type,
    scaler,
    reward_scaler,
):
    impl = DoubleDQNImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        learning_rate=learning_rate,
        optim_factory=optim_factory,
        encoder_factory=encoder_factory,
        q_func_factory=create_q_func_factory(q_func_factory),
        gamma=gamma,
        n_critics=n_critics,
        target_reduction_type=target_reduction_type,
        use_gpu=None,
        scaler=scaler,
        reward_scaler=reward_scaler,
    )
    torch_impl_tester(
        impl, discrete=True, deterministic_best_action=q_func_factory != "iqn"
    )
