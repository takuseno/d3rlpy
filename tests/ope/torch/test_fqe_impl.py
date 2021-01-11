import pytest
import numpy as np
import os

from d3rlpy.ope.torch.fqe_impl import FQEImpl, DiscreteFQEImpl
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.q_functions import create_q_func_factory
from tests.algos.algo_test import DummyScaler, DummyActionScaler


def torch_impl_tester(impl, discrete):
    # setup implementation
    impl.build()

    observations = np.random.random((100,) + impl.observation_shape)
    if discrete:
        actions = np.random.randint(impl.action_size, size=100)
    else:
        actions = np.random.random((100, impl.action_size))

    # check predict_value
    value = impl.predict_value(observations, actions, with_std=False)
    assert value.shape == (100,)

    # check predict_value with standard deviation
    value, std = impl.predict_value(observations, actions, with_std=True)
    assert value.shape == (100,)
    assert std.shape == (100,)

    # check save_model and load_model
    impl.save_model(os.path.join("test_data", "model.pt"))
    impl.load_model(os.path.join("test_data", "model.pt"))


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [1e-3])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("bootstrap", [False])
@pytest.mark.parametrize("share_encoder", [True])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
def test_fqe_impl(
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
    action_scaler,
):
    fqe = FQEImpl(
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
        action_scaler=action_scaler,
    )

    torch_impl_tester(fqe, False)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [1e-3])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("bootstrap", [False])
@pytest.mark.parametrize("share_encoder", [True])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
def test_discrete_fqe_impl(
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
):
    fqe = DiscreteFQEImpl(
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
        action_scaler=None,
    )

    torch_impl_tester(fqe, True)
