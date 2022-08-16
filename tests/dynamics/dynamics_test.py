import os

import numpy as np

from d3rlpy.dynamics.torch.base import TorchImplBase
from tests.base_test import base_tester, base_update_tester


class DummyImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size):
        self._observation_shape = observation_shape
        self._action_size = action_size
        self.batch_size = 32
        self._device = "cpu:0"
        self._scaler = None

    def save_model(self, fname):
        pass

    def load_model(self, fname):
        pass

    def _predict(self, x, action):
        pass

    def _generate(self, x, action):
        pass

    @property
    def observation_shape(self):
        return self._observation_shape

    @property
    def action_size(self):
        return self._action_size

    @property
    def device(self):
        return self._device

    @property
    def scaler(self):
        return self._scaler


def dynamics_tester(
    dynamics, observation_shape, action_size=2, discrete_action=False
):
    # dummy impl object
    impl = DummyImpl(observation_shape, action_size)

    base_tester(dynamics, impl, observation_shape, action_size)

    dynamics.create_impl(observation_shape, action_size)

    # check predict
    x = np.random.random((2, *observation_shape))
    if discrete_action:
        action = np.random.randint(action_size, size=2)
    else:
        action = np.random.random((2, action_size))
    y, reward = dynamics.predict(x, action)
    assert y.shape == (2, *observation_shape)
    assert reward.shape == (2, 1)

    # check with_variance
    y, reward, variance = dynamics.predict(x, action, with_variance=True)
    assert variance.shape == (2, 1)


def dynamics_update_tester(
    dynamics, observation_shape, action_size, discrete=False
):
    transitions = base_update_tester(
        dynamics, observation_shape, action_size, discrete
    )


def impl_tester(impl, discrete, n_ensembles):
    observations = np.random.random((100,) + impl.observation_shape)
    if discrete:
        actions = np.random.randint(impl.action_size, size=100)
    else:
        actions = np.random.random((100, impl.action_size))

    # check predict without indices
    y, rewards, variance = impl.predict(observations, actions, None)
    assert y.shape == (100,) + impl.observation_shape
    assert rewards.shape == (100, 1)
    assert variance.shape == (100, 1)

    # check predict with indices
    indices = np.random.randint(n_ensembles, size=100)
    y, rewards, variance = impl.predict(observations, actions, indices)
    assert y.shape == (100,) + impl.observation_shape
    assert rewards.shape == (100, 1)
    assert variance.shape == (100, 1)


def torch_impl_tester(impl, discrete, n_ensembles):
    impl_tester(impl, discrete, n_ensembles)

    # check save_model and load_model
    impl.save_model(os.path.join("test_data", "model.pt"))
    impl.load_model(os.path.join("test_data", "model.pt"))
