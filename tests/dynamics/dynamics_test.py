import os
import pickle
from unittest.mock import Mock

import numpy as np
import torch

from d3rlpy.dataset import MDPDataset, Transition, TransitionMiniBatch
from d3rlpy.dynamics.torch.base import TorchImplBase
from d3rlpy.logger import D3RLPyLogger
from d3rlpy.preprocessing import Scaler
from tests.algos.algo_test import DummyScaler
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


def dynamics_tester(dynamics, observation_shape, action_size=2):
    # dummy impl object
    impl = DummyImpl(observation_shape, action_size)

    base_tester(dynamics, impl, observation_shape, action_size)

    dynamics._impl = impl

    # check predict
    x = np.random.random((2, 3)).tolist()
    action = np.random.random((2, 3)).tolist()
    ref_y = np.random.random((2, 3)).tolist()
    ref_reward = np.random.random((2, 1)).tolist()
    ref_variance = np.random.random((2, 1)).tolist()
    impl.predict = Mock(return_value=(ref_y, ref_reward, ref_variance))
    y, reward = dynamics.predict(x, action)
    assert y == ref_y
    assert reward == ref_reward
    impl.predict.assert_called_with(x, action)

    # check with_variance
    y, reward, variance = dynamics.predict(x, action, with_variance=True)
    assert variance == ref_variance


def dynamics_update_tester(
    dynamics, observation_shape, action_size, discrete=False
):
    transitions = base_update_tester(
        dynamics, observation_shape, action_size, discrete
    )


def impl_tester(impl, discrete):
    observations = np.random.random((100,) + impl.observation_shape)
    if discrete:
        actions = np.random.randint(impl.action_size, size=100)
    else:
        actions = np.random.random((100, impl.action_size))

    # check predict
    y, rewards, variance = impl.predict(observations, actions)
    assert y.shape == (100,) + impl.observation_shape
    assert rewards.shape == (100, 1)
    assert variance.shape == (100, 1)


def torch_impl_tester(impl, discrete):
    impl_tester(impl, discrete)

    # check save_model and load_model
    impl.save_model(os.path.join("test_data", "model.pt"))
    impl.load_model(os.path.join("test_data", "model.pt"))
