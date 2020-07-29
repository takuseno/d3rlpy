import numpy as np
import os
import torch
import pickle

from unittest.mock import Mock
from tests.base_test import base_tester, base_update_tester
from tests.algos.algo_test import DummyScaler
from d3rlpy.dynamics.torch.base import TorchImplBase
from d3rlpy.dataset import MDPDataset, Transition, TransitionMiniBatch
from d3rlpy.logger import D3RLPyLogger
from d3rlpy.preprocessing import Scaler


class DummyImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size):
        self.observation_shape = observation_shape
        self.action_size = action_size

    def save_model(self, fname):
        pass

    def load_model(self, fname):
        pass

    def predict(self, x, action):
        pass

    def generate(self, x, action):
        pass


class DummyAlgo:
    def __init__(self, action_size):
        self.action_size = action_size

    def sample_action(self, x):
        return np.random.random((x.shape[0], self.action_size))


def dynamics_tester(dynamics, observation_shape, action_size=2):
    # dummy impl object
    impl = DummyImpl(observation_shape, action_size)

    base_tester(dynamics, impl, observation_shape, action_size)

    dynamics.impl = impl

    # check predict
    x = np.random.random((2, 3)).tolist()
    action = np.random.random((2, 3)).tolist()
    ref_y = np.random.random((2, 3)).tolist()
    ref_reward = np.random.random((2, 1)).tolist()
    impl.predict = Mock(return_value=(ref_y, ref_reward))
    y, reward = dynamics.predict(x, action)
    assert y == ref_y
    assert reward == ref_reward
    impl.predict.assert_called_with(x, action)


def dynamics_update_tester(dynamics,
                           observation_shape,
                           action_size,
                           discrete=False):
    transitions = base_update_tester(dynamics, observation_shape, action_size,
                                     discrete)

    # dummy algo
    algo = DummyAlgo(action_size)

    new_transitions = dynamics.generate(algo, transitions)
    assert len(new_transitions) == dynamics.horizon * dynamics.n_transitions


def impl_tester(impl, discrete):
    observations = np.random.random((100, ) + impl.observation_shape)
    if discrete:
        actions = np.random.randint(impl.action_size, size=100)
    else:
        actions = np.random.random((100, impl.action_size))

    # check predict
    y, rewards = impl.predict(observations, actions)
    assert y.shape == (100, ) + impl.observation_shape
    assert rewards.shape == (100, 1)

    # check generate
    y, rewards = impl.generate(observations, actions)
    assert y.shape == (100, ) + impl.observation_shape
    assert rewards.shape == (100, 1)


def torch_impl_tester(impl, discrete):
    impl_tester(impl, discrete)

    # check save_model and load_model
    impl.save_model(os.path.join('test_data', 'model.pt'))
    impl.load_model(os.path.join('test_data', 'model.pt'))
