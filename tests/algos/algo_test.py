import os
from unittest.mock import Mock

import numpy as np
import onnxruntime as ort
import torch

from d3rlpy.algos.torch.base import TorchImplBase
from d3rlpy.constants import ActionSpace
from d3rlpy.datasets import get_cartpole, get_pendulum
from d3rlpy.preprocessing import ActionScaler, ObservationScaler, RewardScaler
from tests.base_test import base_tester, base_update_tester


class DummyImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size):
        self._observation_shape = observation_shape
        self._action_size = action_size
        self.batch_size = 32

    def build(self):
        pass

    def save_model(self, fname):
        pass

    def load_model(self, fname):
        pass

    def save_policy(self, fname):
        pass

    def predict_best_action(self, x):
        return np.random.random((x.shape[0], self._action_size))

    def predict_value(self, x, action, with_std):
        pass

    def sample_action(self, x):
        return np.random.random((x.shape[0], self._action_size))

    @property
    def observation_shape(self):
        return self._observation_shape

    @property
    def action_size(self):
        return self._action_size


class DummyObservationScaler(ObservationScaler):
    def fit(self, episodes):
        pass

    def fit_with_env(self, env):
        pass

    def transform(self, x):
        return 0.1 * x

    def reverse_transform(self, x):
        return 10.0 * x

    def get_type(self):
        return "dummy"

    def get_params(self):
        return {}


class DummyActionScaler(ActionScaler):
    def fit(self, episodes):
        pass

    def fit_with_env(self, env):
        pass

    def transform(self, action):
        return 0.1 * action

    def reverse_transform(self, action):
        return 10.0 * action

    def get_type(self):
        return "dummy"

    def get_params(self):
        return {}


class DummyRewardScaler(RewardScaler):
    def fit(self, episodes):
        pass

    def fit_with_env(self, env):
        pass

    def transform(self, reward):
        return 0.1 * reward

    def reverse_transform(self, reward):
        return 10.0 * reward

    def get_type(self):
        return "dummy"

    def get_params(self):
        return {}


def algo_tester(
    algo,
    observation_shape,
    imitator=False,
    action_size=2,
    state_value=False,
    test_policy_copy=False,
    test_q_function_copy=False,
):
    # dummy impl object
    impl = DummyImpl(observation_shape, action_size)

    base_tester(algo, impl, observation_shape, action_size)

    algo._impl = impl

    # check save policy
    impl.save_policy = Mock()
    algo.save_policy("policy.pt")
    impl.save_policy.assert_called_with("policy.pt")

    # check predict
    x = np.random.random((2, 3)).tolist()
    ref_y = np.random.random((2, action_size)).tolist()
    impl.predict_best_action = Mock(return_value=ref_y)
    y = algo.predict(x)
    assert y == ref_y
    impl.predict_best_action.assert_called_with(x)

    # check predict_value
    if not imitator and not state_value:
        action = np.random.random((2, action_size)).tolist()
        ref_value = np.random.random((2, 3)).tolist()
        impl.predict_value = Mock(return_value=ref_value)
        value = algo.predict_value(x, action)
        assert value == ref_value
        impl.predict_value.assert_called_with(x, action, False)

    # check sample_action
    impl.sample_action = Mock(return_value=ref_y)
    try:
        y = algo.sample_action(x)
        assert y == ref_y
        impl.sample_action.assert_called_with(x)
    except NotImplementedError:
        pass

    algo.create_impl(observation_shape, action_size)

    if test_policy_copy:
        algo2 = algo.__class__(**algo.get_params())
        algo2.create_impl(observation_shape, action_size)
        algo.copy_policy_from(algo2)
        observations = np.random.random((100, *observation_shape))
        action1 = algo.predict(observations)
        action2 = algo.predict(observations)
        assert np.all(action1 == action2)

    if test_q_function_copy:
        algo2 = algo.__class__(**algo.get_params())
        algo2.create_impl(observation_shape, action_size)
        algo.copy_q_function_from(algo2)
        observations = np.random.random((100, *observation_shape))
        if algo.get_action_type() == ActionSpace.CONTINUOUS:
            actions = np.random.random((100, action_size))
        else:
            actions = np.random.randint(action_size, size=100)
        value1 = algo.predict_value(observations, actions)
        value2 = algo2.predict_value(observations, actions)
        assert np.all(value1 == value2)

    algo._impl = None


def algo_update_tester(
    algo,
    observation_shape,
    action_size,
    discrete=False,
    test_q_function_optim_copy=False,
    test_policy_optim_copy=False,
):
    base_update_tester(algo, observation_shape, action_size, discrete)

    if test_q_function_optim_copy:
        algo2 = algo.__class__(**algo.get_params())
        algo2.create_impl(observation_shape, action_size)
        assert not algo2.impl.q_function_optim.state
        algo2.copy_q_function_optim_from(algo)
        assert algo2.impl.q_function_optim.state

    if test_policy_optim_copy:
        algo2 = algo.__class__(**algo.get_params())
        algo2.create_impl(observation_shape, action_size)
        assert not algo2.impl.policy_optim.state
        algo2.copy_policy_optim_from(algo)
        assert algo2.impl.policy_optim.state


def algo_cartpole_tester(algo, n_evaluations=100, n_episodes=100, n_trials=3):
    # load dataset
    dataset, env = get_cartpole()

    # try multiple trials to reduce failures due to random seeds
    trial_count = 0
    for _ in range(n_trials):
        # reset parameters
        algo._impl = None

        # train
        algo.fit(
            dataset.episodes[:n_episodes],
            n_epochs=3,
            logdir="test_data",
            verbose=False,
        )

        # evaluation loop
        success_count = 0
        evaluation_count = 0
        while evaluation_count < n_evaluations:
            observation = env.reset()
            episode_rew = 0.0
            while True:
                action = algo.predict([observation])[0]
                observation, reward, done, _ = env.step(action)
                episode_rew += reward
                if done:
                    break
            evaluation_count += 1
            if episode_rew >= 160:
                success_count += 1

        if success_count >= n_evaluations * 0.8:
            break

        trial_count += 1
        if trial_count == n_trials:
            assert False, "performance is not good enough: %d." % success_count


def algo_pendulum_tester(algo, n_evaluations=100, n_episodes=500, n_trials=3):
    # load dataset
    dataset, env = get_pendulum()
    upper_bound = env.action_space.high

    # try multiple trials to reduce failures due to random seeds
    trial_count = 0
    for _ in range(n_trials):
        # reset parameters
        algo._impl = None

        # train
        algo.fit(
            dataset.episodes[:n_episodes],
            n_epochs=3,
            logdir="test_data",
            verbose=False,
        )

        # evaluation loop
        success_count = 0
        evaluation_count = 0
        while evaluation_count < n_evaluations:
            observation = env.reset()
            episode_rew = 0.0
            while True:
                action = algo.predict([observation])[0]
                observation, reward, done, _ = env.step(upper_bound * action)
                episode_rew += reward
                if done:
                    break
            evaluation_count += 1
            if episode_rew >= -600:
                success_count += 1

        if success_count >= n_evaluations * 0.8:
            break

        trial_count += 1
        if trial_count == n_trials:
            assert False, "performance is not good enough: %d." % success_count


def impl_tester(impl, discrete, imitator, test_with_std):
    # setup implementation
    impl.build()

    observations = np.random.random((100,) + impl.observation_shape)
    if discrete:
        actions = np.random.randint(impl.action_size, size=100)
    else:
        actions = np.random.random((100, impl.action_size))

    # check predict_best_action
    y = impl.predict_best_action(observations)
    if discrete:
        assert y.shape == (100,)
    else:
        assert y.shape == (100, impl.action_size)

    # check predict_values
    if not imitator:
        value = impl.predict_value(observations, actions, with_std=False)
        assert value.shape == (100,)

        if test_with_std:
            value, std = impl.predict_value(
                observations, actions, with_std=True
            )
            assert value.shape == (100,)
            assert std.shape == (100,)

    # check sample_action
    try:
        action = impl.sample_action(observations)
        if discrete:
            assert action.shape == (100,)
        else:
            assert action.shape == (100, impl.action_size)
    except NotImplementedError:
        pass


def torch_impl_tester(
    impl,
    discrete,
    deterministic_best_action=True,
    imitator=False,
    test_with_std=True,
):
    impl_tester(impl, discrete, imitator, test_with_std)

    # check save_model and load_model
    impl.save_model(os.path.join("test_data", "model.pt"))
    impl.load_model(os.path.join("test_data", "model.pt"))

    # check save_policy as TorchScript
    impl.save_policy(os.path.join("test_data", "model.pt"))
    policy = torch.jit.load(os.path.join("test_data", "model.pt"))
    observations = torch.rand(100, *impl.observation_shape)
    action = policy(observations)
    if discrete:
        assert action.shape == (100,)
    else:
        assert action.shape == (100, impl.action_size)

    # check output consistency between the full model and TorchScript
    # TODO: check probablistic policy
    # https://github.com/pytorch/pytorch/pull/25753
    if deterministic_best_action:
        action = action.detach().numpy()
        assert np.allclose(
            action, impl.predict_best_action(observations), atol=1e-6
        )

    # check save_policy as ONNX
    impl.save_policy(os.path.join("test_data", "model.onnx"))
    ort_session = ort.InferenceSession(os.path.join("test_data", "model.onnx"))
    observations = np.random.rand(1, *impl.observation_shape).astype("f4")
    action = ort_session.run(None, {"input_0": observations})[0]
    if discrete:
        assert action.shape == (1,)
    else:
        assert action.shape == (1, impl.action_size)

    # check output consistency between the full model and ONNX
    # TODO: check probablistic policy
    # https://github.com/pytorch/pytorch/pull/25753
    if deterministic_best_action:
        assert np.allclose(
            action, impl.predict_best_action(observations), atol=1e-6
        )
