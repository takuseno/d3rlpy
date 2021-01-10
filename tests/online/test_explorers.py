import numpy as np
import pytest

from d3rlpy.online.explorers import ConstantEpsilonGreedy
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.online.explorers import NormalNoise


@pytest.mark.parametrize("action_size", [3])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("epsilon", [0.5])
def test_constant_epsilon_greedy(action_size, observation_shape, epsilon):
    explorer = ConstantEpsilonGreedy(epsilon)

    ref_x = np.random.random(observation_shape)
    ref_y = np.random.randint(action_size)

    class DummyAlgo:
        def predict(self, x):
            assert np.all(x[0] == ref_x)
            return [ref_y]

        @property
        def impl(self):
            return self

        @property
        def action_size(self):
            return action_size

    algo = DummyAlgo()

    # check sample
    for i in range(10):
        action = np.random.randint(action_size)
        if action != explorer.sample(algo, ref_x, 0):
            break
        elif i == 9:
            assert False


@pytest.mark.parametrize("action_size", [3])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("start_epsilon", [1.0])
@pytest.mark.parametrize("end_epsilon", [0.1])
@pytest.mark.parametrize("duration", [10])
def test_linear_decay_epsilon_greedy(
    action_size, observation_shape, start_epsilon, end_epsilon, duration
):
    explorer = LinearDecayEpsilonGreedy(start_epsilon, end_epsilon, duration)

    # check epsilon
    assert explorer.compute_epsilon(0) == start_epsilon
    assert explorer.compute_epsilon(duration) == end_epsilon
    base = start_epsilon - end_epsilon
    ref_epsilon = end_epsilon + base * (1.0 - 1.0 / duration)
    assert explorer.compute_epsilon(1) == ref_epsilon

    ref_x = np.random.random(observation_shape)
    ref_y = np.random.randint(action_size)

    class DummyAlgo:
        def predict(self, x):
            assert np.all(x[0] == ref_x)
            return [ref_y]

        @property
        def impl(self):
            return self

        @property
        def action_size(self):
            return action_size

    algo = DummyAlgo()

    # check sample
    for i in range(10):
        action = np.random.randint(action_size)
        if action != explorer.sample(algo, ref_x, 0):
            break
        elif i == 9:
            assert False


@pytest.mark.parametrize("action_size", [3])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("mean", [0.0])
@pytest.mark.parametrize("std", [0.1])
def test_normal_noise(action_size, observation_shape, mean, std):
    explorer = NormalNoise(mean, std)

    action = np.random.random(action_size)
    ref_x = np.random.random(observation_shape)
    ref_y = np.random.random(action_size)

    class DummyAlgo:
        def sample_action(self, x):
            assert np.all(x[0] == ref_x)
            return [ref_y]

    algo = DummyAlgo()

    assert not np.allclose(explorer.sample(algo, ref_x, 0), action)
