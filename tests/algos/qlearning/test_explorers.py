from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pytest

from d3rlpy.algos.qlearning.explorers import (
    ConstantEpsilonGreedy,
    LinearDecayEpsilonGreedy,
    NormalNoise,
)
from d3rlpy.preprocessing import ActionScaler, MinMaxActionScaler


class DummyAlgo:
    def __init__(
        self,
        action_size: int,
        ref_x: np.ndarray,
        ref_y: np.ndarray,
        action_scaler: Optional[ActionScaler] = None,
    ):
        self.action_size = action_size
        self.ref_x = ref_x
        self.ref_y = ref_y
        self.action_scaler = action_scaler

    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        assert np.all(x == self.ref_x)
        return self.ref_y


@pytest.mark.parametrize("action_size", [3])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("epsilon", [0.5])
def test_constant_epsilon_greedy(
    action_size: int, observation_shape: Sequence[int], epsilon: float
) -> None:
    explorer = ConstantEpsilonGreedy(epsilon)

    ref_x = np.random.random((1, *observation_shape))
    ref_y = np.random.randint(action_size, size=(1,))

    algo = DummyAlgo(action_size, ref_x, ref_y)

    # check sample
    for i in range(10):
        action = np.random.randint(action_size, size=(1,))
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
    action_size: int,
    observation_shape: Sequence[int],
    start_epsilon: float,
    end_epsilon: float,
    duration: int,
) -> None:
    explorer = LinearDecayEpsilonGreedy(start_epsilon, end_epsilon, duration)

    # check epsilon
    assert explorer.compute_epsilon(0) == start_epsilon
    assert explorer.compute_epsilon(duration) == end_epsilon
    base = start_epsilon - end_epsilon
    ref_epsilon = end_epsilon + base * (1.0 - 1.0 / duration)
    assert explorer.compute_epsilon(1) == ref_epsilon

    ref_x = np.random.random((1, *observation_shape))
    ref_y = np.random.randint(action_size, size=(1,))

    algo = DummyAlgo(action_size, ref_x, ref_y)

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
def test_normal_noise(
    action_size: int, observation_shape: Sequence[int], mean: float, std: float
) -> None:
    explorer = NormalNoise(mean, std)

    ref_x = np.random.random((1, *observation_shape))
    ref_y = np.random.random((1, action_size))

    algo = DummyAlgo(action_size, ref_x, ref_y)

    assert not np.allclose(explorer.sample(algo, ref_x, 0), ref_y)


@pytest.mark.parametrize("action_size", [3])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("mean", [0.0])
@pytest.mark.parametrize("std", [0.1])
def test_normal_noise_with_scaler(
    action_size: int, observation_shape: Sequence[int], mean: float, std: float
) -> None:
    explorer = NormalNoise(mean, std)

    ref_x = np.random.random((1, *observation_shape))
    ref_y = 2.0 * np.array([[0.3, 0.6, 0.8]])

    action_scaler = MinMaxActionScaler(
        minimum=np.array([-2, -2, -2]), maximum=np.array([2, 2, 2])
    )

    algo = DummyAlgo(action_size, ref_x, ref_y, action_scaler)

    assert not np.allclose(explorer.sample(algo, ref_x, 0), ref_y)
    assert np.any(explorer.sample(algo, ref_x, 0) > 1.0)
