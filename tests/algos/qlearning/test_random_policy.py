from typing import Sequence

import numpy as np
import pytest

from d3rlpy.algos.qlearning.random_policy import (
    DiscreteRandomPolicyConfig,
    RandomPolicyConfig,
)


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize("action_size", [4])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("observation_shape", [(100,)])
def test_random_policy(
    distribution: str,
    action_size: int,
    batch_size: int,
    observation_shape: Sequence[int],
) -> None:
    config = RandomPolicyConfig(distribution=distribution)
    algo = config.create()
    algo.create_impl(observation_shape, action_size)

    x = np.random.random((batch_size, *observation_shape))

    # check predict
    action = algo.predict(x)
    assert action.shape == (batch_size, action_size)

    # check sample_action
    action = algo.sample_action(x)
    assert action.shape == (batch_size, action_size)


@pytest.mark.parametrize("action_size", [4])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("observation_shape", [(100,)])
def test_discrete_random_policy(
    action_size: int, batch_size: int, observation_shape: Sequence[int]
) -> None:
    algo = DiscreteRandomPolicyConfig().create()
    algo.create_impl(observation_shape, action_size)

    x = np.random.random((batch_size, *observation_shape))

    # check predict
    action = algo.predict(x)
    assert action.shape == (batch_size,)
    assert np.all(action < action_size)

    # check sample_action
    action = algo.sample_action(x)
    assert action.shape == (batch_size,)
    assert np.all(action < action_size)
