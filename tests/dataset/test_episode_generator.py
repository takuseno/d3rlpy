import re

import numpy as np
import pytest

from d3rlpy.dataset import EpisodeGenerator
from d3rlpy.types import Float32NDArray, Shape

from ..testing_utils import create_observations


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [1000])
@pytest.mark.parametrize(
    "episode_end_type", ["terminal", "truncated", "overlap"]
)
def test_episode_generator(
    observation_shape: Shape,
    action_size: int,
    length: int,
    episode_end_type: str,
) -> None:
    observations = create_observations(observation_shape, length)
    actions = np.random.random((length, action_size))
    rewards: Float32NDArray = np.random.random((length, 1)).astype(np.float32)
    terminals: Float32NDArray = np.zeros(length, dtype=np.float32)
    timeouts: Float32NDArray = np.zeros(length, dtype=np.float32)
    for i in range(length // 100):
        if episode_end_type == "terminal":
            terminals[(i + 1) * 100 - 1] = 1.0
            terminal = True
        else:
            terminal = False
        if episode_end_type == "truncated" or episode_end_type == "overlap":
            timeouts[(i + 1) * 100 - 1] = 1.0

    episode_generator = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )

    episodes = episode_generator()
    assert len(episodes) == length // 100

    for episode in episodes:
        assert len(episode) == 100
        if isinstance(observation_shape[0], tuple):
            for i, shape in enumerate(observation_shape):
                assert isinstance(shape, tuple)
                assert episode.observations[i].shape == (100, *shape)
        else:
            assert isinstance(episode.observations, np.ndarray)
            assert episode.observations.shape == (100, *observation_shape)
        assert episode.actions.shape == (100, action_size)
        assert episode.rewards.shape == (100, 1)
        assert episode.terminated == terminal


def test_episode_generator_raises_on_no_termination() -> None:
    observations = create_observations((4,), 100)
    actions = np.zeros((100, 2))
    rewards: Float32NDArray = np.zeros((100, 1), dtype=np.float32)
    terminals = np.zeros(100, dtype=np.float32)
    timeouts = np.zeros(100, dtype=np.float32)

    expected_msg = (
        "No episode termination was found. "
        "Either terminals or timeouts must include non-zero values."
    )

    with pytest.raises(AssertionError, match=re.escape(expected_msg)):
        EpisodeGenerator(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts,
        )
