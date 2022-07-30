import numpy as np
import pytest

from d3rlpy.dataset import ExperienceWriter, InfiniteBuffer

from ..testing_utils import create_observation


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("terminated", [True, False])
def test_episode_writer(observation_shape, action_size, length, terminated):
    buffer = InfiniteBuffer()
    writer = ExperienceWriter(buffer)

    for _ in range(length):
        writer.write(
            observation=create_observation(observation_shape),
            action=np.random.random(action_size),
            reward=np.random.random(),
        )
    writer.clip_episode(terminated)

    if terminated:
        assert buffer.transition_count == length
    else:
        assert buffer.transition_count == length - 1
    episode = buffer.episodes[0]
    assert tuple(episode.observation_shape) == observation_shape
