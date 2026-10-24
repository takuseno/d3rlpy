import numpy as np
import pytest

from d3rlpy.dataset import Episode, FIFOBuffer


@pytest.mark.parametrize("terminated", [False, True])
def test_fifo_buffer_retains_latest_transition_at_capacity_one(
    terminated: bool,
) -> None:
    episode = Episode(
        observations=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        actions=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        rewards=np.ones((3, 1), dtype=np.float32),
        terminated=terminated,
    )
    next_episode = Episode(
        observations=np.array([[3.0], [4.0], [5.0]], dtype=np.float32),
        actions=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        rewards=np.ones((3, 1), dtype=np.float32),
        terminated=terminated,
    )
    buffer = FIFOBuffer(limit=1)

    buffer.append(episode, 0)
    buffer.append(episode, 1)

    assert buffer.transition_count == 1
    assert buffer[0][0] is episode
    assert buffer[0][1] == 1
    assert len(buffer.episodes) == 1
    assert buffer.episodes[0] is episode

    buffer.append(next_episode, 0)

    assert buffer.transition_count == 1
    assert buffer[0][0] is next_episode
    assert buffer[0][1] == 0
    assert len(buffer.episodes) == 1
    assert buffer.episodes[0] is next_episode
