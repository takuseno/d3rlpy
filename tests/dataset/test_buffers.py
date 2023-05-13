from typing import Sequence

import pytest

from d3rlpy.dataset import FIFOBuffer, InfiniteBuffer

from ..testing_utils import create_episode


@pytest.mark.parametrize("observation_shape", [(4,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("terminated", [False, True])
def test_infinite_buffer(
    observation_shape: Sequence[int],
    action_size: int,
    length: int,
    terminated: bool,
) -> None:
    buffer = InfiniteBuffer()

    for i in range(10):
        episode = create_episode(
            observation_shape, action_size, length, terminated=terminated
        )
        for j in range(episode.transition_count):
            buffer.append(episode, j)

        if terminated:
            assert buffer.transition_count == (i + 1) * (length)
        else:
            assert buffer.transition_count == (i + 1) * (length - 1)
        assert len(buffer.episodes) == i + 1


@pytest.mark.parametrize("observation_shape", [(4,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("limit", [500])
@pytest.mark.parametrize("terminated", [False, True])
def test_fifo_buffer(
    observation_shape: Sequence[int],
    action_size: int,
    length: int,
    limit: int,
    terminated: bool,
) -> None:
    buffer = FIFOBuffer(limit)

    for i in range(10):
        episode = create_episode(
            observation_shape, action_size, length, terminated=terminated
        )
        for j in range(episode.transition_count):
            buffer.append(episode, j)

        if i >= 5:
            assert buffer.transition_count == limit
            if terminated:
                assert len(buffer.episodes) == 5
            else:
                assert len(buffer.episodes) == 6
        else:
            if terminated:
                assert buffer.transition_count == (i + 1) * length
            else:
                assert buffer.transition_count == (i + 1) * (length - 1)
            assert len(buffer.episodes) == i + 1
