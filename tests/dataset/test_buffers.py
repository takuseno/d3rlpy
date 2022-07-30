import pytest

from d3rlpy.dataset import FIFOBuffer, InfiniteBuffer

from ..testing_utils import create_episode


@pytest.mark.parametrize("observation_shape", [(4,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("terminated", [False, True])
def test_infinite_buffer(observation_shape, action_size, length, terminated):
    buffer = InfiniteBuffer()

    for i in range(10):
        episode = create_episode(
            observation_shape, action_size, length, terminated=terminated
        )
        buffer.append(episode)
        assert len(buffer) == i + 1
        if terminated:
            assert buffer.transition_count == (i + 1) * length
        else:
            assert buffer.transition_count == (i + 1) * (length - 1)


@pytest.mark.parametrize("observation_shape", [(4,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("limit", [500])
@pytest.mark.parametrize("terminated", [False, True])
def test_fifo_buffer(observation_shape, action_size, length, limit, terminated):
    buffer = FIFOBuffer(limit)

    for i in range(10):
        episode = create_episode(
            observation_shape, action_size, length, terminated=terminated
        )
        buffer.append(episode)
        if i >= 5:
            assert len(buffer) == 5
            if terminated:
                assert buffer.transition_count == limit
            else:
                assert buffer.transition_count == limit - 5
        else:
            assert len(buffer) == i + 1
            if terminated:
                assert buffer.transition_count == (i + 1) * length
            else:
                assert buffer.transition_count == (i + 1) * (length - 1)
