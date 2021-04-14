import pytest

from d3rlpy.containers import FIFOQueue


@pytest.mark.parametrize("maxlen", [100])
def test_fifo_queue(maxlen):
    queue = FIFOQueue(maxlen)

    # check append
    for i in range(2 * maxlen):
        queue.append(i)
        assert len(queue) == min(i + 1, maxlen)

    # check random access
    for i in range(maxlen):
        assert queue[i] == maxlen + i

    # check iterator
    i = 0
    for v in queue:
        assert v == maxlen + i
        i += 1


@pytest.mark.parametrize("maxlen", [100])
def test_fifo_queue_with_drop_callback(maxlen):
    count = 0

    def callback(value):
        assert value == count - 100

    queue = FIFOQueue(maxlen, drop_callback=callback)

    for i in range(2 * maxlen):
        queue.append(i)
        count += 1
