from d3rlpy.misc import IncrementalCounter


def test_incremental_counter() -> None:
    counter = IncrementalCounter()

    assert counter.get_value() == 0
    assert counter.increment() == 1
    assert counter.get_value() == 1
