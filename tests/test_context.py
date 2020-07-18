from d3rlpy.context import get_parallel_flag, parallel, disable_parallel


def test_parallel():
    assert not get_parallel_flag()
    with parallel():
        assert get_parallel_flag()
        with disable_parallel():
            assert not get_parallel_flag()
        assert get_parallel_flag()
    assert not get_parallel_flag()

    with disable_parallel():
        assert not get_parallel_flag()
    assert not get_parallel_flag()
