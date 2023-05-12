from d3rlpy.itertools import first_flag, last_flag


def test_last_flag() -> None:
    x = [1, 2, 3, 4, 5]
    i = 0
    for is_last, value in last_flag(x):
        if i == len(x) - 1:
            assert is_last
        else:
            assert not is_last
        assert value == x[i]
        i += 1


def test_first_flag() -> None:
    x = [1, 2, 3, 4, 5]
    i = 0
    for is_first, value in first_flag(x):
        if i == 0:
            assert is_first
        else:
            assert not is_first
        assert value == x[i]
        i += 1
