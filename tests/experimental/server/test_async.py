import time

from d3rlpy.experimental.server.async import dispatch, get, is_running, kill


def test_dispatch():
    def _func(a, b):
        return a + b

    uid = dispatch(_func, 1, 2)

    assert get(uid) == 3


def test_is_running():
    def _func():
        time.sleep(0.1)

    uid = dispatch(_func)

    assert is_running(uid)


def test_kill():
    def _func():
        time.sleep(1.0)

    uid = dispatch(_func)

    kill(uid)
    time.sleep(0.1)
    assert not is_running(uid)
