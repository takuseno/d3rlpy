from contextlib import contextmanager

PARALLEL_FLAG = False


def get_parallel_flag():
    return PARALLEL_FLAG


@contextmanager
def parallel():
    global PARALLEL_FLAG
    _prev_flag = PARALLEL_FLAG
    PARALLEL_FLAG = True
    yield
    PARALLEL_FLAG = _prev_flag


@contextmanager
def disable_parallel():
    global PARALLEL_FLAG
    _prev_flag = PARALLEL_FLAG
    PARALLEL_FLAG = False
    yield
    PARALLEL_FLAG = _prev_flag
