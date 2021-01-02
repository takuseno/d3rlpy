from contextlib import contextmanager
from typing import Iterator

PARALLEL_FLAG: bool = False


def get_parallel_flag() -> bool:
    return PARALLEL_FLAG


@contextmanager
def parallel() -> Iterator[None]:
    global PARALLEL_FLAG
    _prev_flag = PARALLEL_FLAG
    PARALLEL_FLAG = True
    yield
    PARALLEL_FLAG = _prev_flag


@contextmanager
def disable_parallel() -> Iterator[None]:
    global PARALLEL_FLAG
    _prev_flag = PARALLEL_FLAG
    PARALLEL_FLAG = False
    yield
    PARALLEL_FLAG = _prev_flag
