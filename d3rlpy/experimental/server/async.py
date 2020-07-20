import uuid

from multiprocessing import Process, Queue

RUNNING_PROCESSES = {}


def _child_process(func, queue, args, kwargs):
    ret = func(*args, **kwargs)
    queue.put(ret)


def dispatch(func, *args, **kwargs):
    queue = Queue()
    process = Process(target=_child_process, args=(func, queue, args, kwargs))
    process.start()
    uid = uuid.uuid1()
    RUNNING_PROCESSES[uid] = (process, queue)
    return uid


def get(uid):
    _, queue = RUNNING_PROCESSES[uid]
    return queue.get()


def is_running(uid):
    process, _ = RUNNING_PROCESSES[uid]
    return process.is_alive()


def kill(uid):
    process, _ = RUNNING_PROCESSES[uid]
    process.terminate()
