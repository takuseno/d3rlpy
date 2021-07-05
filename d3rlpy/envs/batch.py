# pylint: disable=arguments-differ
import os
import tempfile
import uuid
from multiprocessing import Process, get_context
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, List, Sequence, Tuple

import cloudpickle
import gym
import numpy as np

from ..online.utility import get_action_size_from_env


def _subproc(conn: Connection, remote_conn: Connection, fn_path: str) -> None:
    remote_conn.close()

    with open(fn_path, "rb") as f:
        env = cloudpickle.load(f)()

    # notify if it's ready
    conn.send("ready")

    while True:
        command = conn.recv()
        if command[0] == "step":
            observation, reward, terminal, info = env.step(command[1])
            conn.send([observation, reward, terminal, info])
        elif command[0] == "reset":
            conn.send([env.reset()])
        elif command[0] == "close":
            conn.close()
            break
        else:
            raise ValueError(f"invalid {command[0]}.")


class SubprocEnv:

    _conn: Connection
    _remote_conn: Connection
    _proc: Process

    def __init__(self, make_env_fn: Callable[..., gym.Env], dname: str):
        # pickle function
        fn_path = os.path.join(dname, str(uuid.uuid1()))
        with open(fn_path, "wb") as f:
            cloudpickle.dump(make_env_fn, f)

        # spawn process otherwise PyTorch raises error
        ctx = get_context("spawn")
        self._conn, self._remote_conn = ctx.Pipe(duplex=True)
        self._proc = ctx.Process(  # type: ignore
            target=_subproc,
            args=(self._remote_conn, self._conn, fn_path),
            daemon=True,
        )
        self._proc.start()
        self._remote_conn.close()

    def step_send(self, action: np.ndarray) -> None:
        self._conn.send(["step", action])

    def step_get(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return self._conn.recv()  # type: ignore

    def reset_send(self) -> None:
        self._conn.send(["reset"])

    def reset_get(self) -> np.ndarray:
        return self._conn.recv()[0]

    def wait_for_ready(self) -> bool:
        self._conn.recv()
        return True

    def close(self) -> None:
        self._conn.send(["close"])
        self._conn.close()
        self._proc.join()


class BatchEnv(gym.Env):  # type: ignore
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Returns batch of next observations, actions, rewards and infos.

        Args:
            action: batch action.

        Returns:
            batch of next data.

        """
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        """Initializes environments and returns batch of observations.

        Returns:
            batch of observations.

        """
        raise NotImplementedError

    def render(self, mode: str = "human") -> Any:
        raise NotImplementedError("BatchEnvWrapper does not support render.")

    @property
    def n_envs(self) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.n_envs

    def close(self) -> None:
        for env in self._envs:
            env.close()


class SyncBatchEnv(BatchEnv):
    """The environment wrapper for batch training with synchronized
    environments.

    Multiple environments are serially running. Basically, the computational
    cost is linearly increased depending on the number of environments.

    Args:
        envs (list(gym.Env)): a list of environments.

    """

    _envs: List[gym.Env]
    _observation_shape: Sequence[int]
    _action_size: int
    _prev_terminals: np.ndarray

    def __init__(self, envs: List[gym.Env]):
        self._envs = envs
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self._observation_shape = self.observation_space.shape
        self._action_size = get_action_size_from_env(envs[0])
        self._prev_terminals = np.ones(len(self._envs))

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        n_envs = len(self._envs)
        is_image = len(self._observation_shape) == 3
        observations = np.empty(
            (n_envs,) + tuple(self._observation_shape),
            dtype=np.uint8 if is_image else np.float32,
        )
        rewards = np.empty(n_envs, dtype=np.float32)
        terminals = np.empty(n_envs, dtype=np.float32)
        infos = []
        info: Dict[str, Any]
        for i, (env, act) in enumerate(zip(self._envs, action)):
            if self._prev_terminals[i]:
                observation = env.reset()
                reward, terminal, info = 0.0, 0.0, {}
            else:
                observation, reward, terminal, info = env.step(act)
            observations[i] = observation
            rewards[i] = reward
            terminals[i] = terminal
            infos.append(info)
            self._prev_terminals[i] = terminal
        return observations, rewards, terminals, infos

    def reset(self) -> np.ndarray:
        n_envs = len(self._envs)
        is_image = len(self._observation_shape) == 3
        observations = np.empty(
            (n_envs,) + tuple(self._observation_shape),
            dtype=np.uint8 if is_image else np.float32,
        )

        for i, env in enumerate(self._envs):
            observations[i] = env.reset()

        self._prev_terminals = np.ones(len(self._envs))

        return observations

    @property
    def n_envs(self) -> int:
        return len(self._envs)


class AsyncBatchEnv(BatchEnv):
    """The environment wrapper for batch training with asynchronous environment
    workers.

    Multiple environments are running in different processes to maximize the
    computational efficiency.
    Ideally, you can scale the training linearly up to the number of CPUs.

    Args:
        make_env_fns (list(callable)): a list of callable functions to return an environment.

    """

    _envs: List[SubprocEnv]
    _observation_shape: Sequence[int]
    _action_size: int
    _prev_terminals: np.ndarray

    def __init__(self, make_env_fns: List[Callable[..., gym.Env]]):
        # start multiprocesses
        with tempfile.TemporaryDirectory() as dname:
            self._envs = []
            for make_env in make_env_fns:
                self._envs.append(SubprocEnv(make_env, dname))

            # make sure that all environements are created
            for env in self._envs:
                env.wait_for_ready()

        ref_env = make_env_fns[0]()
        self.observation_space = ref_env.observation_space
        self.action_space = ref_env.action_space
        self._observation_shape = self.observation_space.shape
        self._action_size = get_action_size_from_env(ref_env)
        self._prev_terminals = np.ones(len(self._envs))

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        n_envs = len(self._envs)
        is_image = len(self._observation_shape) == 3
        observations = np.empty(
            (n_envs,) + tuple(self._observation_shape),
            dtype=np.uint8 if is_image else np.float32,
        )
        rewards = np.empty(n_envs, dtype=np.float32)
        terminals = np.empty(n_envs, dtype=np.float32)
        infos = []

        # asynchronous environment step
        for i, (env, act) in enumerate(zip(self._envs, action)):
            if self._prev_terminals[i]:
                env.reset_send()
            else:
                env.step_send(act)

        # get the result through pipes
        info: Dict[str, Any]
        for i, env in enumerate(self._envs):
            if self._prev_terminals[i]:
                observation = env.reset_get()
                reward, terminal, info = 0.0, 0.0, {}
            else:
                observation, reward, terminal, info = env.step_get()
            observations[i] = observation
            rewards[i] = reward
            terminals[i] = terminal
            infos.append(info)
            self._prev_terminals[i] = terminal
        return observations, rewards, terminals, infos

    def reset(self) -> np.ndarray:
        n_envs = len(self._envs)
        is_image = len(self._observation_shape) == 3
        observations = np.empty(
            (n_envs,) + tuple(self._observation_shape),
            dtype=np.uint8 if is_image else np.float32,
        )

        # asynchronous step
        for env in self._envs:
            env.reset_send()

        # get the result through pipes
        for i, env in enumerate(self._envs):
            observations[i] = env.reset_get()

        self._prev_terminals = np.ones(len(self._envs))

        return observations

    @property
    def n_envs(self) -> int:
        return len(self._envs)
