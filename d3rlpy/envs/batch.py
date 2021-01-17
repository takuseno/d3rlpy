# pylint: disable=arguments-differ

from typing import Any, Callable, Dict, List, Sequence, Tuple
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection

import numpy as np
import gym
from gym.spaces import Discrete

from ..online.utility import get_action_size_from_env


class SubprocEnv:

    _conn: Connection
    _remote_conn: Connection
    _proc: Process

    def __init__(self, make_env_fn: Callable[..., gym.Env]):
        self._conn, self._remote_conn = Pipe()
        self._proc = Process(
            target=self._subproc, args=(make_env_fn, self._remote_conn)
        )
        self._proc.start()

    def _subproc(
        self, make_env_fn: Callable[..., gym.Env], conn: Connection
    ) -> None:
        env = make_env_fn()
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

    def step_send(self, action: np.ndarray) -> None:
        self._conn.send(["step", action])

    def step_get(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return self._conn.recv()  # type: ignore

    def reset_send(self) -> None:
        self._conn.send(["reset"])

    def reset_get(self) -> np.ndarray:
        return self._conn.recv()[0]

    def close(self) -> None:
        self._conn.send(["close"])
        self._conn.close()
        self._proc.join()


class BatchEnvWrapper(gym.Env):  # type: ignore
    """The environment wrapper for batch training.

    Multiple environments are running in different processes to maximize the
    computational efficiency.
    Ideally, you can scale the training linearly up to the number of CPUs.

    Args:
        make_env_fns: a list of callable functions to return an environment.

    """

    _envs: List[SubprocEnv]
    _observation_shape: Sequence[int]
    _action_size: int
    _discrete_action: bool
    _prev_terminals: np.ndarray

    def __init__(self, make_env_fns: List[Callable[..., gym.Env]]):
        self._envs = [SubprocEnv(make_env) for make_env in make_env_fns]
        ref_env = make_env_fns[0]()
        self.observation_space = ref_env.observation_space
        self.action_space = ref_env.action_space
        self._observation_shape = self.observation_space.shape
        self._action_size = get_action_size_from_env(ref_env)
        self._discrete_action = isinstance(self.action_space, Discrete)
        self._prev_terminals = np.ones(len(self._envs))

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Returns batch of next observations, actions, rewards and infos.

        Args:
            actions: batch action.

        Returns:
            batch of next data.

        """
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
        for i, action in enumerate(actions):
            if self._prev_terminals[i]:
                self._envs[i].reset_send()
            else:
                self._envs[i].step_send(action)

        # get the result through pipes
        info: Dict[str, Any]
        for i, action in enumerate(actions):
            if self._prev_terminals[i]:
                observation = self._envs[i].reset_get()
                reward, terminal, info = 0.0, 0.0, {}
            else:
                observation, reward, terminal, info = self._envs[i].step_get()
            observations[i] = observation
            rewards[i] = reward
            terminals[i] = terminal
            infos.append(info)
            self._prev_terminals[i] = terminal
        return observations, rewards, terminals, infos

    def reset(self) -> np.ndarray:
        """Initializes environments and returns batch of observations.

        Returns:
            batch of observations.

        """
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

    def render(self, mode: str = "human") -> Any:
        raise NotImplementedError("BatchEnvWrapper does not support render.")

    @property
    def n_envs(self) -> int:
        return len(self._envs)

    def __len__(self) -> int:
        return self.n_envs

    def __del__(self) -> None:
        for env in self._envs:
            env.close()
