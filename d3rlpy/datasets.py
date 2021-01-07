# pylint: disable=unused-import

import urllib.request as request
import os
import pickle
from typing import Tuple

import numpy as np
import gym

from .dataset import MDPDataset

DATA_DIRECTORY = "d3rlpy_data"
CARTPOLE_URL = "https://www.dropbox.com/s/2tmo7ul00268l3e/cartpole.pkl?dl=1"
PENDULUM_URL = "https://www.dropbox.com/s/90z7a84ngndrqt4/pendulum.pkl?dl=1"


def get_cartpole() -> Tuple[MDPDataset, gym.Env]:
    """Returns cartpole dataset and environment.

    The dataset is automatically downloaded to `d3rlpy_data/cartpole.pkl` if it
    does not exist.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    data_path = os.path.join(DATA_DIRECTORY, "cartpole.pkl")

    # download dataset
    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print("Donwloading cartpole.pkl into %s..." % data_path)
        request.urlretrieve(CARTPOLE_URL, data_path)

    # load dataset
    with open(data_path, "rb") as f:
        observations, actions, rewards, terminals = pickle.load(f)

    # environment
    env = gym.make("CartPole-v0")

    dataset = MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        discrete_action=True,
    )

    return dataset, env


def get_pendulum() -> Tuple[MDPDataset, gym.Env]:
    """Returns pendulum dataset and environment.

    The dataset is automatically downloaded to `d3rlpy_data/pendulum.pkl` if it
    does not exist.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    data_path = os.path.join(DATA_DIRECTORY, "pendulum.pkl")

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print("Donwloading pendulum.pkl into %s..." % data_path)
        request.urlretrieve(PENDULUM_URL, data_path)

    # load dataset
    with open(data_path, "rb") as f:
        observations, actions, rewards, terminals = pickle.load(f)

    # environment
    env = gym.make("Pendulum-v0")

    dataset = MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    return dataset, env


def get_pybullet(env_name: str) -> Tuple[MDPDataset, gym.Env]:
    """Returns pybullet dataset and envrironment.

    The dataset is provided through d4rl-pybullet. See more details including
    available dataset from its GitHub page.

    .. code-block:: python

        from d3rlpy.datasets import get_pybullet

        dataset, env = get_pybullet('hopper-bullet-mixed-v0')

    References:
        * https://github.com/takuseno/d4rl-pybullet

    Args:
        env_name: environment id of d4rl-pybullet dataset.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    try:
        import d4rl_pybullet  # type: ignore

        env = gym.make(env_name)
        dataset = MDPDataset(**env.get_dataset())
        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-pybullet is not installed.\n"
            "pip install git+https://github.com/takuseno/d4rl-pybullet"
        ) from e


def get_atari(env_name: str) -> Tuple[MDPDataset, gym.Env]:
    """Returns atari dataset and envrironment.

    The dataset is provided through d4rl-atari. See more details including
    available dataset from its GitHub page.

    .. code-block:: python

        from d3rlpy.datasets import get_atari

        dataset, env = get_atari('breakout-mixed-v0')

    References:
        * https://github.com/takuseno/d4rl-atari

    Args:
        env_name: environment id of d4rl-atari dataset.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    try:
        import d4rl_atari  # type: ignore

        env = gym.make(env_name, stack=False)
        dataset = MDPDataset(discrete_action=True, **env.get_dataset())
        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n"
            "pip install git+https://github.com/takuseno/d4rl-atari"
        ) from e
