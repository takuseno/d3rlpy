# pylint: disable=unused-import

import os
import pickle
import re
import urllib.request as request
from typing import Tuple

import gym
import numpy as np

from .dataset import MDPDataset
from .envs import ChannelFirst

DATA_DIRECTORY = "d3rlpy_data"
DROPBOX_URL = "https://www.dropbox.com/s"
CARTPOLE_URL = f"{DROPBOX_URL}/l1sdnq3zvoot2um/cartpole.h5?dl=1"
CARTPOLE_RANDOM_URL = f"{DROPBOX_URL}/rwf4pns5x0ku848/cartpole_random.h5?dl=1"
PENDULUM_URL = f"{DROPBOX_URL}/vsiz9pwvshj7sly/pendulum.h5?dl=1"
PENDULUM_RANDOM_URL = f"{DROPBOX_URL}/qldf2vjvvc5thsb/pendulum_random.h5?dl=1"


def get_cartpole(
    create_mask: bool = False, mask_size: int = 1, dataset_type: str = "replay"
) -> Tuple[MDPDataset, gym.Env]:
    """Returns cartpole dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/cartpole.h5`` if
    it does not exist.

    Args:
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.
        dataset_type: dataset type. Available options are
            ``['replay', 'random']``.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    if dataset_type == "replay":
        url = CARTPOLE_URL
        file_name = "cartpole_replay.h5"
    elif dataset_type == "random":
        url = CARTPOLE_RANDOM_URL
        file_name = "cartpole_random.h5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}.")

    data_path = os.path.join(DATA_DIRECTORY, file_name)

    # download dataset
    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print("Donwloading cartpole.pkl into %s..." % data_path)
        request.urlretrieve(url, data_path)

    # load dataset
    dataset = MDPDataset.load(
        data_path, create_mask=create_mask, mask_size=mask_size
    )

    # environment
    env = gym.make("CartPole-v0")

    return dataset, env


def get_pendulum(
    create_mask: bool = False,
    mask_size: int = 1,
    dataset_type: str = "replay",
) -> Tuple[MDPDataset, gym.Env]:
    """Returns pendulum dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/pendulum.h5`` if
    it does not exist.

    Args:
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.
        dataset_type: dataset type. Available options are
            ``['replay', 'random']``.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    if dataset_type == "replay":
        url = PENDULUM_URL
        file_name = "pendulum_replay.h5"
    elif dataset_type == "random":
        url = PENDULUM_RANDOM_URL
        file_name = "pendulum_random.h5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}.")

    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print("Donwloading pendulum.pkl into %s..." % data_path)
        request.urlretrieve(url, data_path)

    # load dataset
    dataset = MDPDataset.load(
        data_path, create_mask=create_mask, mask_size=mask_size
    )

    # environment
    env = gym.make("Pendulum-v0")

    return dataset, env


def get_pybullet(
    env_name: str, create_mask: bool = False, mask_size: int = 1
) -> Tuple[MDPDataset, gym.Env]:
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
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    try:
        import d4rl_pybullet  # type: ignore

        env = gym.make(env_name)
        dataset = MDPDataset(
            create_mask=create_mask, mask_size=mask_size, **env.get_dataset()
        )
        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-pybullet is not installed.\n"
            "pip install git+https://github.com/takuseno/d4rl-pybullet"
        ) from e


def get_atari(
    env_name: str, create_mask: bool = False, mask_size: int = 1
) -> Tuple[MDPDataset, gym.Env]:
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
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    try:
        import d4rl_atari  # type: ignore

        env = ChannelFirst(gym.make(env_name))
        dataset = MDPDataset(
            discrete_action=True,
            create_mask=create_mask,
            mask_size=mask_size,
            **env.get_dataset(),
        )
        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n"
            "pip install git+https://github.com/takuseno/d4rl-atari"
        ) from e


def get_d4rl(
    env_name: str, create_mask: bool = False, mask_size: int = 1
) -> Tuple[MDPDataset, gym.Env]:
    """Returns d4rl dataset and envrironment.

    The dataset is provided through d4rl.

    .. code-block:: python

        from d3rlpy.datasets import get_d4rl

        dataset, env = get_d4rl('hopper-medium-v0')

    References:
        * `Fu et al., D4RL: Datasets for Deep Data-Driven Reinforcement
          Learning. <https://arxiv.org/abs/2004.07219>`_
        * https://github.com/rail-berkeley/d4rl

    Args:
        env_name: environment id of d4rl dataset.
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    try:
        import d4rl  # type: ignore

        env = gym.make(env_name)
        dataset = env.get_dataset()

        observations = dataset["observations"][1:]
        actions = dataset["actions"][1:]
        rewards = dataset["rewards"][:-1]
        terminals = np.logical_and(
            dataset["terminals"][:-1], np.logical_not(dataset["timeouts"][:-1])
        )
        episode_terminals = np.logical_or(
            dataset["terminals"][:-1], dataset["timeouts"][:-1]
        )
        episode_terminals[-1] = 1.0

        mdp_dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            episode_terminals=episode_terminals,
            create_mask=create_mask,
            mask_size=mask_size,
        )

        return mdp_dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl is not installed.\n"
            "pip install git+https://github.com/rail-berkeley/d4rl"
        ) from e


ATARI_GAMES = [
    "adventure",
    "air-raid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bank-heist",
    "battle-zone",
    "beam-rider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "centipede",
    "chopper-command",
    "crazy-climber",
    "defender",
    "demon-attack",
    "double-dunk",
    "elevator-action",
    "enduro",
    "fishing-derby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "ice-hockey",
    "jamesbond",
    "journey-escape",
    "kangaroo",
    "krull",
    "kung-fu-master",
    "montezuma-revenge",
    "ms-pacman",
    "name-this-game",
    "phoenix",
    "pitfall",
    "pong",
    "pooyan",
    "private-eye",
    "qbert",
    "riverraid",
    "road-runner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "space-invaders",
    "star-gunner",
    "tennis",
    "time-pilot",
    "tutankham",
    "up-n-down",
    "venture",
    "video-pinball",
    "wizard-of-wor",
    "yars-revenge",
    "zaxxon",
]


def get_dataset(
    env_name: str, create_mask: bool = False, mask_size: int = 1
) -> Tuple[MDPDataset, gym.Env]:
    """Returns dataset and envrironment by guessing from name.

    This function returns dataset by matching name with the following datasets.

    - cartpole
    - pendulum
    - d4rl-pybullet
    - d4rl-atari
    - d4rl

    .. code-block:: python

       import d3rlpy

       # cartpole dataset
       dataset, env = d3rlpy.datasets.get_dataset('cartpole')

       # pendulum dataset
       dataset, env = d3rlpy.datasets.get_dataset('pendulum')

       # d4rl-pybullet dataset
       dataset, env = d3rlpy.datasets.get_dataset('hopper-bullet-mixed-v0')

       # d4rl-atari dataset
       dataset, env = d3rlpy.datasets.get_dataset('breakout-mixed-v0')

       # d4rl dataset
       dataset, env = d3rlpy.datasets.get_dataset('hopper-medium-v0')

    Args:
        env_name: environment id of the dataset.
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    if env_name == "cartpole":
        return get_cartpole(create_mask, mask_size)
    elif env_name == "pendulum":
        return get_pendulum(create_mask, mask_size)
    elif re.match(r"^bullet-.+$", env_name):
        return get_d4rl(env_name, create_mask, mask_size)
    elif re.match(r"^.+-bullet-.+$", env_name):
        return get_pybullet(env_name, create_mask, mask_size)
    elif re.match(r"hopper|halfcheetah|walker|ant", env_name):
        return get_d4rl(env_name, create_mask, mask_size)
    elif re.match(re.compile("|".join(ATARI_GAMES)), env_name):
        return get_atari(env_name, create_mask, mask_size)
    raise ValueError(f"Unrecognized env_name: {env_name}.")
