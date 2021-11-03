# pylint: disable=unused-import,too-many-return-statements

import os
import random
import re
from typing import List, Tuple
from urllib import request

import gym
import numpy as np

from .dataset import Episode, MDPDataset, Transition
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
        print(f"Donwloading cartpole.pkl into {data_path}...")
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
        print(f"Donwloading pendulum.pkl into {data_path}...")
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


def get_atari_transitions(
    game_name: str, fraction: float = 0.01, index: int = 0
) -> Tuple[List[Transition], gym.Env]:
    """Returns atari dataset as a list of Transition objects and envrironment.

    The dataset is provided through d4rl-atari.
    The difference from ``get_atari`` function is that this function will
    sample transitions from all epochs.
    This function is necessary for reproducing Atari experiments.

    .. code-block:: python

        from d3rlpy.datasets import get_atari_transitions

        # get 1% of transitions from all epochs (1M x 50 epoch x 1% = 0.5M)
        dataset, env = get_atari_transitions('breakout', fraction=0.01)

    References:
        * https://github.com/takuseno/d4rl-atari

    Args:
        game_name: Atari 2600 game name in lower_snake_case.
        fraction: fraction of sampled transitions.
        index: index to specify which trial to load.

    Returns:
        tuple of a list of :class:`d3rlpy.dataset.Transition` and gym
        environment.

    """
    try:
        import d4rl_atari

        # each epoch consists of 1M steps
        num_transitions_per_epoch = int(1000000 * fraction)

        transitions = []
        for i in range(50):
            env = gym.make(
                f"{game_name}-epoch-{i + 1}-v{index}", sticky_action=True
            )
            dataset = MDPDataset(discrete_action=True, **env.get_dataset())
            episodes = list(dataset.episodes)

            # copy episode data to release memory of unused data
            random.shuffle(episodes)
            num_data = 0
            copied_episodes = []
            for episode in episodes:
                copied_episode = Episode(
                    observation_shape=tuple(episode.get_observation_shape()),
                    action_size=episode.get_action_size(),
                    observations=episode.observations.copy(),
                    actions=episode.actions.copy(),
                    rewards=episode.rewards.copy(),
                    terminal=episode.terminal,
                )
                copied_episodes.append(copied_episode)

                num_data += len(copied_episode)
                if num_data > num_transitions_per_epoch:
                    break

            transitions_per_epoch = []
            for episode in copied_episodes:
                transitions_per_epoch += episode.transitions
            transitions += transitions_per_epoch[:num_transitions_per_epoch]

        return transitions, ChannelFirst(env)
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

        observations = []
        actions = []
        rewards = []
        terminals = []
        episode_terminals = []
        episode_step = 0
        cursor = 0
        dataset_size = dataset["observations"].shape[0]
        while cursor < dataset_size:
            # collect data for step=t
            observation = dataset["observations"][cursor]
            action = dataset["actions"][cursor]
            if episode_step == 0:
                reward = 0.0
            else:
                reward = dataset["rewards"][cursor - 1]

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminals.append(0.0)

            # skip adding the last step when timeout
            if dataset["timeouts"][cursor]:
                episode_terminals.append(1.0)
                episode_step = 0
                cursor += 1
                continue

            episode_terminals.append(0.0)
            episode_step += 1

            if dataset["terminals"][cursor]:
                # collect data for step=t+1
                dummy_observation = observation.copy()
                dummy_action = action.copy()
                next_reward = dataset["rewards"][cursor]

                # the last observation is rarely used
                observations.append(dummy_observation)
                actions.append(dummy_action)
                rewards.append(next_reward)
                terminals.append(1.0)
                episode_terminals.append(1.0)
                episode_step = 0

            cursor += 1

        mdp_dataset = MDPDataset(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            terminals=np.array(terminals, dtype=np.float32),
            episode_terminals=np.array(episode_terminals, dtype=np.float32),
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

    - cartpole-replay
    - cartpole-random
    - pendulum-replay
    - pendulum-random
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
    if env_name == "cartpole-replay":
        return get_cartpole(create_mask, mask_size, dataset_type="replay")
    elif env_name == "cartpole-random":
        return get_cartpole(create_mask, mask_size, dataset_type="random")
    elif env_name == "pendulum-replay":
        return get_pendulum(create_mask, mask_size, dataset_type="replay")
    elif env_name == "pendulum-random":
        return get_pendulum(create_mask, mask_size, dataset_type="random")
    elif re.match(r"^bullet-.+$", env_name):
        return get_d4rl(env_name, create_mask, mask_size)
    elif re.match(r"^.+-bullet-.+$", env_name):
        return get_pybullet(env_name, create_mask, mask_size)
    elif re.match(r"hopper|halfcheetah|walker|ant", env_name):
        return get_d4rl(env_name, create_mask, mask_size)
    elif re.match(re.compile("|".join(ATARI_GAMES)), env_name):
        return get_atari(env_name, create_mask, mask_size)
    raise ValueError(f"Unrecognized env_name: {env_name}.")
