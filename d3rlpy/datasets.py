# pylint: disable=unused-import,too-many-return-statements

import os
import random
import re
from typing import Optional, Tuple
from urllib import request

import gym
import numpy as np

from .dataset import (
    Episode,
    EpisodeGenerator,
    InfiniteBuffer,
    ReplayBuffer,
    TrajectorySlicerProtocol,
    TransitionPickerProtocol,
    create_infinite_replay_buffer,
    load_v1,
)
from .envs import ChannelFirst

__all__ = [
    "DATA_DIRECTORY",
    "DROPBOX_URL",
    "CARTPOLE_URL",
    "CARTPOLE_RANDOM_URL",
    "PENDULUM_URL",
    "PENDULUM_RANDOM_URL",
    "get_cartpole",
    "get_pendulum",
    "get_atari",
    "get_atari_transitions",
    "get_d4rl",
    "get_dataset",
]

DATA_DIRECTORY = "d3rlpy_data"
DROPBOX_URL = "https://www.dropbox.com/s"
CARTPOLE_URL = f"{DROPBOX_URL}/uep0lzlhxpi79pd/cartpole_v1.1.0.h5?dl=1"
CARTPOLE_RANDOM_URL = f"{DROPBOX_URL}/4lgai7tgj84cbov/cartpole_random_v1.1.0.h5?dl=1"  # pylint: disable=line-too-long
PENDULUM_URL = f"{DROPBOX_URL}/ukkucouzys0jkfs/pendulum_v1.1.0.h5?dl=1"
PENDULUM_RANDOM_URL = f"{DROPBOX_URL}/hhbq9i6ako24kzz/pendulum_random_v1.1.0.h5?dl=1"  # pylint: disable=line-too-long


def get_cartpole(
    dataset_type: str = "replay",
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
) -> Tuple[ReplayBuffer, gym.Env]:
    """Returns cartpole dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/cartpole.h5`` if
    it does not exist.

    Args:
        dataset_type: dataset type. Available options are
            ``['replay', 'random']``.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.

    """
    if dataset_type == "replay":
        url = CARTPOLE_URL
        file_name = "cartpole_replay_v1.1.0.h5"
    elif dataset_type == "random":
        url = CARTPOLE_RANDOM_URL
        file_name = "cartpole_random_v1.1.0.h5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}.")

    data_path = os.path.join(DATA_DIRECTORY, file_name)

    # download dataset
    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Downloading cartpole.pkl into {data_path}...")
        request.urlretrieve(url, data_path)

    # load dataset
    with open(data_path, "rb") as f:
        episodes = load_v1(f)
    dataset = ReplayBuffer(
        InfiniteBuffer(),
        episodes=episodes,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
    )

    # environment
    env = gym.make("CartPole-v0")

    return dataset, env


def get_pendulum(
    dataset_type: str = "replay",
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
) -> Tuple[ReplayBuffer, gym.Env]:
    """Returns pendulum dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/pendulum.h5`` if
    it does not exist.

    Args:
        dataset_type: dataset type. Available options are
            ``['replay', 'random']``.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.

    """
    if dataset_type == "replay":
        url = PENDULUM_URL
        file_name = "pendulum_replay_v1.1.0.h5"
    elif dataset_type == "random":
        url = PENDULUM_RANDOM_URL
        file_name = "pendulum_random_v1.1.0.h5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}.")

    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading pendulum.pkl into {data_path}...")
        request.urlretrieve(url, data_path)

    # load dataset
    with open(data_path, "rb") as f:
        episodes = load_v1(f)
    dataset = ReplayBuffer(
        InfiniteBuffer(),
        episodes=episodes,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
    )

    # environment
    env = gym.make("Pendulum-v0")

    return dataset, env


def get_atari(
    env_name: str,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
) -> Tuple[ReplayBuffer, gym.Env]:
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
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.

    """
    try:
        import d4rl_atari  # type: ignore

        env = ChannelFirst(gym.make(env_name))
        episode_generator = EpisodeGenerator(**env.get_dataset())
        dataset = create_infinite_replay_buffer(
            episode_generator,
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )
        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n"
            "pip install git+https://github.com/takuseno/d4rl-atari"
        ) from e


def get_atari_transitions(
    game_name: str,
    fraction: float = 0.01,
    index: int = 0,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
) -> Tuple[ReplayBuffer, gym.Env]:
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
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.

    Returns:
        tuple of a list of :class:`d3rlpy.dataset.Transition` and gym
        environment.

    """
    try:
        import d4rl_atari

        # each epoch consists of 1M steps
        num_transitions_per_epoch = int(1000000 * fraction)

        copied_episodes = []
        for i in range(50):
            env = gym.make(
                f"{game_name}-epoch-{i + 1}-v{index}", sticky_action=True
            )
            episode_generator = EpisodeGenerator(**env.get_dataset())
            episodes = list(episode_generator())

            # copy episode data to release memory of unused data
            random.shuffle(episodes)
            num_data = 0
            for episode in episodes:
                if num_data >= num_transitions_per_epoch:
                    break

                copied_episode = Episode(
                    observations=episode.observations.copy(),  # type: ignore
                    actions=episode.actions.copy(),
                    rewards=episode.rewards.copy(),
                    terminated=episode.terminated,
                )

                # trim episode
                if num_data + copied_episode.size() > num_transitions_per_epoch:
                    end = num_transitions_per_epoch - num_data
                    copied_episode = Episode(
                        observations=copied_episode.observations[:end],
                        actions=copied_episode.actions[:end],
                        rewards=copied_episode.rewards[:end],
                        terminated=False,
                    )

                copied_episodes.append(copied_episode)
                num_data += copied_episode.size()

        dataset = ReplayBuffer(
            InfiniteBuffer(),
            episodes=copied_episodes,
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )

        return dataset, ChannelFirst(env)
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n"
            "pip install git+https://github.com/takuseno/d4rl-atari"
        ) from e


def get_d4rl(
    env_name: str,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
) -> Tuple[ReplayBuffer, gym.Env]:
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
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.

    """
    try:
        import d4rl  # type: ignore

        env = gym.make(env_name)
        dataset = env.get_dataset()

        observations = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        terminals = dataset["terminals"]
        timeouts = dataset["timeouts"]
        episode_terminals = np.logical_or(terminals, timeouts)

        episode_generator = EpisodeGenerator(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            terminals=np.array(terminals, dtype=np.float32),
            episode_terminals=np.array(episode_terminals, dtype=np.float32),
        )
        dataset = create_infinite_replay_buffer(
            episode_generator,
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )

        return dataset, env
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
    env_name: str,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
) -> Tuple[ReplayBuffer, gym.Env]:
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

       # d4rl-atari dataset
       dataset, env = d3rlpy.datasets.get_dataset('breakout-mixed-v0')

       # d4rl dataset
       dataset, env = d3rlpy.datasets.get_dataset('hopper-medium-v0')

    Args:
        env_name: environment id of the dataset.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.

    """
    if env_name == "cartpole-replay":
        return get_cartpole(
            dataset_type="replay",
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )
    elif env_name == "cartpole-random":
        return get_cartpole(
            dataset_type="random",
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )
    elif env_name == "pendulum-replay":
        return get_pendulum(
            dataset_type="replay",
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )
    elif env_name == "pendulum-random":
        return get_pendulum(
            dataset_type="random",
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )
    elif re.match(r"^bullet-.+$", env_name):
        return get_d4rl(
            env_name,
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )
    elif re.match(r"hopper|halfcheetah|walker|ant", env_name):
        return get_d4rl(
            env_name,
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )
    elif re.match(re.compile("|".join(ATARI_GAMES)), env_name):
        return get_atari(
            env_name,
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )
    raise ValueError(f"Unrecognized env_name: {env_name}.")
