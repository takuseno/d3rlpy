import gym
import pytest

from d3rlpy.algos.qlearning import DQNConfig, SACConfig
from d3rlpy.algos.qlearning.explorers import LinearDecayEpsilonGreedy
from d3rlpy.dataset import InfiniteBuffer, ReplayBuffer
from d3rlpy.envs import ChannelFirst

from ...dummy_env import DummyAtari


def test_fit_online_cartpole_with_dqn():
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    algo = DQNConfig().create()

    buffer = ReplayBuffer(InfiniteBuffer())

    explorer = LinearDecayEpsilonGreedy()

    algo.fit_online(
        env,
        buffer,
        explorer,
        n_steps=100,
        eval_env=eval_env,
        logdir="test_data",
    )


def test_fit_online_atari_with_dqn():
    env = ChannelFirst(DummyAtari())
    eval_env = ChannelFirst(DummyAtari())

    algo = DQNConfig().create()

    buffer = ReplayBuffer(InfiniteBuffer())

    explorer = LinearDecayEpsilonGreedy()

    algo.fit_online(
        env,
        buffer,
        explorer,
        n_steps=100,
        eval_env=eval_env,
        logdir="test_data",
    )

    assert algo.impl.observation_shape == (1, 84, 84)


def test_fit_online_pendulum_with_sac():
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")

    algo = SACConfig().create()

    buffer = ReplayBuffer(InfiniteBuffer())

    algo.fit_online(
        env,
        buffer,
        n_steps=500,
        eval_env=eval_env,
        logdir="test_data",
    )


@pytest.mark.parametrize("deterministic", [False, True])
def test_collect_pendulum_with_sac(deterministic):
    env = gym.make("Pendulum-v1")

    algo = SACConfig().create()

    buffer = algo.collect(env, n_steps=500, deterministic=deterministic)

    assert buffer.transition_count > 490 and buffer.transition_count < 500


def test_collect_atari_with_dqn():
    env = ChannelFirst(DummyAtari())

    algo = DQNConfig().create()

    explorer = LinearDecayEpsilonGreedy()

    buffer = algo.collect(env, explorer=explorer, n_steps=100)

    assert algo.impl.observation_shape == (1, 84, 84)
    assert buffer.transition_count > 90 and buffer.transition_count < 100
