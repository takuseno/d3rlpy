import gym
import pytest

from d3rlpy.algos import DQN, SAC
from d3rlpy.envs import AsyncBatchEnv, ChannelFirst
from d3rlpy.online.buffers import BatchReplayBuffer, ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy

from ..dummy_env import DummyAtari


def test_fit_online_cartpole_with_dqn():
    env = gym.make("CartPole-v0")
    eval_env = gym.make("CartPole-v0")

    algo = DQN()

    buffer = ReplayBuffer(1000, env)

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
    import d4rl_atari

    env = ChannelFirst(DummyAtari())
    eval_env = ChannelFirst(DummyAtari())

    algo = DQN(n_frames=4)

    buffer = ReplayBuffer(1000, env)

    explorer = LinearDecayEpsilonGreedy()

    algo.fit_online(
        env,
        buffer,
        explorer,
        n_steps=100,
        eval_env=eval_env,
        logdir="test_data",
    )

    assert algo.impl.observation_shape == (4, 84, 84)


def test_fit_online_pendulum_with_sac():
    env = gym.make("Pendulum-v0")
    eval_env = gym.make("Pendulum-v0")

    algo = SAC()

    buffer = ReplayBuffer(1000, env)

    algo.fit_online(
        env,
        buffer,
        n_steps=500,
        eval_env=eval_env,
        logdir="test_data",
    )


@pytest.mark.parametrize("deterministic", [False, True])
def test_collect_pendulum_with_sac(deterministic):
    env = gym.make("Pendulum-v0")

    algo = SAC()

    buffer = algo.collect(env, n_steps=500, deterministic=deterministic)

    assert buffer.size() > 490 and buffer.size() < 500


def test_collect_atari_with_dqn():
    import d4rl_atari

    env = ChannelFirst(DummyAtari())

    algo = DQN(n_frames=4)

    explorer = LinearDecayEpsilonGreedy()

    buffer = algo.collect(env, explorer=explorer, n_steps=100)

    assert algo.impl.observation_shape == (4, 84, 84)
    assert buffer.size() > 90 and buffer.size() < 100


@pytest.mark.parametrize("timelimit_aware", [False, True])
def test_timelimit_aware(timelimit_aware):
    env = gym.make("Pendulum-v0")

    algo = SAC()

    buffer = ReplayBuffer(1000, env)

    algo.fit_online(
        env,
        buffer,
        n_steps=500,
        logdir="test_data",
        timelimit_aware=timelimit_aware,
    )

    terminal_count = 0
    for i in range(len(buffer)):
        terminal_count += int(buffer.transitions[i].terminal)

    if timelimit_aware:
        assert terminal_count == 0
    else:
        assert terminal_count > 0


def test_fit_batch_online_cartpole_with_dqn():
    make_env = lambda: gym.make("CartPole-v0")
    env = AsyncBatchEnv([make_env for _ in range(5)])
    eval_env = gym.make("CartPole-v0")

    algo = DQN()

    buffer = BatchReplayBuffer(1000, env)

    explorer = LinearDecayEpsilonGreedy()

    algo.fit_batch_online(
        env,
        buffer,
        explorer,
        n_epochs=1,
        n_steps_per_epoch=500,
        n_updates_per_epoch=1,
        eval_env=eval_env,
        logdir="test_data",
    )


@pytest.mark.skip(reason="This test seems to not be working in tests.")
def test_fit_batch_online_atari_with_dqn():
    import d4rl_atari

    make_env = lambda: ChannelFirst(DummyAtari())
    env = AsyncBatchEnv([make_env for _ in range(2)])
    eval_env = ChannelFirst(DummyAtari())

    algo = DQN(n_frames=4)

    buffer = BatchReplayBuffer(1000, env)

    explorer = LinearDecayEpsilonGreedy()

    algo.fit_batch_online(
        env,
        buffer,
        explorer,
        n_epochs=1,
        n_steps_per_epoch=500,
        n_updates_per_epoch=1,
        eval_env=eval_env,
        logdir="test_data",
    )

    assert algo.impl.observation_shape == (4, 84, 84)


def test_fit_batch_online_pendulum_with_sac():
    make_env = lambda: gym.make("Pendulum-v0")
    env = AsyncBatchEnv([make_env for _ in range(5)])
    eval_env = gym.make("Pendulum-v0")

    algo = SAC()

    buffer = BatchReplayBuffer(1000, env)

    algo.fit_batch_online(
        env,
        buffer,
        n_epochs=1,
        n_steps_per_epoch=500,
        n_updates_per_epoch=1,
        eval_env=eval_env,
        logdir="test_data",
    )
