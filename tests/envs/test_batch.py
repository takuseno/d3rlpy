import tempfile

import gym
import numpy as np
import pytest

from d3rlpy.envs import AsyncBatchEnv, SyncBatchEnv
from d3rlpy.envs.batch import SubprocEnv


def test_subproc_env():
    with tempfile.TemporaryDirectory() as dname:
        ref_env = gym.make("CartPole-v0")
        env = SubprocEnv(lambda: gym.make("CartPole-v0"), dname)
        env.wait_for_ready()

        env.reset_send()
        observation = env.reset_get()
        assert observation.shape == ref_env.observation_space.shape

        env.step_send(0)
        observation, reward, terminal, info = env.step_get()
        assert observation.shape == ref_env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminal, bool)
        assert isinstance(info, dict)

        env.close()


@pytest.mark.parametrize("n_envs", [5])
@pytest.mark.parametrize("n_steps", [1000])
def test_async_batch_env_discrete(n_envs, n_steps):
    make_env_fn = lambda: gym.make("CartPole-v0")
    env = AsyncBatchEnv([make_env_fn for _ in range(n_envs)])

    observation_shape = env.observation_space.shape
    action_size = env.action_space.n

    observations = env.reset()
    assert observations.shape == (n_envs,) + observation_shape

    for _ in range(n_steps):
        actions = np.random.randint(action_size, size=n_envs)

        observations, rewards, terminals, infos = env.step(actions)

        assert observations.shape == (n_envs,) + observation_shape
        assert rewards.shape == (n_envs,)
        assert terminals.shape == (n_envs,)
        assert len(infos) == n_envs


@pytest.mark.parametrize("n_envs", [5])
@pytest.mark.parametrize("n_steps", [1000])
def test_async_batch_env_continuous(n_envs, n_steps):
    make_env_fn = lambda: gym.make("Pendulum-v0")
    env = AsyncBatchEnv([make_env_fn for _ in range(n_envs)])

    observation_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]

    observations = env.reset()
    assert observations.shape == (n_envs,) + observation_shape

    for _ in range(n_steps):
        actions = np.random.random((n_envs, action_size))

        observations, rewards, terminals, infos = env.step(actions)

        assert observations.shape == (n_envs,) + observation_shape
        assert rewards.shape == (n_envs,)
        assert terminals.shape == (n_envs,)
        assert len(infos) == n_envs


@pytest.mark.parametrize("n_envs", [5])
@pytest.mark.parametrize("n_steps", [1000])
def test_sync_batch_env_discrete(n_envs, n_steps):
    env = SyncBatchEnv([gym.make("CartPole-v0") for _ in range(n_envs)])

    observation_shape = env.observation_space.shape
    action_size = env.action_space.n

    observations = env.reset()
    assert observations.shape == (n_envs,) + observation_shape

    for _ in range(n_steps):
        actions = np.random.randint(action_size, size=n_envs)

        observations, rewards, terminals, infos = env.step(actions)

        assert observations.shape == (n_envs,) + observation_shape
        assert rewards.shape == (n_envs,)
        assert terminals.shape == (n_envs,)
        assert len(infos) == n_envs


@pytest.mark.parametrize("n_envs", [5])
@pytest.mark.parametrize("n_steps", [1000])
def test_sync_batch_env_continuous(n_envs, n_steps):
    env = SyncBatchEnv([gym.make("Pendulum-v0") for _ in range(n_envs)])

    observation_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]

    observations = env.reset()
    assert observations.shape == (n_envs,) + observation_shape

    for _ in range(n_steps):
        actions = np.random.random((n_envs, action_size))

        observations, rewards, terminals, infos = env.step(actions)

        assert observations.shape == (n_envs,) + observation_shape
        assert rewards.shape == (n_envs,)
        assert terminals.shape == (n_envs,)
        assert len(infos) == n_envs
