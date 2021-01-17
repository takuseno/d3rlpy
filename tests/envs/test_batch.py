import pytest
import gym
import numpy as np

from d3rlpy.envs import BatchEnvWrapper


@pytest.mark.parametrize("n_envs", [5])
@pytest.mark.parametrize("n_steps", [1000])
def test_batch_env_wrapper_discrete(n_envs, n_steps):
    env = BatchEnvWrapper([gym.make("CartPole-v0") for _ in range(n_envs)])

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
def test_batch_env_wrapper_continuous(n_envs, n_steps):
    env = BatchEnvWrapper([gym.make("Pendulum-v0") for _ in range(n_envs)])

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
