import pytest
import gym

from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.dataset import TransitionMiniBatch


@pytest.mark.parametrize('n_episodes', [10])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('maxlen', [50])
@pytest.mark.parametrize('gamma', [0.99])
def test_replay_buffer(n_episodes, batch_size, maxlen, gamma):
    env = gym.make('CartPole-v0')

    buffer = ReplayBuffer(maxlen, env, gamma)

    total_step = 0
    for episode in range(n_episodes):
        observation, reward, terminal = env.reset(), 0.0, False
        while not terminal:
            action = env.action_space.sample()
            buffer.append(observation, action, reward, terminal)
            observation, reward, terminal, _ = env.step(action)
            total_step += 1
        buffer.append(observation, action, reward, terminal)
        total_step += 1

    assert len(buffer) == maxlen

    observation_shape = env.observation_space.shape
    batch = buffer.sample(batch_size)
    assert len(batch) == batch_size
    assert batch.observations.shape == (batch_size, ) + observation_shape
    assert batch.actions.shape == (batch_size, 1)
    assert batch.rewards.shape == (batch_size, 1)
    assert batch.next_observations.shape == (batch_size, ) + observation_shape
    assert batch.next_actions.shape == (batch_size, 1)
    assert batch.next_rewards.shape == (batch_size, 1)
    assert batch.terminals.shape == (batch_size, 1)
    assert len(batch.returns) == batch_size
    assert len(batch.consequent_observations) == batch_size
