import gym
import numpy as np

from ..algos.qlearning.interface import QLearningAlgoProtocol
from ..algos.transformer.interface import StatefulTransformerAlgoProtocol

__all__ = ["evaluate_qlearning_with_environment", "evaluate_transformer_with_environment"]


def evaluate_qlearning_with_environment(
    algo: QLearningAlgoProtocol,
    env: gym.Env,
    n_trials: int = 10,
    epsilon: float = 0.0,
    render: bool = False,
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.
        render: flag to render environment.

    Returns:
        average score.

    """
    episode_rewards = []
    for _ in range(n_trials):
        observation = env.reset()
        episode_reward = 0.0

        while True:
            # take action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = algo.predict(np.expand_dims(observation, axis=0))[0]

            observation, reward, done, _ = env.step(action)
            episode_reward += reward

            if render:
                env.render()

            if done:
                break
        episode_rewards.append(episode_reward)
    return float(np.mean(episode_rewards))


def evaluate_transformer_with_environment(
    algo: StatefulTransformerAlgoProtocol,
    env: gym.Env,
    n_trials: int = 10,
    render: bool = False,
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        render: flag to render environment.

    Returns:
        average score.

    """
    episode_rewards = []
    for _ in range(n_trials):
        algo.reset()
        observation, reward = env.reset(), 0.0
        episode_reward = 0.0

        while True:
            # take action
            action = algo.predict(observation, reward)

            observation, reward, done, _ = env.step(action)
            episode_reward += reward

            if render:
                env.render()

            if done:
                break
        episode_rewards.append(episode_reward)
    return float(np.mean(episode_rewards))
