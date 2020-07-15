import numpy as np

from ..dataset import TransitionMiniBatch


def td_error_scorer(algo, episodes):
    total_errors = []
    for episode in episodes:
        batch = TransitionMiniBatch(episode.transitions)

        # estimate values for current observations
        values = algo.predict_value(batch.observations, batch.actions)

        # estimate values for next observations
        next_actions = algo.predict(batch.next_observations)
        next_values = algo.predict_value(batch.next_observations, next_actions)

        # calculate td errors
        mask = (1.0 - batch.terminals)
        y = batch.next_rewards + algo.gamma * next_values * mask
        total_errors += ((values - y)**2).reshape(-1).tolist()

    # smaller is better
    return -np.mean(total_errors)


def discounted_sum_of_advantage_scorer(algo, episodes):
    total_sums = []
    for episode in episodes:
        batch = TransitionMiniBatch(episode.transitions)

        # estimate values for dataset actions
        dataset_values = algo.predict_value(batch.observations, batch.actions)

        # estimate values for the current policy
        actions = algo.predict(batch.observations)
        on_policy_values = algo.predict_value(batch.observations, actions)

        # calculate advantages
        advantages = (dataset_values - on_policy_values).reshape(-1).tolist()

        # calculate discounted sum of advantages
        A = advantages[-1]
        sum_advantages = [A]
        for advantage in reversed(advantages[:-1]):
            A = advantage + algo.gamma * A
            sum_advantages.append(A)

        total_sums += sum_advantages

    # smaller is better
    return -np.mean(total_sums)


def average_value_estimation_scorer(algo, episodes):
    total_values = []
    for episode in episodes:
        batch = TransitionMiniBatch(episode.transitions)
        actions = algo.predict(batch.observations)
        values = algo.predict_value(batch.observations, actions)
        total_values += values.tolist()
    return np.mean(total_values)


def evaluate_on_environment(env, n_trials=10, epsilon=0.0, render=False):
    def scorer(algo, *args):
        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0
            while True:
                action = algo.predict([observation])[0]
                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if render:
                    env.render()

                if done:
                    break
            episode_rewards.append(episode_reward)
        return np.mean(episode_rewards)

    return scorer


NEGATED_SCORER = [td_error_scorer, discounted_sum_of_advantage_scorer]
