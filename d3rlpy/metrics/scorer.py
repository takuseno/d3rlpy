import numpy as np

from ..dataset import TransitionMiniBatch


def _make_batches_from_episode(episode, window_size):
    n_batches = len(episode) // window_size
    if len(episode) % window_size != 0:
        n_batches += 1
    for i in range(n_batches):
        head_index = i * window_size
        last_index = min(head_index + window_size, len(episode))
        transitions = episode.transitions[head_index:last_index]
        batch = TransitionMiniBatch(transitions)
        yield batch


def td_error_scorer(algo, episodes, window_size=1024):
    total_errors = []
    for episode in episodes:
        for batch in _make_batches_from_episode(episode, window_size):
            # estimate values for current observations
            values = algo.predict_value(batch.observations, batch.actions)

            # estimate values for next observations
            next_actions = algo.predict(batch.next_observations)
            next_values = algo.predict_value(batch.next_observations,
                                             next_actions)

            # calculate td errors
            mask = (1.0 - batch.terminals).reshape(-1)
            rewards = batch.next_rewards.reshape(-1)
            y = rewards + algo.gamma * next_values * mask
            total_errors += ((values - y)**2).tolist()

    # smaller is better
    return -np.mean(total_errors)


def discounted_sum_of_advantage_scorer(algo, episodes, window_size=1024):
    total_sums = []
    for episode in episodes:
        for batch in _make_batches_from_episode(episode, window_size):
            # estimate values for dataset actions
            dataset_values = algo.predict_value(batch.observations,
                                                batch.actions)

            # estimate values for the current policy
            actions = algo.predict(batch.observations)
            on_policy_values = algo.predict_value(batch.observations, actions)

            # calculate advantages
            advantages = (dataset_values - on_policy_values).tolist()

            # calculate discounted sum of advantages
            A = advantages[-1]
            sum_advantages = [A]
            for advantage in reversed(advantages[:-1]):
                A = advantage + algo.gamma * A
                sum_advantages.append(A)

            total_sums += sum_advantages

    # smaller is better
    return -np.mean(total_sums)


def average_value_estimation_scorer(algo, episodes, window_size=1024):
    total_values = []
    for episode in episodes:
        for batch in _make_batches_from_episode(episode, window_size):
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
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
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
