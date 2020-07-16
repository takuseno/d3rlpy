import numpy as np

from .scorer import _make_batches_from_episode


def compare_continuous_action_diff(base_algo, window_size=1024):
    def scorer(algo, episodes):
        total_diffs = []
        for episode in episodes:
            for batch in _make_batches_from_episode(episode, window_size):
                base_actions = base_algo.predict(batch.observations)
                actions = algo.predict(batch.observations)
                diff = ((actions - base_actions)**2).sum(axis=1).tolist()
                total_diffs += diff
        # smaller is better, sometimes?
        return -np.mean(total_diffs)

    return scorer


def compare_discrete_action_match(base_algo, window_size=1024):
    def scorer(algo, episodes):
        total_matches = []
        for episode in episodes:
            for batch in _make_batches_from_episode(episode, window_size):
                base_actions = base_algo.predict(batch.observations)
                actions = algo.predict(batch.observations)
                match = (base_actions == actions).tolist()
                total_matches += match
        return np.mean(total_matches)

    return scorer
