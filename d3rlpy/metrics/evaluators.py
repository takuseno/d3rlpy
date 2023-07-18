from typing import Any, Iterator, Optional, Sequence, cast

import gym
import numpy as np
from typing_extensions import Protocol

from ..dataset import (
    EpisodeBase,
    ReplayBuffer,
    TransitionMiniBatch,
    TransitionPickerProtocol,
)
from ..interface import QLearningAlgoProtocol
from .utility import evaluate_qlearning_with_environment

__all__ = [
    "EvaluatorProtocol",
    "make_batches",
    "TDErrorEvaluator",
    "DiscountedSumOfAdvantageEvaluator",
    "AverageValueEstimationEvaluator",
    "InitialStateValueEstimationEvaluator",
    "SoftOPCEvaluator",
    "ContinuousActionDiffEvaluator",
    "DiscreteActionMatchEvaluator",
    "CompareContinuousActionDiffEvaluator",
    "CompareDiscreteActionMatchEvaluator",
    "EnvironmentEvaluator",
]


WINDOW_SIZE = 1024


class EvaluatorProtocol(Protocol):
    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        """Computes metrics.

        Args:
            algo: Q-learning algorithm.
            dataset: ReplayBuffer.

        Returns:
            Computed metrics.
        """
        raise NotImplementedError


def make_batches(
    episode: EpisodeBase,
    window_size: int,
    transition_picker: TransitionPickerProtocol,
) -> Iterator[TransitionMiniBatch]:
    n_batches = len(episode) // window_size
    if len(episode) % window_size != 0:
        n_batches += 1
    for i in range(n_batches):
        head_index = i * window_size
        last_index = min(head_index + window_size, episode.transition_count)
        transitions = [
            transition_picker(episode, index)
            for index in range(head_index, last_index)
        ]
        batch = TransitionMiniBatch.from_transitions(transitions)
        yield batch


class TDErrorEvaluator(EvaluatorProtocol):
    r"""Returns average TD error.

    This metics suggests how Q functions overfit to training sets.
    If the TD error is large, the Q functions are overfitting.

    .. math::

        \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D}
            [(Q_\theta (s_t, a_t)
             - r_{t+1} - \gamma \max_a Q_\theta (s_{t+1}, a))^2]

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_errors = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                # estimate values for current observations
                values = algo.predict_value(batch.observations, batch.actions)

                # estimate values for next observations
                next_actions = algo.predict(batch.next_observations)
                next_values = algo.predict_value(
                    batch.next_observations, next_actions
                )

                # calculate td errors
                mask = (1.0 - np.asarray(batch.terminals)).reshape(-1)
                rewards = np.asarray(batch.rewards).reshape(-1)
                if algo.reward_scaler:
                    rewards = algo.reward_scaler.transform_numpy(rewards)
                y = rewards + algo.gamma * cast(np.ndarray, next_values) * mask
                total_errors += ((values - y) ** 2).tolist()

        return float(np.mean(total_errors))


class DiscountedSumOfAdvantageEvaluator(EvaluatorProtocol):
    r"""Returns average of discounted sum of advantage.

    This metrics suggests how the greedy-policy selects different actions in
    action-value space.
    If the sum of advantage is small, the policy selects actions with larger
    estimated action-values.

    .. math::

        \mathbb{E}_{s_t, a_t \sim D}
            [\sum_{t' = t} \gamma^{t' - t} A(s_{t'}, a_{t'})]

    where :math:`A(s_t, a_t) = Q_\theta (s_t, a_t)
    - \mathbb{E}_{a \sim \pi} [Q_\theta (s_t, a)]`.

    References:
        * `Murphy., A generalization error for Q-Learning.
          <http://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf>`_

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_sums = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                # estimate values for dataset actions
                dataset_values = algo.predict_value(
                    batch.observations, batch.actions
                )
                dataset_values = cast(np.ndarray, dataset_values)

                # estimate values for the current policy
                actions = algo.predict(batch.observations)
                on_policy_values = algo.predict_value(
                    batch.observations, actions
                )

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
        return float(np.mean(total_sums))


class AverageValueEstimationEvaluator(EvaluatorProtocol):
    r"""Returns average value estimation.

    This metrics suggests the scale for estimation of Q functions.
    If average value estimation is too large, the Q functions overestimate
    action-values, which possibly makes training failed.

    .. math::

        \mathbb{E}_{s_t \sim D} [ \max_a Q_\theta (s_t, a)]

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_values = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                actions = algo.predict(batch.observations)
                values = algo.predict_value(batch.observations, actions)
                total_values += cast(np.ndarray, values).tolist()
        return float(np.mean(total_values))


class InitialStateValueEstimationEvaluator(EvaluatorProtocol):
    r"""Returns mean estimated action-values at the initial states.

    This metrics suggests how much return the trained policy would get from
    the initial states by deploying the policy to the states.
    If the estimated value is large, the trained policy is expected to get
    higher returns.

    .. math::

        \mathbb{E}_{s_0 \sim D} [Q(s_0, \pi(s_0))]

    References:
        * `Paine et al., Hyperparameter Selection for Offline Reinforcement
          Learning <https://arxiv.org/abs/2007.09055>`_

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """

    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_values = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                # estimate action-value in initial states
                actions = algo.predict([batch.observations[0]])
                values = algo.predict_value([batch.observations[0]], actions)
                total_values.append(values[0])
        return float(np.mean(total_values))


class SoftOPCEvaluator(EvaluatorProtocol):
    r"""Returns Soft Off-Policy Classification metrics.

    The metrics of the scorer funciton is evaluating gaps of action-value
    estimation between the success episodes and the all episodes.
    If the learned Q-function is optimal, action-values in success episodes
    are expected to be higher than the others.
    The success episode is defined as an episode with a return above the given
    threshold.

    .. math::

        \mathbb{E}_{s, a \sim D_{success}} [Q(s, a)]
            - \mathbb{E}_{s, a \sim D} [Q(s, a)]

    References:
        * `Irpan et al., Off-Policy Evaluation via Off-Policy Classification.
          <https://arxiv.org/abs/1906.01624>`_

    Args:
        return_threshold: Return threshold of success episodes.
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _return_threshold: float
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(
        self,
        return_threshold: float,
        episodes: Optional[Sequence[EpisodeBase]] = None,
    ):
        self._return_threshold = return_threshold
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        success_values = []
        all_values = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            is_success = episode.compute_return() >= self._return_threshold
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                values = algo.predict_value(batch.observations, batch.actions)
                values = cast(np.ndarray, values)
                all_values += values.reshape(-1).tolist()
                if is_success:
                    success_values += values.reshape(-1).tolist()
        return float(np.mean(success_values) - np.mean(all_values))


class ContinuousActionDiffEvaluator(EvaluatorProtocol):
    r"""Returns squared difference of actions between algorithm and dataset.

    This metrics suggests how different the greedy-policy is from the given
    episodes in continuous action-space.
    If the given episodes are near-optimal, the small action difference would
    be better.

    .. math::

        \mathbb{E}_{s_t, a_t \sim D} [(a_t - \pi_\phi (s_t))^2]

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_diffs = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                actions = algo.predict(batch.observations)
                diff = ((batch.actions - actions) ** 2).sum(axis=1).tolist()
                total_diffs += diff
        return float(np.mean(total_diffs))


class DiscreteActionMatchEvaluator(EvaluatorProtocol):
    r"""Returns percentage of identical actions between algorithm and dataset.

    This metrics suggests how different the greedy-policy is from the given
    episodes in discrete action-space.
    If the given episdoes are near-optimal, the large percentage would be
    better.

    .. math::

        \frac{1}{N} \sum^N \parallel
            \{a_t = \text{argmax}_a Q_\theta (s_t, a)\}

    Args:
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(self, episodes: Optional[Sequence[EpisodeBase]] = None):
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_matches = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                actions = algo.predict(batch.observations)
                match = (batch.actions.reshape(-1) == actions).tolist()
                total_matches += match
        return float(np.mean(total_matches))


class CompareContinuousActionDiffEvaluator(EvaluatorProtocol):
    r"""Action difference between algorithms.

    This metrics suggests how different the two algorithms are in continuous
    action-space.
    If the algorithm to compare with is near-optimal, the small action
    difference would be better.

    .. math::

        \mathbb{E}_{s_t \sim D}
            [(\pi_{\phi_1}(s_t) - \pi_{\phi_2}(s_t))^2]

    Args:
        base_algo: Target algorithm to comapre with.
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _base_algo: QLearningAlgoProtocol
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(
        self,
        base_algo: QLearningAlgoProtocol,
        episodes: Optional[Sequence[EpisodeBase]] = None,
    ):
        self._base_algo = base_algo
        self._episodes = episodes

    def __call__(
        self,
        algo: QLearningAlgoProtocol,
        dataset: ReplayBuffer,
    ) -> float:
        total_diffs = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            # TODO: handle different n_frames
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                base_actions = self._base_algo.predict(batch.observations)
                actions = algo.predict(batch.observations)
                diff = ((actions - base_actions) ** 2).sum(axis=1).tolist()
                total_diffs += diff
        return float(np.mean(total_diffs))


class CompareDiscreteActionMatchEvaluator(EvaluatorProtocol):
    r"""Action matches between algorithms.

    This metrics suggests how different the two algorithms are in discrete
    action-space.
    If the algorithm to compare with is near-optimal, the small action
    difference would be better.

    .. math::

        \mathbb{E}_{s_t \sim D} [\parallel
            \{\text{argmax}_a Q_{\theta_1}(s_t, a)
            = \text{argmax}_a Q_{\theta_2}(s_t, a)\}]

    Args:
        base_algo: Target algorithm to comapre with.
        episodes: Optional evaluation episodes. If it's not given, dataset
            used in training will be used.
    """
    _base_algo: QLearningAlgoProtocol
    _episodes: Optional[Sequence[EpisodeBase]]

    def __init__(
        self,
        base_algo: QLearningAlgoProtocol,
        episodes: Optional[Sequence[EpisodeBase]] = None,
    ):
        self._base_algo = base_algo
        self._episodes = episodes

    def __call__(
        self, algo: QLearningAlgoProtocol, dataset: ReplayBuffer
    ) -> float:
        total_matches = []
        episodes = self._episodes if self._episodes else dataset.episodes
        for episode in episodes:
            # TODO: handle different n_frames
            for batch in make_batches(
                episode, WINDOW_SIZE, dataset.transition_picker
            ):
                base_actions = self._base_algo.predict(batch.observations)
                actions = algo.predict(batch.observations)
                match = (base_actions == actions).tolist()
                total_matches += match
        return float(np.mean(total_matches))


class EnvironmentEvaluator(EvaluatorProtocol):
    r"""Action matches between algorithms.

    This metrics suggests how different the two algorithms are in discrete
    action-space.
    If the algorithm to compare with is near-optimal, the small action
    difference would be better.

    .. math::

        \mathbb{E}_{s_t \sim D} [\parallel
            \{\text{argmax}_a Q_{\theta_1}(s_t, a)
            = \text{argmax}_a Q_{\theta_2}(s_t, a)\}]

    Args:
        env: Gym environment.
        n_trials: Number of episodes to evaluate.
        epsilon: Probability of random action.
        render: Flag to turn on rendering.
    """
    _env: gym.Env[Any, Any]
    _n_trials: int
    _epsilon: float
    _render: bool

    def __init__(
        self,
        env: gym.Env[Any, Any],
        n_trials: int = 10,
        epsilon: float = 0.0,
        render: bool = False,
    ):
        self._env = env
        self._n_trials = n_trials
        self._epsilon = epsilon
        self._render = render

    def __call__(
        self, algo: QLearningAlgoProtocol, dataset: ReplayBuffer
    ) -> float:
        return evaluate_qlearning_with_environment(
            algo=algo,
            env=self._env,
            n_trials=self._n_trials,
            epsilon=self._epsilon,
            render=self._render,
        )
