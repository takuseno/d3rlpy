from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from typing_extensions import Protocol

from ..dataset import Episode, TransitionMiniBatch, TransitionPickerProtocol
from ..preprocessing.reward_scalers import RewardScaler

__all__ = [
    "EvaluatorProtocol",
    "make_batches",
    "TDErrorEvaluator",
    "DiscountedSumOfAdvantageEvaluator",
    "AverageValueEstimationEvaluator",
    "ValueEstimationStdEvaluator",
    "InitialStateValueEstimationEvaluator",
    "SoftOPCEvaluator",
    "ContinuousActionDiffEvaluator",
    "DiscreteActionMatchEvaluator",
    "CompareContinuousActionDiffEvaluator",
    "CompareDiscreteActionMatchEvaluator",
]


WINDOW_SIZE = 1024


class AlgoProtocol(Protocol):
    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        ...

    @property
    def gamma(self) -> float:
        ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        ...


class EvaluatorProtocol(Protocol):
    def __call__(
        self,
        algo: AlgoProtocol,
        episodes: Sequence[Episode],
        transition_picker: TransitionPickerProtocol,
    ) -> float:
        ...


def make_batches(
    episode: Episode,
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

    """

    def __call__(
        self,
        algo: AlgoProtocol,
        episodes: Sequence[Episode],
        transition_picker: TransitionPickerProtocol,
    ) -> float:
        total_errors = []
        for episode in episodes:
            for batch in make_batches(episode, WINDOW_SIZE, transition_picker):
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

    """

    def __call__(
        self,
        algo: AlgoProtocol,
        episodes: Sequence[Episode],
        transition_picker: TransitionPickerProtocol,
    ) -> float:
        total_sums = []
        for episode in episodes:
            for batch in make_batches(episode, WINDOW_SIZE, transition_picker):
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

    """

    def __call__(
        self,
        algo: AlgoProtocol,
        episodes: Sequence[Episode],
        transition_picker: TransitionPickerProtocol,
    ) -> float:
        total_values = []
        for episode in episodes:
            for batch in make_batches(episode, WINDOW_SIZE, transition_picker):
                actions = algo.predict(batch.observations)
                values = algo.predict_value(batch.observations, actions)
                total_values += cast(np.ndarray, values).tolist()
        return float(np.mean(total_values))


class ValueEstimationStdEvaluator(EvaluatorProtocol):
    r"""Returns standard deviation of value estimation.

    This metrics suggests how confident Q functions are for the given
    episodes.
    This metrics will be more accurate with `boostrap` enabled and the larger
    `n_critics` at algorithm.
    If standard deviation of value estimation is large, the Q functions are
    overfitting to the training set.

    .. math::

        \mathbb{E}_{s_t \sim D, a \sim \text{argmax}_a Q_\theta(s_t, a)}
            [Q_{\text{std}}(s_t, a)]

    where :math:`Q_{\text{std}}(s, a)` is a standard deviation of action-value
    estimation over ensemble functions.

    """

    def __call__(
        self,
        algo: AlgoProtocol,
        episodes: Sequence[Episode],
        transition_picker: TransitionPickerProtocol,
    ) -> float:
        total_stds = []
        for episode in episodes:
            for batch in make_batches(episode, WINDOW_SIZE, transition_picker):
                actions = algo.predict(batch.observations)
                _, stds = algo.predict_value(batch.observations, actions, True)
                total_stds += stds.tolist()
        return float(np.mean(total_stds))


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

    """

    def __call__(
        self,
        algo: AlgoProtocol,
        episodes: Sequence[Episode],
        transition_picker: TransitionPickerProtocol,
    ) -> float:
        total_values = []
        for episode in episodes:
            for batch in make_batches(episode, WINDOW_SIZE, transition_picker):
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
        return_threshold: threshold of success episodes.

    """
    _return_threshold: float

    def __init__(self, return_threshold: float):
        self._return_threshold = return_threshold

    def __call__(
        self,
        algo: AlgoProtocol,
        episodes: Sequence[Episode],
        transition_picker: TransitionPickerProtocol,
    ) -> float:
        success_values = []
        all_values = []
        for episode in episodes:
            is_success = episode.compute_return() >= self._return_threshold
            for batch in make_batches(episode, WINDOW_SIZE, transition_picker):
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

    """

    def __call__(
        self,
        algo: AlgoProtocol,
        episodes: Sequence[Episode],
        transition_picker: TransitionPickerProtocol,
    ) -> float:
        total_diffs = []
        for episode in episodes:
            for batch in make_batches(episode, WINDOW_SIZE, transition_picker):
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

    """

    def __call__(
        self,
        algo: AlgoProtocol,
        episodes: Sequence[Episode],
        transition_picker: TransitionPickerProtocol,
    ) -> float:
        total_matches = []
        for episode in episodes:
            for batch in make_batches(episode, WINDOW_SIZE, transition_picker):
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
        base_algo: algorithm to comapre with.

    """
    _base_algo: AlgoProtocol

    def __init__(self, base_algo: AlgoProtocol):
        self._base_algo = base_algo

    def __call__(
        self,
        algo: AlgoProtocol,
        episodes: Sequence[Episode],
        transition_picker: TransitionPickerProtocol,
    ) -> float:
        total_diffs = []
        for episode in episodes:
            # TODO: handle different n_frames
            for batch in make_batches(episode, WINDOW_SIZE, transition_picker):
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
        base_algo: algorithm to comapre with.

    """
    _base_algo: AlgoProtocol

    def __init__(self, base_algo: AlgoProtocol):
        self._base_algo = base_algo

    def __call__(
        self,
        algo: AlgoProtocol,
        episodes: Sequence[Episode],
        transition_picker: TransitionPickerProtocol,
    ) -> float:
        total_matches = []
        for episode in episodes:
            # TODO: handle different n_frames
            for batch in make_batches(episode, WINDOW_SIZE, transition_picker):
                base_actions = self._base_algo.predict(batch.observations)
                actions = algo.predict(batch.observations)
                match = (base_actions == actions).tolist()
                total_matches += match
        return float(np.mean(total_matches))
