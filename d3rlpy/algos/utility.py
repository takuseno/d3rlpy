from typing import List, Optional, Tuple, cast

import numpy as np

from ..constants import DYNAMICS_NOT_GIVEN_ERROR, IMPL_NOT_INITIALIZED_ERROR
from ..dataset import Transition, TransitionMiniBatch
from ..dynamics import DynamicsBase
from .base import AlgoImplBase


class ModelBaseMixin:
    _impl: Optional[AlgoImplBase]
    _dynamics: Optional[DynamicsBase]

    def generate_new_data(
        self, epoch: int, total_step: int, transitions: List[Transition]
    ) -> Optional[List[Transition]]:
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert self._dynamics, DYNAMICS_NOT_GIVEN_ERROR

        if not self._is_generating_new_data(epoch, total_step):
            return None

        init_transitions = self._sample_initial_transitions(transitions)

        rets: List[Transition] = []

        # rollout
        batch = TransitionMiniBatch(init_transitions)
        observations = batch.observations
        actions = self._rollout_policy(observations)
        rewards = batch.rewards
        prev_transitions: List[Transition] = []
        for _ in range(self._rollout_length()):
            # predict next state
            pred = self._dynamics.predict(observations, actions, True)
            pred = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], pred)
            next_observations, next_rewards, variances = pred

            # regularize by uncertainty
            next_observations, next_rewards = self._mutate_transition(
                next_observations, next_rewards, variances
            )

            # sample policy action
            next_actions = self._rollout_policy(next_observations)

            # append new transitions
            new_transitions = []
            for i in range(len(init_transitions)):
                transition = Transition(
                    observation_shape=self._impl.observation_shape,
                    action_size=self._impl.action_size,
                    observation=observations[i],
                    action=actions[i],
                    reward=float(rewards[i][0]),
                    next_observation=next_observations[i],
                    next_action=next_actions[i],
                    next_reward=float(next_rewards[i][0]),
                    terminal=0.0,
                )

                if prev_transitions:
                    prev_transitions[i].next_transition = transition
                    transition.prev_transition = prev_transitions[i]

                new_transitions.append(transition)

            prev_transitions = new_transitions
            rets += new_transitions
            observations = next_observations.copy()
            actions = next_actions.copy()
            rewards = next_rewards.copy()

        return rets

    def _is_generating_new_data(self, epoch: int, total_step: int) -> bool:
        raise NotImplementedError

    def _sample_initial_transitions(
        self, transitions: List[Transition]
    ) -> List[Transition]:
        raise NotImplementedError

    def _rollout_policy(self, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _rollout_length(self) -> int:
        raise NotImplementedError

    def _mutate_transition(
        self,
        observations: np.ndarray,
        rewards: np.ndarray,
        variances: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return observations, rewards
