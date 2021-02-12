from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from ..base import ImplBase, LearnableBase
from ..algos import AlgoBase, DataGenerator
from ..dataset import Transition, TransitionMiniBatch
from ..argument_utility import ScalerArg, ActionScalerArg
from ..constants import IMPL_NOT_INITIALIZED_ERROR


class DynamicsImplBase(ImplBase):
    @abstractmethod
    def predict(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def generate(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class DynamicsBase(DataGenerator, LearnableBase):

    _n_transitions: int
    _horizon: int
    _impl: Optional[DynamicsImplBase]

    def __init__(
        self,
        batch_size: int,
        n_frames: int,
        n_transitions: int,
        horizon: int,
        scaler: ScalerArg,
        action_scaler: ActionScalerArg,
    ):
        super().__init__(
            batch_size, n_frames, 1, 1.0, False, 1, scaler, action_scaler
        )
        self._n_transitions = n_transitions
        self._horizon = horizon
        self._impl = None

    def predict(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_variance: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Returns predicted observation and reward.

        Args:
            x: observation
            action: action
            with_variance: flag to return prediction variance.

        Returns:
            tuple of predicted observation and reward.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        observations, rewards, variances = self._impl.predict(x, action)
        if with_variance:
            return observations, rewards, variances
        return observations, rewards

    def generate(
        self, algo: AlgoBase, transitions: List[Transition]
    ) -> List[Transition]:
        """Returns new transitions for data augmentation.

        Args:
            algo: algorithm.
            transitions: list of transitions.

        Returns:
            list: list of generated transitions.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        # uniformly sample transitions
        init_transitions: List[Transition] = []
        for i in np.random.randint(len(transitions), size=self._n_transitions):
            init_transitions.append(transitions[i])

        observation_shape = transitions[0].get_observation_shape()
        action_size = transitions[0].get_action_size()

        rets: List[Transition] = []

        # rollout
        batch = TransitionMiniBatch(init_transitions)
        observations = batch.observations
        actions = algo.sample_action(observations)
        rewards = batch.rewards
        for _ in range(self._horizon):
            # predict next state
            next_observations, next_rewards = self._impl.generate(
                observations, actions
            )

            # sample policy action
            next_actions = algo.sample_action(next_observations)

            # append new transitions
            new_transitions = []
            for i in range(self._n_transitions):
                transition = Transition(
                    observation_shape=observation_shape,
                    action_size=action_size,
                    observation=observations[i],
                    action=actions[i],
                    reward=float(rewards[i][0]),
                    next_observation=next_observations[i],
                    next_action=next_actions[i],
                    next_reward=float(next_rewards[i][0]),
                    terminal=0.0,
                )
                new_transitions.append(transition)

            rets += new_transitions
            observations = next_observations.copy()
            actions = next_actions.copy()
            rewards = next_rewards.copy()

        return rets

    @property
    def n_transitions(self) -> int:
        return self._n_transitions

    @property
    def horizon(self) -> int:
        return self._horizon
