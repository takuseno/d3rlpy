import numpy as np

from abc import abstractmethod
from ..base import ImplBase, LearnableBase
from ..dataset import Transition, TransitionMiniBatch


class DynamicsImplBase(ImplBase):
    @abstractmethod
    def predict(self, x, action, with_variance):
        pass

    @abstractmethod
    def generate(self, x, action):
        pass


class DynamicsBase(LearnableBase):
    def __init__(self, batch_size, n_frames, n_transitions, horizon, scaler):
        super().__init__(batch_size, n_frames, scaler)
        self.n_transitions = n_transitions
        self.horizon = horizon

    def predict(self, x, action, with_variance=False):
        """ Returns predicted observation and reward.

        Args:
            x (numpy.ndarray): observation
            action (numpy.ndarray): action
            with_variance (bool): flag to return prediction variance.

        Returns:
            tuple: tuple of predicted observation and reward.

        """
        observations, rewards, variances = self.impl.predict(x, action)
        if with_variance:
            return observations, rewards, variances
        return observations, rewards

    def generate(self, algo, transitions):
        """ Returns new transitions for data augmentation.

        Args:
            algo (d3rlpy.algos.base.AlgoBase): algorithm.
            transitions (list(d3rlpy.dataset.Transition)): list of transitions.

        Returns:
            list(d3rlpy.dataset.Transition): list of generated transitions.

        """
        # uniformly sample transitions
        init_transitions = []
        for i in np.random.randint(len(transitions), size=self.n_transitions):
            init_transitions.append(transitions[i])

        observation_shape = transitions[0].get_observation_shape()
        action_size = transitions[0].get_action_size()

        rets = []

        # rollout
        batch = TransitionMiniBatch(init_transitions)
        observations = batch.observations
        actions = algo.sample_action(observations)
        rewards = batch.rewards
        for _ in range(self.horizon):
            # predict next state
            next_observations, next_rewards = self.impl.generate(
                observations, actions)

            # sample policy action
            next_actions = algo.sample_action(next_observations)

            # append new transitions
            new_transitions = []
            for i in range(self.n_transitions):
                transition = Transition(observation_shape=observation_shape,
                                        action_size=action_size,
                                        observation=observations[i],
                                        action=actions[i],
                                        reward=float(rewards[i][0]),
                                        next_observation=next_observations[i],
                                        next_action=next_actions[i],
                                        next_reward=float(next_rewards[i][0]),
                                        terminal=0.0)
                new_transitions.append(transition)

            rets += new_transitions
            observations = next_observations.copy()
            actions = next_actions.copy()
            rewards = next_rewards.copy()

        return rets
