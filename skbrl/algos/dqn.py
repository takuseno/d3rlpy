import copy

from abc import ABCMeta, abstractmethod
from .base import AlgoBase


class IDQNImpl(metaclass=ABCMeta):
    @abstractmethod
    def update(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        pass

    @abstractmethod
    def update_target(self):
        pass


class DQN(AlgoBase):
    """ Deep Q-Network algorithm.

    .. math::

        L(\\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma \max_a Q_{\\theta'}(s_{t+1}, a) - Q_\\theta(s_t, a_t))^2]

    where :math:`\\theta'` is the target network parameter. The target network
    parameter is synchronized every `target_update_interval` iterations.

    References:
        * `Mnih et al., Human-level control through deep reinforcement
          learning. <https://www.nature.com/articles/nature14236>`_

    Args:
        learning_rate (float): learning rate.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        eps (float): :math:`\epsilon` for Adam optimizer.
        target_update_interval (int): interval to update the target network.
        use_batch_norm (bool): flag to insert batch normalization layers
        distribution_type (str): type of distributional Q function.
            If None, the normal Q function will be used. Available options are
            `['qr', 'iqn']`.
        n_epochs (int): the number of epochs to train.
        use_gpu (bool): flag to use GPU.
        impl (skbrl.algos.dqn.IDQNImpl): algorithm implementation.

    Attributes:
        learning_rate (float): learning rate.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        eps (float): :math:`\epsilon` for Adam optimizer.
        target_update_interval (int): interval to update the target network.
        use_batch_norm (bool): flag to insert batch normalization layers
        distribution_type (str): type of distributional Q function.
        n_epochs (int): the number of epochs to train.
        use_gpu (bool): flag to use GPU.
        impl (skbrl.algos.dqn.IDQNImpl): algorithm implementation.

    """
    def __init__(self,
                 learning_rate=6.25e-5,
                 batch_size=32,
                 gamma=0.99,
                 eps=1.5e-4,
                 target_update_interval=8e3,
                 use_batch_norm=True,
                 distribution_type=None,
                 n_epochs=1000,
                 use_gpu=False,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = eps
        self.target_update_interval = target_update_interval
        self.use_batch_norm = use_batch_norm
        self.distribution_type = distribution_type
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        from .torch.dqn_impl import DQNImpl
        self.impl = DQNImpl(observation_shape=observation_shape,
                            action_size=action_size,
                            learning_rate=self.learning_rate,
                            gamma=self.gamma,
                            eps=self.eps,
                            use_batch_norm=self.use_batch_norm,
                            distribution_type=self.distribution_type,
                            use_gpu=self.use_gpu)

    def update(self, epoch, total_step, batch):
        loss = self.impl.update(batch.observations, batch.actions,
                                batch.next_rewards, batch.next_observations,
                                batch.terminals)
        if total_step % self.target_update_interval == 0:
            self.impl.update_target()
        return (loss, )

    def _get_loss_labels(self):
        return ['value_loss']


class DoubleDQN(DQN):
    """ Double Deep Q-Network algorithm.

    The difference from DQN is that the action is taken from the current Q
    function instead of the target Q function.
    This modification significantly decreases overestimation bias of TD
    learning.

    .. math::

        L(\\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma Q_{\\theta'}(s_{t+1}, \\text{argmax}_a
            Q_\\theta(s_{t+1}, a)) - Q_\\theta(s_t, a_t))^2]

    where :math:`\\theta'` is the target network parameter. The target network
    parameter is synchronized every `target_update_interval` iterations.

    References:
        * `Hasselt et al., Deep reinforcement learning with double Q-learning.
          <https://arxiv.org/abs/1509.06461>`_

    Args:
        learning_rate (float): learning rate.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        eps (float): :math:`\epsilon` for Adam optimizer.
        target_update_interval (int): interval to synchronize the target
            network.
        use_batch_norm (bool): flag to insert batch normalization layers
        distribution_type (str): type of distributional Q function.
            If None, the normal Q function will be used. Available options are
            `['qr', 'iqn']`.
        n_epochs (int): the number of epochs to train.
        use_gpu (bool): flag to use GPU.
        impl (skbrl.algos.dqn.IDQNImpl): algorithm implementation.

    Attributes:
        learning_rate (float): learning rate.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        eps (float): :math:`\epsilon` for Adam optimizer.
        target_update_interval (int): interval to synchronize the target
            network.
        use_batch_norm (bool): flag to insert batch normalization layers
        distribution_type (str): type of distributional Q function.
        n_epochs (int): the number of epochs to train.
        use_gpu (bool): flag to use GPU.
        impl (skbrl.algos.dqn.IDQNImpl): algorithm implementation.

    """
    def create_impl(self, observation_shape, action_size):
        from .torch.dqn_impl import DoubleDQNImpl
        self.impl = DoubleDQNImpl(observation_shape=observation_shape,
                                  action_size=action_size,
                                  learning_rate=self.learning_rate,
                                  gamma=self.gamma,
                                  eps=self.eps,
                                  use_batch_norm=self.use_batch_norm,
                                  distribution_type=self.distribution_type,
                                  use_gpu=self.use_gpu)
