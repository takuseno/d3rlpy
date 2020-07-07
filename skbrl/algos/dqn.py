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
    def __init__(self,
                 learning_rate=6.25e-5,
                 batch_size=32,
                 gamma=0.99,
                 eps=1.5e-4,
                 target_update_interval=8e3,
                 use_batch_norm=True,
                 use_quantile_regression=None,
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
        self.use_quantile_regression = use_quantile_regression
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        from .torch.dqn_impl import DQNImpl
        self.impl = DQNImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            eps=self.eps,
            use_batch_norm=self.use_batch_norm,
            use_quantile_regression=self.use_quantile_regression,
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
    def create_impl(self, observation_shape, action_size):
        from .torch.dqn_impl import DoubleDQNImpl
        self.impl = DoubleDQNImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            eps=self.eps,
            use_batch_norm=self.use_batch_norm,
            use_quantile_regression=self.use_quantile_regression,
            use_gpu=self.use_gpu)
