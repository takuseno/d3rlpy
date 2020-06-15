import copy

from .base import AlgoBase
from .torch.dqn_impl import DQNImpl, DoubleDQNImpl


class DQN(AlgoBase):
    def __init__(self,
                 learning_rate=2.5e-4,
                 batch_size=32,
                 gamma=0.99,
                 alpha=0.95,
                 eps=1e-2,
                 grad_clip=10.0,
                 target_update_interval=100,
                 use_batch_norm=True,
                 n_epochs=1000,
                 use_gpu=False,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.grad_clip = grad_clip
        self.target_update_interval = target_update_interval
        self.use_batch_norm = use_batch_norm
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = DQNImpl(observation_shape=observation_shape,
                            action_size=action_size,
                            learning_rate=self.learning_rate,
                            gamma=self.gamma,
                            alpha=self.alpha,
                            eps=self.eps,
                            grad_clip=self.grad_clip,
                            use_batch_norm=self.use_batch_norm,
                            use_gpu=self.use_gpu)

    def update(self, epoch, itr, batch):
        loss = self.impl.update(batch.observations, batch.actions,
                                batch.next_rewards, batch.next_observations,
                                batch.terminals)
        if (itr + 1) * (epoch + 1) % self.target_update_interval == 0:
            self.impl.update_target()
        return loss


class DoubleDQN(DQN):
    def create_impl(self, observation_shape, action_size):
        self.impl = DoubleDQNImpl(observation_shape=observation_shape,
                                  action_size=action_size,
                                  learning_rate=self.learning_rate,
                                  gamma=self.gamma,
                                  alpha=self.alpha,
                                  eps=self.eps,
                                  grad_clip=self.grad_clip,
                                  use_batch_norm=self.use_batch_norm,
                                  use_gpu=self.use_gpu)
