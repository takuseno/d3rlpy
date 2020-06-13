import numpy as np
import random
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
        self.impl.update_target()
        return loss

    def get_params(self, deep=True):
        impl = self.impl
        if deep:
            impl = copy.deepcopy(impl)

        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'eps': self.eps,
            'use_batch_norm': self.use_batch_norm,
            'n_epochs': self.n_epochs,
            'use_gpu': self.use_gpu,
            'impl': impl
        }

    def set_params(self, **params):
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.eps = params['eps']
        self.use_batch_norm = params['use_batch_norm']
        self.n_epochs = params['n_epochs']
        self.use_gpu = params['use_gpu']
        self.impl = params['impl']


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
