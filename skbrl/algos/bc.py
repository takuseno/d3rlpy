from abc import ABCMeta, abstractmethod
from .base import AlgoBase


class IBCImpl(metaclass=ABCMeta):
    @abstractmethod
    def update_imitator(self, obs_t, act_t):
        pass


class BC(AlgoBase):
    def __init__(self,
                 learning_rate=1e-3,
                 batch_size=100,
                 eps=1e-8,
                 use_batch_norm=False,
                 n_epochs=1000,
                 use_gpu=False,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size)
        self.learning_rate = learning_rate
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        from .torch.bc_impl import BCImpl
        self.impl = BCImpl(observation_shape=observation_shape,
                           action_size=action_size,
                           learning_rate=self.learning_rate,
                           eps=self.eps,
                           use_batch_norm=self.use_batch_norm,
                           use_gpu=self.use_gpu)

    def update(self, epoch, itr, batch):
        loss = self.impl.update_imitator(batch.observations, batch.actions)
        return (loss, )

    def _get_loss_labels(self):
        return ['loss']


class DiscreteBC(BC):
    def __init__(self,
                 learning_rate=1e-3,
                 batch_size=100,
                 eps=1e-8,
                 beta=0.5,
                 use_batch_norm=True,
                 n_epochs=1000,
                 use_gpu=False,
                 impl=None,
                 **kwargs):
        super().__init__(learning_rate, batch_size, eps, use_batch_norm,
                         n_epochs, use_gpu, impl, **kwargs)
        self.beta = beta

    def create_impl(self, observation_shape, action_size):
        from .torch.bc_impl import DiscreteBCImpl
        self.impl = DiscreteBCImpl(observation_shape=observation_shape,
                                   action_size=action_size,
                                   learning_rate=self.learning_rate,
                                   eps=self.eps,
                                   beta=self.beta,
                                   use_batch_norm=self.use_batch_norm,
                                   use_gpu=self.use_gpu)
