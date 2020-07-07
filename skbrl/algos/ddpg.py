from abc import ABCMeta, abstractmethod
from .base import AlgoBase


class IDDPGImpl(metaclass=ABCMeta):
    @abstractmethod
    def update_critic(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        pass

    @abstractmethod
    def update_actor(self, obs_t):
        pass

    @abstractmethod
    def update_actor_target(self):
        pass

    @abstractmethod
    def update_critic_target(self):
        pass


class DDPG(AlgoBase):
    def __init__(self,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-3,
                 batch_size=100,
                 gamma=0.99,
                 tau=0.005,
                 reguralizing_rate=1e-10,
                 eps=1e-8,
                 use_batch_norm=False,
                 use_quantile_regression=None,
                 n_epochs=1000,
                 use_gpu=False,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.reguralizing_rate = reguralizing_rate
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.use_quantile_regression = use_quantile_regression
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        from .torch.ddpg_impl import DDPGImpl
        self.impl = DDPGImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            gamma=self.gamma,
            tau=self.tau,
            reguralizing_rate=self.reguralizing_rate,
            eps=self.eps,
            use_batch_norm=self.use_batch_norm,
            use_quantile_regression=self.use_quantile_regression,
            use_gpu=self.use_gpu)

    def update(self, epoch, itr, batch):
        critic_loss = self.impl.update_critic(batch.observations,
                                              batch.actions,
                                              batch.next_rewards,
                                              batch.next_observations,
                                              batch.terminals)
        actor_loss = self.impl.update_actor(batch.observations)
        self.impl.update_critic_target()
        self.impl.update_actor_target()
        return critic_loss, actor_loss

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss']
