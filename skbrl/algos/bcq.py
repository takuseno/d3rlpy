from abc import abstractmethod
from .base import AlgoBase
from .dqn import IDQNImpl
from .ddpg import IDDPGImpl


class IBCQImpl(IDDPGImpl):
    @abstractmethod
    def update_imitator(self, obs_t, act_t):
        pass


class BCQ(AlgoBase):
    def __init__(self,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-3,
                 imitator_learning_rate=1e-3,
                 batch_size=100,
                 gamma=0.99,
                 tau=0.005,
                 n_critics=2,
                 lam=0.75,
                 n_action_samples=100,
                 action_flexibility=0.05,
                 rl_start_epoch=0,
                 latent_size=32,
                 beta=0.5,
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
        self.imitator_learning_rate = imitator_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.n_critics = n_critics
        self.lam = lam
        self.n_action_samples = n_action_samples
        self.action_flexibility = action_flexibility
        self.rl_start_epoch = rl_start_epoch
        self.latent_size = latent_size
        self.beta = beta
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.use_quantile_regression = use_quantile_regression
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        from skbrl.algos.torch.bcq_impl import BCQImpl
        self.impl = BCQImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            imitator_learning_rate=self.imitator_learning_rate,
            gamma=self.gamma,
            tau=self.tau,
            n_critics=self.n_critics,
            lam=self.lam,
            n_action_samples=self.n_action_samples,
            action_flexibility=self.action_flexibility,
            latent_size=self.latent_size,
            beta=self.beta,
            eps=self.eps,
            use_batch_norm=self.use_batch_norm,
            use_quantile_regression=self.use_quantile_regression,
            use_gpu=self.use_gpu)

    def update(self, epoch, total_step, batch):
        imitator_loss = self.impl.update_imitator(batch.observations,
                                                  batch.actions)
        if epoch >= self.rl_start_epoch:
            critic_loss = self.impl.update_critic(batch.observations,
                                                  batch.actions,
                                                  batch.next_rewards,
                                                  batch.next_observations,
                                                  batch.terminals)
            actor_loss = self.impl.update_actor(batch.observations)
            self.impl.update_actor_target()
            self.impl.update_critic_target()
        else:
            critic_loss = None
            actor_loss = None
        return critic_loss, actor_loss, imitator_loss

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss', 'imitator_loss']


class DiscreteBCQ(AlgoBase):
    def __init__(self,
                 learning_rate=6.25e-5,
                 batch_size=32,
                 gamma=0.99,
                 action_flexibility=0.3,
                 beta=0.5,
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
        self.action_flexibility = action_flexibility
        self.beta = beta
        self.eps = eps
        self.target_update_interval = target_update_interval
        self.use_batch_norm = use_batch_norm
        self.use_quantile_regression = use_quantile_regression
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        from skbrl.algos.torch.bcq_impl import DiscreteBCQImpl
        self.impl = DiscreteBCQImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            action_flexibility=self.action_flexibility,
            beta=self.beta,
            eps=self.eps,
            use_batch_norm=self.use_batch_norm,
            use_quantile_regression=self.use_quantile_regression,
            use_gpu=self.use_gpu)

    def update(self, epoch, total_step, batch):
        value_loss, imitator_loss = self.impl.update(batch.observations,
                                                     batch.actions,
                                                     batch.next_rewards,
                                                     batch.next_observations,
                                                     batch.terminals)
        if total_step % self.target_update_interval == 0:
            self.impl.update_target()
        return value_loss, imitator_loss

    def _get_loss_labels(self):
        return ['value_loss', 'imitator_loss']
