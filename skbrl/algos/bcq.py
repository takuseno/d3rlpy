from .base import AlgoBase
from skbrl.algos.torch.bcq_impl import BCQImpl


class BCQ(AlgoBase):
    def __init__(self,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-3,
                 generator_learning_rate=1e-3,
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
                 use_batch_norm=True,
                 n_epochs=1000,
                 use_gpu=False,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.generator_learning_rate = generator_learning_rate
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
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = BCQImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            generator_learning_rate=self.generator_learning_rate,
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
            use_gpu=self.use_gpu)

    def update(self, epoch, total_step, batch):
        generator_loss = self.impl.update_generator(batch.observations,
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
        return critic_loss, actor_loss, generator_loss

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss', 'generator_loss']
