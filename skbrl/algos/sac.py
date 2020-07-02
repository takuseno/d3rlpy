from .base import AlgoBase
from skbrl.algos.torch.sac_impl import SACImpl


class SAC(AlgoBase):
    def __init__(self,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-3,
                 temp_learning_rate=5e-3,
                 batch_size=100,
                 gamma=0.99,
                 tau=0.005,
                 n_critics=2,
                 update_actor_interval=2,
                 initial_temperature=1.0,
                 eps=1e-8,
                 use_batch_norm=False,
                 use_quantile_regression=False,
                 n_epochs=1000,
                 use_gpu=False,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.temp_learning_rate = temp_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.n_critics = n_critics
        self.update_actor_interval = update_actor_interval
        self.initial_temperature = initial_temperature
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.use_quantile_regression = use_quantile_regression
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = SACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            temp_learning_rate=self.temp_learning_rate,
            gamma=self.gamma,
            tau=self.tau,
            n_critics=self.n_critics,
            initial_temperature=self.initial_temperature,
            eps=self.eps,
            use_batch_norm=self.use_batch_norm,
            use_quantile_regression=self.use_quantile_regression,
            use_gpu=self.use_gpu)

    def update(self, epoch, total_step, batch):
        critic_loss = self.impl.update_critic(batch.observations,
                                              batch.actions,
                                              batch.next_rewards,
                                              batch.next_observations,
                                              batch.terminals)
        # delayed policy update
        if total_step % self.update_actor_interval == 0:
            actor_loss = self.impl.update_actor(batch.observations)
            temp_loss = self.impl.update_temperature(batch.observations)
            self.impl.update_critic_target()
            self.impl.update_actor_target()
        else:
            actor_loss = None
            temp_loss = None
        return critic_loss, actor_loss, temp_loss

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss', 'temp_loss']
