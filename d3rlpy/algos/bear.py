from .base import AlgoBase


class BEAR(AlgoBase):
    def __init__(self,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-3,
                 imitator_learning_rate=1e-3,
                 temp_learning_rate=1e-3,
                 alpha_learning_rate=1e-3,
                 batch_size=100,
                 gamma=0.99,
                 tau=0.005,
                 n_critics=2,
                 initial_temperature=1.0,
                 initial_alpha=1.0,
                 alpha_threshold=0.05,
                 lam=0.75,
                 n_action_samples=4,
                 mmd_sigma=20.0,
                 rl_start_epoch=0,
                 eps=1e-8,
                 use_batch_norm=False,
                 q_func_type='mean',
                 n_epochs=1000,
                 use_gpu=False,
                 scaler=None,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size, scaler)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.imitator_learning_rate = imitator_learning_rate
        self.temp_learning_rate = temp_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.n_critics = n_critics
        self.initial_temperature = initial_temperature
        self.initial_alpha = initial_alpha
        self.alpha_threshold = alpha_threshold
        self.lam = lam
        self.n_action_samples = n_action_samples
        self.mmd_sigma = mmd_sigma
        self.rl_start_epoch = rl_start_epoch
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.q_func_type = q_func_type
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        from .torch.bear_impl import BEARImpl
        self.impl = BEARImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            imitator_learning_rate=self.imitator_learning_rate,
            temp_learning_rate=self.temp_learning_rate,
            alpha_learning_rate=self.alpha_learning_rate,
            gamma=self.gamma,
            tau=self.tau,
            n_critics=self.n_critics,
            initial_temperature=self.initial_temperature,
            initial_alpha=self.initial_alpha,
            alpha_threshold=self.alpha_threshold,
            lam=self.lam,
            n_action_samples=self.n_action_samples,
            mmd_sigma=self.mmd_sigma,
            eps=self.eps,
            use_batch_norm=self.use_batch_norm,
            q_func_type=self.q_func_type,
            use_gpu=self.use_gpu,
            scaler=self.scaler)

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
            temp_loss, temp = self.impl.update_temperature(batch.observations)
            alpha_loss, alpha = self.impl.update_alpha(batch.observations)
            self.impl.update_actor_target()
            self.impl.update_critic_target()
        else:
            critic_loss = None
            actor_loss = None
            temp_loss = None
            temp = None
            alpha_loss = None
            alpha = None
        return critic_loss, actor_loss, imitator_loss, temp_loss, temp,\
               alpha_loss, alpha

    def _get_loss_labels(self):
        return [
            'critic_loss', 'actor_loss', 'imitator_loss', 'temp_loss', 'temp',
            'alpha_loss', 'alpha'
        ]
