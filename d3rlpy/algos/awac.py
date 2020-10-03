from .base import AlgoBase
from .torch.awac_impl import AWACImpl


class AWAC(AlgoBase):
    def __init__(self,
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 batch_size=1024,
                 n_frames=1,
                 gamma=0.99,
                 tau=0.005,
                 lam=1.0,
                 n_action_samples=10,
                 max_weight=20.0,
                 actor_weight_decay=1e-4,
                 n_critics=2,
                 bootstrap=False,
                 share_encoder=False,
                 update_actor_interval=1,
                 eps=1e-4,
                 use_batch_norm=False,
                 q_func_type='mean',
                 n_epochs=1000,
                 use_gpu=False,
                 scaler=None,
                 augmentation=[],
                 n_augmentations=1,
                 encoder_params={},
                 dynamics=None,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size, n_frames, scaler, augmentation,
                         dynamics, use_gpu)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.lam = lam
        self.n_action_samples = n_action_samples
        self.max_weight = max_weight
        self.actor_weight_decay = actor_weight_decay
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.update_actor_interval = update_actor_interval
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.q_func_type = q_func_type
        self.n_epochs = n_epochs
        self.n_augmentations = n_augmentations
        self.encoder_params = encoder_params
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = AWACImpl(observation_shape=observation_shape,
                             action_size=action_size,
                             actor_learning_rate=self.actor_learning_rate,
                             critic_learning_rate=self.critic_learning_rate,
                             gamma=self.gamma,
                             tau=self.tau,
                             lam=self.lam,
                             n_action_samples=self.n_action_samples,
                             max_weight=self.max_weight,
                             actor_weight_decay=self.actor_weight_decay,
                             n_critics=self.n_critics,
                             bootstrap=self.bootstrap,
                             share_encoder=self.share_encoder,
                             eps=self.eps,
                             use_batch_norm=self.use_batch_norm,
                             q_func_type=self.q_func_type,
                             use_gpu=self.use_gpu,
                             scaler=self.scaler,
                             augmentation=self.augmentation,
                             n_augmentations=self.n_augmentations,
                             encoder_params=self.encoder_params)
        self.impl.build()

    def update(self, epoch, total_step, batch):
        critic_loss = self.impl.update_critic(batch.observations,
                                              batch.actions,
                                              batch.next_rewards,
                                              batch.next_observations,
                                              batch.terminals)
        # delayed policy update
        if total_step % self.update_actor_interval == 0:
            actor_loss = self.impl.update_actor(batch.observations,
                                                batch.actions)
            self.impl.update_critic_target()
            self.impl.update_actor_target()
        else:
            actor_loss = None
        return critic_loss, actor_loss

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss']
