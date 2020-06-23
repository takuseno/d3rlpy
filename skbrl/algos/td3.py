from .base import AlgoBase
from skbrl.algos.torch.td3_impl import TD3Impl


class TD3(AlgoBase):
    def __init__(self,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-3,
                 batch_size=100,
                 gamma=0.99,
                 tau=0.005,
                 reguralizing_rate=0.0,
                 n_critics=2,
                 target_smoothing_sigma=0.2,
                 target_smoothing_clip=0.5,
                 update_actor_interval=2,
                 eps=1e-8,
                 use_batch_norm=True,
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
        self.n_critics = n_critics
        self.target_smoothing_sigma = target_smoothing_sigma
        self.target_smoothing_clip = target_smoothing_clip
        self.update_actor_interval = update_actor_interval
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = TD3Impl(observation_shape=observation_shape,
                            action_size=action_size,
                            actor_learning_rate=self.actor_learning_rate,
                            critic_learning_rate=self.critic_learning_rate,
                            gamma=self.gamma,
                            tau=self.tau,
                            reguralizing_rate=self.reguralizing_rate,
                            n_critics=self.n_critics,
                            target_smoothing_sigma=self.target_smoothing_sigma,
                            target_smoothing_clip=self.target_smoothing_clip,
                            eps=self.eps,
                            use_batch_norm=self.use_batch_norm,
                            use_gpu=self.use_gpu)

    def update(self, epoch, itr, batch):
        critic_loss = self.impl.update_critic(batch.observations,
                                              batch.actions,
                                              batch.next_rewards,
                                              batch.next_observations,
                                              batch.terminals)
        # delayed policy update
        if itr % self.update_actor_interval == 0:
            actor_loss = self.impl.update_actor(batch.observations)
            self.impl.update_critic_target()
            self.impl.update_actor_target()
        else:
            actor_loss = None
        return critic_loss, actor_loss
