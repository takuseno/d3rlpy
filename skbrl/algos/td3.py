from .base import AlgoBase
from .ddpg import IDDPGImpl


class TD3(AlgoBase):
    """ Twin Delayed Deep Deterministic Policy Gradients algorithm.

    TD3 is an improved DDPG-based algorithm.
    Major differences from DDPG are as follows.

    * TD3 has twin Q functions to reduce overestimation bias at TD learning.
      The number of Q functions can be designated by `n_critics`.
    * TD3 adds noise to target value estimation to avoid overfitting with the
      deterministic policy. 
    * TD3 updates the policy function after several Q function updates in order
      to reduce variance of action-value estimation. The interval of the policy
      function update can be designated by `update_actor_interval`.

    .. math::

        L(\\theta_i) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma \min_j Q_{\\theta_j'}(s_{t+1}, \pi_{\phi'}(s_{t+1}) +
            \epsilon) - Q_{\\theta_i}(s_t, a_t))^2]

    .. math::

        J(\\phi) = \mathbb{E}_{s_t \sim D}
            [\min_i Q_{\\theta_i}(s_t, \pi_\phi(s_t))]

    where :math:`\\epsilon \sim clip (N(0, \\sigma), -c, c)`

    References:
        * `Fujimoto et al., Addressing Function Approximation Error in
          Actor-Critic Methods. <https://arxiv.org/abs/1802.09477>`_

    Args:
        actor_learning_rate (float): learning rate for a policy function.
        critic_learning_rate (float): learning rate for Q functions.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        reguralizing_rate (float): reguralizing term for policy function.
        n_critics (int): the number of Q functions for ensemble.
        target_smoothing_sigma (float): standard deviation for target noise.
        target_smoothing_clip (float): clipping range for target noise.
        update_actor_interval (int): interval to update policy function
            described as `delayed policy update` in the paper.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
        distribution_type (str): type of distributional Q function.
            If None, the normal Q function will be used. Available options are
            `['qr', 'iqn']`.
        n_epochs (int): the number of epochs to train.
        use_gpu (bool): flag to use GPU.
        impl (skbrl.algos.ddpg.IDDPGImpl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for a policy function.
        critic_learning_rate (float): learning rate for Q functions.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        reguralizing_rate (float): reguralizing term for policy function.
        n_critics (int): the number of Q functions for ensemble.
        target_smoothing_sigma (float): standard deviation for target noise.
        target_smoothing_clip (float): clipping range for target noise.
        update_actor_interval (int): interval to update policy function
            described as `delayed policy update` in the paper.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
        distribution_type (str): type of distributional Q function..
        n_epochs (int): the number of epochs to train.
        use_gpu (bool): flag to use GPU.
        impl (skbrl.algos.ddpg.IDDPGImpl): algorithm implementation.

    """
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
                 use_batch_norm=False,
                 distribution_type=None,
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
        self.distribution_type = distribution_type
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        from skbrl.algos.torch.td3_impl import TD3Impl
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
                            distribution_type=self.distribution_type,
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
            self.impl.update_critic_target()
            self.impl.update_actor_target()
        else:
            actor_loss = None
        return critic_loss, actor_loss

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss']
