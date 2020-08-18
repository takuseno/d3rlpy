from .base import AlgoBase
from .torch.bcq_impl import BCQImpl, DiscreteBCQImpl


class BCQ(AlgoBase):
    """ Batch-Constrained Q-learning algorithm.

    BCQ is the very first practical data-driven deep reinforcement learning 
    lgorithm.
    The major difference from DDPG is that the policy function is represented
    as combination of conditional VAE and perturbation function in order to
    remedy extrapolation error emerging from target value estimation.

    The encoder and the decoder of the conditional VAE is represented as
    :math:`E_\omega` and :math:`D_\omega` respectively.

    .. math::

        L(\omega) = E_{s_t, a_t \sim D} [(a - \\tilde{a})^2
            + D_{KL}(N(\mu, \sigma)|N(0, 1))]

    where :math:`\mu, \sigma = E_\omega(s_t, a_t)`,
    :math:`\\tilde{a} = D_\omega(s_t, z)` and :math:`z \sim N(\mu, \sigma)`.

    The policy function is represented as a residual function
    with the VAE and the perturbation function represented as
    :math:`\\xi_\phi (s, a)`.

    .. math::

        \pi(s, a) = a + \Phi \\xi_\phi (s, a)

    where :math:`a = D_\omega (s, z)`, :math:`z \sim N(0, 0.5)` and
    :math:`\Phi` is a perturbation scale designated by `action_flexibility`.
    Although the policy is learned closely to data distribution, the
    perturbation function can lead to more rewarded states.

    BCQ also leverages twin Q functions and computes weighted average over
    maximum values and minimum values.

    .. math::

        L(\\theta_i) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D}
            [(y - Q_{\\theta_i}(s_t, a_t))^2]

    .. math::

        y = r_{t+1} + \gamma \max_{a_i} [
            \lambda \min_j Q_{\\theta_j'}(s_{t+1}, a_i)
            + (1 - \lambda) \max_j Q_{\\theta_j'}(s_{t+1}, a_i)]

    where :math:`\{a_i \sim D(s_{t+1}, z), z \sim N(0, 0.5)\}_{i=1}^n`.
    The number of sampled actions is designated with `n_action_samples`.

    Finally, the perturbation function is trained just like DDPG's policy
    function.

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D, a_t \sim D_\omega(s_t, z),
                              z \sim N(0, 0.5)}
            [Q_{\\theta_1} (s_t, \pi(s_t, a_t))]

    At inference time, action candidates are sampled as many as
    `n_action_samples`, and the action with highest value estimation is taken.

    .. math::

        \pi'(s) = \\text{argmax}_{\pi(s, a_i)} Q_{\\theta_1} (s, \pi(s, a_i))

    Note:
        The greedy action is not deterministic because the action candidates
        are always randomly sampled. This might affect `save_policy` method and
        the performance at production.

    References:
        * `Fujimoto et al., Off-Policy Deep Reinforcement Learning without
          Exploration. <https://arxiv.org/abs/1812.02900>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        imitator_learning_rate (float): learning rate for Conditional VAE.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        update_actor_interval (int): interval to update policy function.
        lam (float): weight factor for critic ensemble.
        n_action_samples (int): the number of action samples to estimate
            action-values.
        action_flexibility (float): output scale of perturbation function
            represented as :math:`\Phi`.
        rl_start_epoch (int): epoch to start to update policy function and Q
            functions. If this is large, RL training would be more stabilized.
        latent_size (int): size of latent vector for Conditional VAE.
        beta (float): KL reguralization term for Conditional VAE.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function. Available options are
            `['mean', 'qr', 'iqn', 'fqf']`.
        n_epochs (int): the number of eopchs to train.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        encoder_params (dict): optional arguments for encoder setup. If the
            observation is pixel, you can pass ``filters`` with list of tuples
            consisting with ``(filter_size, kernel_size, stride)`` and
            ``feature_size`` with an integer scaler for the last linear layer
            size. If the observation is vector, you can pass ``hidden_units``
            with list of hidden unit sizes.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.bcq_impl.BCQImpl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        imitator_learning_rate (float): learning rate for Conditional VAE.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        update_actor_interval (int): interval to update policy function.
        lam (float): weight factor for critic ensemble.
        n_action_samples (int): the number of action samples to estimate
            action-values.
        action_flexibility (float): output scale of perturbation function.
        rl_start_epoch (int): epoch to start to update policy function and Q
            functions.
        latent_size (int): size of latent vector for Conditional VAE.
        beta (float): KL reguralization term for Conditional VAE.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function.
        n_epochs (int): the number of eopchs to train.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        encoder_params (dict): optional arguments for encoder setup.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.bcq_impl.BCQImpl): algorithm implementation.

    """
    def __init__(self,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-3,
                 imitator_learning_rate=1e-3,
                 batch_size=100,
                 gamma=0.99,
                 tau=0.005,
                 n_critics=2,
                 bootstrap=False,
                 share_encoder=False,
                 update_actor_interval=1,
                 lam=0.75,
                 n_action_samples=100,
                 action_flexibility=0.05,
                 rl_start_epoch=0,
                 latent_size=32,
                 beta=0.5,
                 eps=1e-8,
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
        super().__init__(n_epochs, batch_size, scaler, augmentation, dynamics,
                         use_gpu)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.imitator_learning_rate = imitator_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.update_actor_interval = update_actor_interval
        self.lam = lam
        self.n_action_samples = n_action_samples
        self.action_flexibility = action_flexibility
        self.rl_start_epoch = rl_start_epoch
        self.latent_size = latent_size
        self.beta = beta
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.q_func_type = q_func_type
        self.n_augmentations = n_augmentations
        self.encoder_params = encoder_params
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = BCQImpl(observation_shape=observation_shape,
                            action_size=action_size,
                            actor_learning_rate=self.actor_learning_rate,
                            critic_learning_rate=self.critic_learning_rate,
                            imitator_learning_rate=self.imitator_learning_rate,
                            gamma=self.gamma,
                            tau=self.tau,
                            n_critics=self.n_critics,
                            bootstrap=self.bootstrap,
                            share_encoder=self.share_encoder,
                            lam=self.lam,
                            n_action_samples=self.n_action_samples,
                            action_flexibility=self.action_flexibility,
                            latent_size=self.latent_size,
                            beta=self.beta,
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
        imitator_loss = self.impl.update_imitator(batch.observations,
                                                  batch.actions)
        if epoch >= self.rl_start_epoch:
            critic_loss = self.impl.update_critic(batch.observations,
                                                  batch.actions,
                                                  batch.next_rewards,
                                                  batch.next_observations,
                                                  batch.terminals)
            if total_step % self.update_actor_interval == 0:
                actor_loss = self.impl.update_actor(batch.observations)
                self.impl.update_actor_target()
                self.impl.update_critic_target()
            else:
                actor_loss = None
        else:
            critic_loss = None
            actor_loss = None
        return critic_loss, actor_loss, imitator_loss

    def sample_action(self, x):
        """ BCQ does not support sampling action.
        """
        raise NotImplementedError('BCQ does not support sampling action.')

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss', 'imitator_loss']


class DiscreteBCQ(AlgoBase):
    """ Discrete version of Batch-Constrained Q-learning algorithm.

    Discrete version takes theories from the continuous version, but the
    algorithm is much simpler than that.
    The imitation function :math:`G_\omega(a|s)` is trained as supervised
    learning just like Behavior Cloning.

    .. math::

        L(\omega) = \mathbb{E}_{a_t, s_t \sim D}
            [-\sum_a p(a|s_t) \log G_\omega(a|s_t)]

    With this imitation function, the greedy policy is defined as follows.

    .. math::

        \pi(s_t) = \\text{argmax}_{a|G_\omega(a|s_t)
                / \max_{\\tilde{a}} G_\omega(\\tilde{a}|s_t) > \\tau}
            Q_\\theta (s_t, a)

    which eliminates actions with probabilities :math:`\\tau` times smaller
    than the maximum one.

    Finally, the loss function is computed in Double DQN style with the above
    constrained policy.

    .. math::

        L(\\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma Q_{\\theta'}(s_{t+1}, \pi(s_{t+1}))
            - Q_\\theta(s_t, a_t))^2]

    References:
        * `Fujimoto et al., Off-Policy Deep Reinforcement Learning without
          Exploration. <https://arxiv.org/abs/1812.02900>`_
        * `Fujimoto et al., Benchmarking Batch Deep Reinforcement Learning
          Algorithms. <https://arxiv.org/abs/1910.01708>`_

    Args:
        learning_rate (float): learning rate.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        action_flexibility (float): probability threshold represented as
            :math:`\tau`.
        beta (float): reguralization term for imitation function.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        target_update_interval (int): interval to update the target network.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function. Available options are
            `['mean', 'qr', 'iqn', 'fqf']`.
        n_epochs (int): the number of epochs to train.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        encoder_params (dict): optional arguments for encoder setup. If the
            observation is pixel, you can pass ``filters`` with list of tuples
            consisting with ``(filter_size, kernel_size, stride)`` and
            ``feature_size`` with an integer scaler for the last linear layer
            size. If the observation is vector, you can pass ``hidden_units``
            with list of hidden unit sizes.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.bcq_impl.DiscreteBCQImpl):
            algorithm implementation.

    Attributes:
        learning_rate (float): learning rate.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        action_flexibility (float): probability threshold represented as
            :math:`\tau`.
        beta (float): reguralization term for imitation function.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        target_update_interval (int): interval to update the target network.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function.
        n_epochs (int): the number of epochs to train.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        encoder_params (dict): optional arguments for encoder setup.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.bcq_impl.DiscreteBCQImpl):
            algorithm implementation.

    """
    def __init__(self,
                 learning_rate=6.25e-5,
                 batch_size=32,
                 gamma=0.99,
                 n_critics=1,
                 bootstrap=False,
                 share_encoder=False,
                 action_flexibility=0.3,
                 beta=0.5,
                 eps=1.5e-4,
                 target_update_interval=8e3,
                 use_batch_norm=True,
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
        super().__init__(n_epochs, batch_size, scaler, augmentation, dynamics,
                         use_gpu)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.action_flexibility = action_flexibility
        self.beta = beta
        self.eps = eps
        self.target_update_interval = target_update_interval
        self.use_batch_norm = use_batch_norm
        self.q_func_type = q_func_type
        self.n_augmentations = n_augmentations
        self.encoder_params = encoder_params
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = DiscreteBCQImpl(observation_shape=observation_shape,
                                    action_size=action_size,
                                    learning_rate=self.learning_rate,
                                    gamma=self.gamma,
                                    n_critics=self.n_critics,
                                    bootstrap=self.bootstrap,
                                    share_encoder=self.share_encoder,
                                    action_flexibility=self.action_flexibility,
                                    beta=self.beta,
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
        loss = self.impl.update(batch.observations, batch.actions,
                                batch.next_rewards, batch.next_observations,
                                batch.terminals)
        if total_step % self.target_update_interval == 0:
            self.impl.update_target()
        return [loss]

    def _get_loss_labels(self):
        return ['loss']
