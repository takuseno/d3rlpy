from .base import AlgoBase
from .torch.bear_impl import BEARImpl


class BEAR(AlgoBase):
    """ Bootstrapping Error Accumulation Reduction algorithm.

    BEAR is a SAC-based data-driven deep reinforcement learning algorithm.

    BEAR constrains the support of the policy function within data distribution
    by minimizing Maximum Mean Discreptancy (MMD) between the policy function
    and the approximated beahvior policy function :math:`\\pi_\\beta(a|s)`
    which is optimized through L2 loss.

    .. math::

        L(\\beta) = \\mathbb{E}_{s_t, a_t \\sim D, a \\sim
            \\pi_\\beta(\\cdot|s_t)} [(a - a_t)^2]

    The policy objective is a combination of SAC's objective and MMD penalty.

    .. math::

        J(\\phi) = J_{SAC}(\\phi) - \\mathbb{E}_{s_t \sim D} \\alpha (
            \\text{MMD}(\\pi_\\beta(\\cdot|s_t), \\pi_\\phi(\\cdot|s_t))
            - \\epsilon)

    where MMD is computed as follows.

    .. math::

        \\text{MMD}(x, y) = \\frac{1}{N^2} \\sum_{i, i'} k(x_i, x_{i'})
            - \\frac{2}{NM} \\sum_{i, j} k(x_i, y_j)
            + \\frac{1}{M^2} \\sum_{j, j'} k(y_j, y_{j'})

    where :math:`k(x, y)` is a gaussian kernel
    :math:`k(x, y) = \\exp{((x - y)^2 / (2 \\sigma^2))}`.

    :math:`\\alpha` is also adjustable through dual gradient decsent where
    :math:`\\alpha` becomes smaller if MMD is smaller than the threshold
    :math:`\\epsilon`.

    References:
        * `Kumar et al., Stabilizing Off-Policy Q-Learning via Bootstrapping
          Error Reduction. <https://arxiv.org/abs/1906.00949>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        imitator_learning_rate (float): learning rate for behavior policy
            function.
        temp_learning_rate (float): learning rate for temperature parameter.
        alpha_learning_rate (float): learning rate for :math:`\\alpha`.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        update_actor_interval (int): interval to update policy function.
        initial_temperature (float): initial temperature value.
        initial_alpha (float): initial :math:`\\alpha` value.
        alpha_threshold (float): threshold value described as
            :math:`\\epsilon`.
        lam (float): weight for critic ensemble.
        n_action_samples (int): the number of action samples to estimate
            action-values.
        mmd_sigma (float): :math:`\\sigma` for gaussian kernel in MMD
            calculation.
        rl_start_epoch (int): epoch to start to update policy function and Q
            functions. If this is large, RL training would be more stabilized.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function. Avaiable options are
            `['mean', 'qr', 'iqn', 'fqf']`.
        n_epochs (int): the number of epochs to train.
        use_gpu (bool or d3rlpy.gpu.Device): flag to use GPU or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The avaiable options are `['pixel', 'min_max', 'standard']`.
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.bear_impl.BEARImpl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        imitator_learning_rate (float): learning rate for behavior policy
            function.
        temp_learning_rate (float): learning rate for temperature parameter.
        alpha_learning_rate (float): learning rate for :math:`\\alpha`.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        update_actor_interval (int): interval to update policy function.
        initial_temperature (float): initial temperature value.
        initial_alpha (float): initial :math:`\\alpha` value.
        alpha_threshold (float): threshold value described as
            :math:`\\epsilon`.
        lam (float): weight for critic ensemble.
        n_action_samples (int): the number of action samples to estimate
            action-values.
        mmd_sigma (float): :math:`\\sigma` for gaussian kernel in MMD
            calculation.
        rl_start_epoch (int): epoch to start to update policy function and Q
            functions. If this is large, RL training would be more stabilized.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function..
        n_epochs (int): the number of epochs to train.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.bear_impl.BEARImpl): algorithm implementation.

    """
    def __init__(self,
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 imitator_learning_rate=1e-3,
                 temp_learning_rate=3e-4,
                 alpha_learning_rate=1e-3,
                 batch_size=100,
                 gamma=0.99,
                 tau=0.005,
                 n_critics=2,
                 bootstrap=False,
                 share_encoder=False,
                 update_actor_interval=1,
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
                 augmentation=[],
                 n_augmentations=1,
                 dynamics=None,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size, scaler, augmentation, dynamics,
                         use_gpu)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.imitator_learning_rate = imitator_learning_rate
        self.temp_learning_rate = temp_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.update_actor_interval = update_actor_interval
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
        self.n_augmentations = n_augmentations
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
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
            bootstrap=self.bootstrap,
            share_encoder=self.share_encoder,
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
            scaler=self.scaler,
            augmentation=self.augmentation,
            n_augmentations=self.n_augmentations)

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
                temp_loss, temp = self.impl.update_temp(batch.observations)
                alpha_loss, alpha = self.impl.update_alpha(batch.observations)
                self.impl.update_actor_target()
                self.impl.update_critic_target()
            else:
                actor_loss = None
                temp_loss = None
                temp = None
                alpha_loss = None
                alpha = None
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
