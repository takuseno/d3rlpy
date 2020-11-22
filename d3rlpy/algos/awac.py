from .base import AlgoBase
from .torch.awac_impl import AWACImpl
from ..optimizers import AdamFactory


class AWAC(AlgoBase):
    r""" Advantage Weighted Actor-Critic algorithm.

    AWAC is a TD3-based actor-critic algorithm that enables efficient
    fine-tuning where the policy is trained with offline datasets and is
    deployed to online training.

    The policy is trained as a supervised regression.

    .. math::

        J(\phi) = \mathbb{E}_{s_t, a_t \sim D}
            [\log \pi_\phi(a_t|s_t)
                \exp(\frac{1}{\lambda} A^\pi (s_t, a_t))]

    where :math:`A^\pi (s_t, a_t) = Q_\theta(s_t, a_t) -
    Q_\theta(s_t, a'_t)` and :math:`a'_t \sim \pi_\phi(\cdot|s_t)`

    The key difference from AWR is that AWAC uses Q-function trained via TD
    learning for the better sample-efficiency.

    References:
        * `Nair et al., Accelerating Online Reinforcement Learning with Offline
          Datasets. <https://arxiv.org/abs/2006.09359>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        lam (float): :math:`\lambda` for weight calculation.
        n_action_samples (int): the number of sampled actions to calculate
            :math:`A^\pi(s_t, a_t)`.
        max_weight (float): maximum weight for cross-entropy loss.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        update_actor_interval (int): interval to update policy function.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function. Available options are
            `['mean', 'qr', 'iqn', 'fqf']`.
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
        impl (d3rlpy.algos.torch.sac_impl.SACImpl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        lam (float): :math:`\lambda` for weight calculation.
        n_action_samples (int): the number of sampled actions to calculate
            :math:`A^\pi(s_t, a_t)`.
        max_weight (float): maximum weight for cross-entropy loss.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        update_actor_interval (int): interval to update policy function.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        encoder_params (dict): optional arguments for encoder setup.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.sac_impl.SACImpl): algorithm implementation.

    """
    def __init__(self,
                 *,
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 actor_optim_factory=AdamFactory(weight_decay=1e-4),
                 critic_optim_factory=AdamFactory(),
                 batch_size=1024,
                 n_frames=1,
                 gamma=0.99,
                 tau=0.005,
                 lam=1.0,
                 n_action_samples=1,
                 max_weight=20.0,
                 n_critics=2,
                 bootstrap=False,
                 share_encoder=False,
                 update_actor_interval=1,
                 use_batch_norm=False,
                 q_func_type='mean',
                 use_gpu=False,
                 scaler=None,
                 augmentation=[],
                 n_augmentations=1,
                 encoder_params={},
                 dynamics=None,
                 impl=None,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         n_frames=n_frames,
                         scaler=scaler,
                         augmentation=augmentation,
                         dynamics=dynamics,
                         use_gpu=use_gpu)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_optim_factory = actor_optim_factory
        self.critic_optim_factory = critic_optim_factory
        self.gamma = gamma
        self.tau = tau
        self.lam = lam
        self.n_action_samples = n_action_samples
        self.max_weight = max_weight
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.update_actor_interval = update_actor_interval
        self.use_batch_norm = use_batch_norm
        self.q_func_type = q_func_type
        self.n_augmentations = n_augmentations
        self.encoder_params = encoder_params
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = AWACImpl(observation_shape=observation_shape,
                             action_size=action_size,
                             actor_learning_rate=self.actor_learning_rate,
                             critic_learning_rate=self.critic_learning_rate,
                             actor_optim_factory=self.actor_optim_factory,
                             critic_optim_factory=self.critic_optim_factory,
                             gamma=self.gamma,
                             tau=self.tau,
                             lam=self.lam,
                             n_action_samples=self.n_action_samples,
                             max_weight=self.max_weight,
                             n_critics=self.n_critics,
                             bootstrap=self.bootstrap,
                             share_encoder=self.share_encoder,
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
            actor_loss, mean_std = self.impl.update_actor(
                batch.observations, batch.actions)
            self.impl.update_critic_target()
            self.impl.update_actor_target()
        else:
            actor_loss, mean_std = None, None
        return critic_loss, actor_loss, mean_std

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss', 'mean_std']
