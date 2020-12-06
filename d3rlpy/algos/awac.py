from .base import AlgoBase
from .torch.awac_impl import AWACImpl
from ..optimizers import AdamFactory
from ..argument_utils import check_encoder
from ..argument_utils import check_use_gpu
from ..argument_utils import check_q_func
from ..argument_utils import check_augmentation


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
        actor_encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
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
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
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
        actor_encoder_factory (d3rlpy.encoders.EncoderFactory):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.encoders.EncoderFactory):
            encoder factory for the critic.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
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
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
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
                 actor_encoder_factory='default',
                 critic_encoder_factory='default',
                 q_func_factory='mean',
                 batch_size=1024,
                 n_frames=1,
                 n_steps=1,
                 gamma=0.99,
                 tau=0.005,
                 lam=1.0,
                 n_action_samples=1,
                 max_weight=20.0,
                 n_critics=2,
                 bootstrap=False,
                 share_encoder=False,
                 update_actor_interval=1,
                 use_gpu=False,
                 scaler=None,
                 augmentation=None,
                 dynamics=None,
                 impl=None,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         n_frames=n_frames,
                         n_steps=n_steps,
                         gamma=gamma,
                         scaler=scaler,
                         dynamics=dynamics)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_optim_factory = actor_optim_factory
        self.critic_optim_factory = critic_optim_factory
        self.actor_encoder_factory = check_encoder(actor_encoder_factory)
        self.critic_encoder_factory = check_encoder(critic_encoder_factory)
        self.q_func_factory = check_q_func(q_func_factory)
        self.tau = tau
        self.lam = lam
        self.n_action_samples = n_action_samples
        self.max_weight = max_weight
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.update_actor_interval = update_actor_interval
        self.augmentation = check_augmentation(augmentation)
        self.use_gpu = check_use_gpu(use_gpu)
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = AWACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            actor_optim_factory=self.actor_optim_factory,
            critic_optim_factory=self.critic_optim_factory,
            actor_encoder_factory=self.actor_encoder_factory,
            critic_encoder_factory=self.critic_encoder_factory,
            q_func_factory=self.q_func_factory,
            gamma=self.gamma,
            tau=self.tau,
            lam=self.lam,
            n_action_samples=self.n_action_samples,
            max_weight=self.max_weight,
            n_critics=self.n_critics,
            bootstrap=self.bootstrap,
            share_encoder=self.share_encoder,
            use_gpu=self.use_gpu,
            scaler=self.scaler,
            augmentation=self.augmentation)
        self.impl.build()

    def update(self, epoch, total_step, batch):
        critic_loss = self.impl.update_critic(batch.observations,
                                              batch.actions,
                                              batch.next_rewards,
                                              batch.next_observations,
                                              batch.terminals, batch.n_steps)
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
