from .base import AlgoBase
from .torch.td3_impl import TD3Impl
from ..optimizers import AdamFactory
from ..argument_utils import check_encoder
from ..argument_utils import check_use_gpu
from ..argument_utils import check_augmentation
from ..argument_utils import check_q_func


class TD3(AlgoBase):
    r""" Twin Delayed Deep Deterministic Policy Gradients algorithm.

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

        L(\theta_i) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma \min_j Q_{\theta_j'}(s_{t+1}, \pi_{\phi'}(s_{t+1}) +
            \epsilon) - Q_{\theta_i}(s_t, a_t))^2]

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D}
            [\min_i Q_{\theta_i}(s_t, \pi_\phi(s_t))]

    where :math:`\epsilon \sim clip (N(0, \sigma), -c, c)`

    References:
        * `Fujimoto et al., Addressing Function Approximation Error in
          Actor-Critic Methods. <https://arxiv.org/abs/1802.09477>`_

    Args:
        actor_learning_rate (float): learning rate for a policy function.
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
        reguralizing_rate (float): reguralizing term for policy function.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        target_smoothing_sigma (float): standard deviation for target noise.
        target_smoothing_clip (float): clipping range for target noise.
        update_actor_interval (int): interval to update policy function
            described as `delayed policy update` in the paper.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.td3_impl.TD3Impl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for a policy function.
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
        reguralizing_rate (float): reguralizing term for policy function.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        target_smoothing_sigma (float): standard deviation for target noise.
        target_smoothing_clip (float): clipping range for target noise.
        update_actor_interval (int): interval to update policy function
            described as `delayed policy update` in the paper.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.td3_impl.TD3Impl): algorithm implementation.
        eval_results_ (dict): evaluation results.

    """
    def __init__(self,
                 *,
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 actor_optim_factory=AdamFactory(),
                 critic_optim_factory=AdamFactory(),
                 actor_encoder_factory='default',
                 critic_encoder_factory='default',
                 q_func_factory='mean',
                 batch_size=100,
                 n_frames=1,
                 n_steps=1,
                 gamma=0.99,
                 tau=0.005,
                 reguralizing_rate=0.0,
                 n_critics=2,
                 bootstrap=False,
                 share_encoder=False,
                 target_smoothing_sigma=0.2,
                 target_smoothing_clip=0.5,
                 update_actor_interval=2,
                 use_batch_norm=False,
                 use_gpu=False,
                 scaler=None,
                 augmentation=[],
                 encoder_params={},
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
        self.reguralizing_rate = reguralizing_rate
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.target_smoothing_sigma = target_smoothing_sigma
        self.target_smoothing_clip = target_smoothing_clip
        self.update_actor_interval = update_actor_interval
        self.use_batch_norm = use_batch_norm
        self.augmentation = check_augmentation(augmentation)
        self.encoder_params = encoder_params
        self.use_gpu = check_use_gpu(use_gpu)
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = TD3Impl(observation_shape=observation_shape,
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
                            reguralizing_rate=self.reguralizing_rate,
                            n_critics=self.n_critics,
                            bootstrap=self.bootstrap,
                            share_encoder=self.share_encoder,
                            target_smoothing_sigma=self.target_smoothing_sigma,
                            target_smoothing_clip=self.target_smoothing_clip,
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
            actor_loss = self.impl.update_actor(batch.observations)
            self.impl.update_critic_target()
            self.impl.update_actor_target()
        else:
            actor_loss = None
        return critic_loss, actor_loss

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss']
