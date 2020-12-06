from .base import AlgoBase
from .torch.ddpg_impl import DDPGImpl
from ..optimizers import AdamFactory
from ..argument_utils import check_encoder
from ..argument_utils import check_use_gpu
from ..argument_utils import check_q_func
from ..argument_utils import check_augmentation


class DDPG(AlgoBase):
    r""" Deep Deterministic Policy Gradients algorithm.

    DDPG is an actor-critic algorithm that trains a Q function parametrized
    with :math:`\theta` and a policy function parametrized with :math:`\phi`.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma Q_{\theta'}(s_{t+1}, \pi_{\phi'}(s_{t+1}))
            - Q_\theta(s_t, a_t))^2]

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D} [Q_\theta(s_t, \pi_\phi(s_t))]

    where :math:`\theta'` and :math:`\phi` are the target network parameters.
    There target network parameters are updated every iteration.

    .. math::

        \theta' \gets \tau \theta + (1 - \tau) \theta'

        \phi' \gets \tau \phi + (1 - \tau) \phi'

    References:
        * `Silver et al., Deterministic policy gradient algorithms.
          <http://proceedings.mlr.press/v32/silver14.html>`_
        * `Lillicrap et al., Continuous control with deep reinforcement
          learning. <https://arxiv.org/abs/1509.02971>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q function.
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
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        reguralizing_rate (float): reguralizing term for policy function.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.ddpg_impl.DDPGImpl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q function.
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
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstraep Q functions.
        share_encoder (bool): flag to share encoder network.
        reguralizing_rate (float): reguralizing term for policy function.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.ddpg_impl.DDPGImpl): algorithm implementation.
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
                 n_critics=1,
                 bootstrap=False,
                 share_encoder=False,
                 reguralizing_rate=1e-10,
                 use_batch_norm=False,
                 use_gpu=False,
                 scaler=None,
                 augmentation=None,
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
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.reguralizing_rate = reguralizing_rate
        self.use_batch_norm = use_batch_norm
        self.augmentation = check_augmentation(augmentation)
        self.encoder_params = encoder_params
        self.use_gpu = check_use_gpu(use_gpu)
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = DDPGImpl(
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
            n_critics=self.n_critics,
            bootstrap=self.bootstrap,
            share_encoder=self.share_encoder,
            reguralizing_rate=self.reguralizing_rate,
            use_gpu=self.use_gpu,
            scaler=self.scaler,
            augmentation=self.augmentation)
        self.impl.build()

    def update(self, epoch, itr, batch):
        critic_loss = self.impl.update_critic(batch.observations,
                                              batch.actions,
                                              batch.next_rewards,
                                              batch.next_observations,
                                              batch.terminals, batch.n_steps)
        actor_loss = self.impl.update_actor(batch.observations)
        self.impl.update_critic_target()
        self.impl.update_actor_target()
        return critic_loss, actor_loss

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss']
