from .base import AlgoBase
from .torch.sac_impl import SACImpl, DiscreteSACImpl
from ..optimizers import AdamFactory
from ..argument_utils import check_encoder
from ..argument_utils import check_use_gpu
from ..argument_utils import check_augmentation
from ..argument_utils import check_q_func


class SAC(AlgoBase):
    """ Soft Actor-Critic algorithm.

    SAC is a DDPG-based maximum entropy RL algorithm, which produces
    state-of-the-art performance in online RL settings.
    SAC leverages twin Q functions proposed in TD3. Additionally,
    `delayed policy update` in TD3 is also implemented, which is not done in
    the paper.

    .. math::

        L(\\theta_i) = \\mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \\sim D,
                                   a_{t+1} \\sim \\pi_\\phi(\\cdot|s_{t+1})} [
            (y - Q_{\\theta_i}(s_t, a_t))^2]

    .. math::

        y = r_{t+1} + \\gamma (\\min_j Q_{\\theta_j}(s_{t+1}, a_{t+1})
            - \\alpha \\log (\\pi_\\phi(a_{t+1}|s_{t+1})))

    .. math::

        J(\\phi) = \\mathbb{E}_{s_t \\sim D, a_t \\sim \\pi_\\phi(\\cdot|s_t)}
            [\\alpha \\log (\\pi_\\phi (a_t|s_t))
              - \\min_i Q_{\\theta_i}(s_t, \\pi_\\phi(a_t|s_t))]

    The temperature parameter :math:`\\alpha` is also automatically adjustable.

    .. math::

        J(\\alpha) = \\mathbb{E}_{s_t \\sim D, a_t \\sim \\pi_\\phi\(\cdot|s_t)}
            [-\\alpha (\\log (\\pi_\\phi(a_t|s_t)) + H)]

    where :math:`H` is a target
    entropy, which is defined as :math:`\dim a`.

    References:
        * `Haarnoja et al., Soft Actor-Critic: Off-Policy Maximum Entropy Deep
          Reinforcement Learning with a Stochastic Actor.
          <https://arxiv.org/abs/1801.01290>`_
        * `Haarnoja et al., Soft Actor-Critic Algorithms and Applications.
          <https://arxiv.org/abs/1812.05905>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float): learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        update_actor_interval (int): interval to update policy function.
        initial_temperature (float): initial temperature value.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.sac_impl.SACImpl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float): learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.encoders.EncoderFactory):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.encoders.EncoderFactory):
            encoder factory for the critic.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        update_actor_interval (int): interval to update policy function.
        initial_temperature (float): initial temperature value.
        q_func_type (str): type of Q function.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.sac_impl.SACImpl): algorithm implementation.
        eval_results_ (dict): evaluation results.

    """
    def __init__(self,
                 *,
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 temp_learning_rate=3e-4,
                 actor_optim_factory=AdamFactory(),
                 critic_optim_factory=AdamFactory(),
                 temp_optim_factory=AdamFactory(),
                 actor_encoder_factory='default',
                 critic_encoder_factory='default',
                 q_func_factory='mean',
                 batch_size=100,
                 n_frames=1,
                 gamma=0.99,
                 tau=0.005,
                 n_critics=2,
                 bootstrap=False,
                 share_encoder=False,
                 update_actor_interval=2,
                 initial_temperature=1.0,
                 use_gpu=False,
                 scaler=None,
                 augmentation=[],
                 n_augmentations=1,
                 dynamics=None,
                 impl=None,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         n_frames=n_frames,
                         scaler=scaler,
                         dynamics=dynamics)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.temp_learning_rate = temp_learning_rate
        self.actor_optim_factory = actor_optim_factory
        self.critic_optim_factory = critic_optim_factory
        self.temp_optim_factory = temp_optim_factory
        self.actor_encoder_factory = check_encoder(actor_encoder_factory)
        self.critic_encoder_factory = check_encoder(critic_encoder_factory)
        self.q_func_factory = check_q_func(q_func_factory)
        self.gamma = gamma
        self.tau = tau
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.update_actor_interval = update_actor_interval
        self.initial_temperature = initial_temperature
        self.augmentation = check_augmentation(augmentation)
        self.n_augmentations = n_augmentations
        self.use_gpu = check_use_gpu(use_gpu)
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = SACImpl(observation_shape=observation_shape,
                            action_size=action_size,
                            actor_learning_rate=self.actor_learning_rate,
                            critic_learning_rate=self.critic_learning_rate,
                            temp_learning_rate=self.temp_learning_rate,
                            actor_optim_factory=self.actor_optim_factory,
                            critic_optim_factory=self.critic_optim_factory,
                            temp_optim_factory=self.temp_optim_factory,
                            actor_encoder_factory=self.actor_encoder_factory,
                            critic_encoder_factory=self.critic_encoder_factory,
                            q_func_factory=self.q_func_factory,
                            gamma=self.gamma,
                            tau=self.tau,
                            n_critics=self.n_critics,
                            bootstrap=self.bootstrap,
                            share_encoder=self.share_encoder,
                            initial_temperature=self.initial_temperature,
                            use_gpu=self.use_gpu,
                            scaler=self.scaler,
                            augmentation=self.augmentation,
                            n_augmentations=self.n_augmentations)
        self.impl.build()

    def update(self, epoch, total_step, batch):
        critic_loss = self.impl.update_critic(batch.observations,
                                              batch.actions,
                                              batch.next_rewards,
                                              batch.next_observations,
                                              batch.terminals)
        # delayed policy update
        if total_step % self.update_actor_interval == 0:
            actor_loss = self.impl.update_actor(batch.observations)
            temp_loss, temp = self.impl.update_temp(batch.observations)
            self.impl.update_critic_target()
            self.impl.update_actor_target()
        else:
            actor_loss = None
            temp_loss = None
            temp = None
        return critic_loss, actor_loss, temp_loss, temp

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss', 'temp_loss', 'temp']


class DiscreteSAC(AlgoBase):
    r""" Soft Actor-Critic algorithm for discrete action-space.

    This discrete version of SAC is built based on continuous version of SAC
    with additional modifications.

    The target state-value is calculated as expectation of all action-values.

    .. math::

        V(s_t) = \pi_\phi (s_t)^T [Q_\theta(s_t) - \alpha \log (\pi_\phi (s_t))]

    Similarly, the objective function for the temperature parameter is as
    follows.

    .. math::

        J(\alpha) = \pi_\phi (s_t)^T [-\alpha (\log(\pi_\phi (s_t)) + H)]

    Finally, the objective function for the policy function is as follows.

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D}
            [\pi_\phi(s_t)^T [\alpha \log(\pi_\phi(s_t)) - Q_\theta(s_t)]]

    References:
        * `Christodoulou, Soft Actor-Critic for Discrete Action Settings.
          <https://arxiv.org/abs/1910.07207>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float): learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        initial_temperature (float): initial temperature value.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.torch.sac_impl.SACImpl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float): learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.encoders.EncoderFactory):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.encoders.EncoderFactory):
            encoder factory for the critic.
        q_func_factory (d3rlpy.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        initial_temperature (float): initial temperature value.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.sac_impl.SACImpl): algorithm implementation.
        eval_results_ (dict): evaluation results.

    """
    def __init__(self,
                 *,
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 temp_learning_rate=3e-4,
                 actor_optim_factory=AdamFactory(eps=1e-4),
                 critic_optim_factory=AdamFactory(eps=1e-4),
                 temp_optim_factory=AdamFactory(eps=1e-4),
                 actor_encoder_factory='default',
                 critic_encoder_factory='default',
                 q_func_factory='mean',
                 batch_size=64,
                 n_frames=1,
                 gamma=0.99,
                 n_critics=2,
                 bootstrap=False,
                 share_encoder=False,
                 initial_temperature=1.0,
                 target_update_interval=8000,
                 use_gpu=False,
                 scaler=None,
                 augmentation=None,
                 n_augmentations=1,
                 dynamics=None,
                 impl=None,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         n_frames=n_frames,
                         scaler=scaler,
                         dynamics=dynamics)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.temp_learning_rate = temp_learning_rate
        self.actor_optim_factory = actor_optim_factory
        self.critic_optim_factory = critic_optim_factory
        self.temp_optim_factory = temp_optim_factory
        self.actor_encoder_factory = check_encoder(actor_encoder_factory)
        self.critic_encoder_factory = check_encoder(critic_encoder_factory)
        self.q_func_factory = check_q_func(q_func_factory)
        self.gamma = gamma
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.initial_temperature = initial_temperature
        self.target_update_interval = target_update_interval
        self.augmentation = check_augmentation(augmentation)
        self.n_augmentations = n_augmentations
        self.use_gpu = check_use_gpu(use_gpu)
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = DiscreteSACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            temp_learning_rate=self.temp_learning_rate,
            actor_optim_factory=self.actor_optim_factory,
            critic_optim_factory=self.critic_optim_factory,
            temp_optim_factory=self.temp_optim_factory,
            actor_encoder_factory=self.actor_encoder_factory,
            critic_encoder_factory=self.critic_encoder_factory,
            q_func_factory=self.q_func_factory,
            gamma=self.gamma,
            n_critics=self.n_critics,
            bootstrap=self.bootstrap,
            share_encoder=self.share_encoder,
            initial_temperature=self.initial_temperature,
            use_gpu=self.use_gpu,
            scaler=self.scaler,
            augmentation=self.augmentation,
            n_augmentations=self.n_augmentations)
        self.impl.build()

    def update(self, epoch, total_step, batch):
        critic_loss = self.impl.update_critic(batch.observations,
                                              batch.actions,
                                              batch.next_rewards,
                                              batch.next_observations,
                                              batch.terminals)
        actor_loss = self.impl.update_actor(batch.observations)
        temp_loss, temp = self.impl.update_temp(batch.observations)
        if total_step % self.target_update_interval == 0:
            self.impl.update_target()
        return critic_loss, actor_loss, temp_loss, temp

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss', 'temp_loss', 'temp']
