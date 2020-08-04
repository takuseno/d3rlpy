from abc import ABCMeta, abstractmethod
from .base import AlgoBase
from .ddpg import IDDPGImpl


class ISACImpl(IDDPGImpl):
    @abstractmethod
    def update_temp(self, obs_t):
        pass


class SAC(AlgoBase):
    """ Soft Actor-Critic algorithm.

    SAC is a DDPG-based maximum entropy RL algorithm, which produces
    state-of-the-art performance in online RL settings.
    SAC leverages twin Q functions proposed in TD3. Additionally,
    `delayed policy update` in TD3 is also implemented, which is not done in
    the paper.

    .. math::

        L(\\theta_i) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D,
                                   a_{t+1} \sim \pi_\phi(\cdot|s_{t+1})} [
            (y - Q_{\\theta_i}(s_t, a_t))^2]

    .. math::

        y = r_{t+1} + \gamma (\min_j Q_{\\theta_j}(s_{t+1}, a_{t+1})
            - \\alpha \log (\pi_\phi(a_{t+1}|s_{t+1})))

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D, a_t \sim \pi_\phi(\cdot|s_t)}
            [\\alpha \log (\pi_\phi (a_t|s_t))
              - \min_i Q_{\\theta_i}(s_t, \pi_\phi(a_t|s_t))]

    The temperature parameter :math:`\\alpha` is also automatically adjustable.

    .. math::

        J(\\alpha) = \mathbb{E}_{s_t \sim D, a_t \sim \pi_\phi(\cdot|s_t)}
            [-\\alpha (\log (\pi_\phi(a_t|s_t)) + H)]

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
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        update_actor_interval (int): interval to update policy function.
        initial_temperature (float): initial temperature value.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function. Available options are
            `['mean', 'qr', 'iqn', 'fqf']`.
        n_epochs (int): the number of epochs to train.
        use_gpu (bool or d3rlpy.gpu.Device): flag to use GPU or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.sac.ISACImpl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float): learning rate for temperature parameter.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        update_actor_interval (int): interval to update policy function.
        initial_temperature (float): initial temperature value.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function.
        n_epochs (int): the number of epochs to train.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.sac.ISACImpl): algorithm implementation.

    """
    def __init__(self,
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 temp_learning_rate=3e-4,
                 batch_size=100,
                 gamma=0.99,
                 tau=0.005,
                 n_critics=2,
                 bootstrap=False,
                 share_encoder=False,
                 update_actor_interval=2,
                 initial_temperature=1.0,
                 eps=1e-8,
                 use_batch_norm=False,
                 q_func_type='mean',
                 n_epochs=1000,
                 use_gpu=False,
                 scaler=None,
                 dynamics=None,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size, scaler, dynamics, use_gpu)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.temp_learning_rate = temp_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.share_encoder = share_encoder
        self.update_actor_interval = update_actor_interval
        self.initial_temperature = initial_temperature
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.q_func_type = q_func_type
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        from .torch.sac_impl import SACImpl
        self.impl = SACImpl(observation_shape=observation_shape,
                            action_size=action_size,
                            actor_learning_rate=self.actor_learning_rate,
                            critic_learning_rate=self.critic_learning_rate,
                            temp_learning_rate=self.temp_learning_rate,
                            gamma=self.gamma,
                            tau=self.tau,
                            n_critics=self.n_critics,
                            bootstrap=self.bootstrap,
                            share_encoder=self.share_encoder,
                            initial_temperature=self.initial_temperature,
                            eps=self.eps,
                            use_batch_norm=self.use_batch_norm,
                            q_func_type=self.q_func_type,
                            use_gpu=self.use_gpu,
                            scaler=self.scaler)

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
