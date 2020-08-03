from abc import abstractmethod
from .base import AlgoBase
from .sac import ISACImpl
from .dqn import DoubleDQN


class ICQLImpl(ISACImpl):
    @abstractmethod
    def update_alpha(self, obs_t, act_t):
        pass


class CQL(AlgoBase):
    """ Conservative Q-Learning algorithm.

    CQL is a SAC-based data-driven deep reinforcement learning algorithm, which
    achieves state-of-the-art performance in offline RL problems.

    CQL mitigates overestimation error by minimizing action-values under the
    current policy and maximizing values under data distribution for
    underestimation issue.

    .. math::

        L(\\theta_i) = \\alpha \\mathbb{E}_{s_t \sim D}
            [\\log{\\sum_a \exp{Q_{\\theta_i}(s_t, a)}}
             - \mathbb{E}_{a \sim D} [Q_{\\theta_i}(s, a)] - \\tau]
            + L_{SAC}(\\theta_i)

    where :math:`\\alpha` is an automatically adjustable value via Lagrangian
    dual gradient descent and :math:`\\tau` is a threshold value.
    If the action-value difference is smaller than :math:`\\tau`, the
    :math:`\\alpha` will become smaller.
    Otherwise, the :math:`\\alpha` will become larger to aggressively penalize
    action-values.

    In continuous control, :math:`\\log{\\sum_a \\exp{Q(s, a)}}` is computed as
    follows.

    .. math::

        \\log{\\sum_a \\exp{Q(s, a)}} \\approx \log{(
            \\frac{1}{2N} \\sum_{a_i \sim \\text{Unif}(a)}^N
                [\\frac{\\exp{Q(s, a_i)}}{\\text{Unif}(a)}]
            + \\frac{1}{2N} \\sum_{a_i \sim \pi_\\phi(a|s)}^N
                [\\frac{\\exp{Q(s, a_i)}}{\\pi_\\phi(a_i|s)}])}

    where :math:`N` is the number of sampled actions.

    The rest of optimization is exactly same as :class:`d3rlpy.algos.SAC`.

    References:
        * `Kumar et al., Conservative Q-Learning for Offline Reinforcement
          Learning. <https://arxiv.org/abs/2006.04779>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float):
            learning rate for temperature parameter of SAC.
        alpha_learning_rate (float): learning rate for :math:`\\alpha`.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        update_actor_interval (int): interval to update policy function.
        initial_temperature (float): initial temperature value.
        initial_alpha (float): initial :math:`\\alpha` value.
        alpha_threshold (float): threshold value described as :math:`\\tau`.
        n_action_samples (int): the number of sampled actions to compute
            :math:`\\log{\\sum_a \\exp{Q(s, a)}}`.
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
        impl (d3rlpy.algos.cql.ICQLImpl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float):
            learning rate for temperature parameter of SAC.
        alpha_learning_rate (float): learning rate for :math:`\\alpha`.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        update_actor_interval (int): interval to update policy function.
        initial_temperature (float): initial temperature value.
        initial_alpha (float): initial :math:`\\alpha` value.
        alpha_threshold (float): threshold value described as :math:`\\tau`.
        n_action_samples (int): the number of sampled actions to compute
            :math:`\\log{\\sum_a \\exp{Q(s, a)}}`.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function.
        n_epochs (int): the number of epochs to train.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.cql.ICQLImpl): algorithm implementation.

    """
    def __init__(self,
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 temp_learning_rate=3e-4,
                 alpha_learning_rate=3e-4,
                 batch_size=100,
                 gamma=0.99,
                 tau=0.005,
                 n_critics=2,
                 bootstrap=False,
                 update_actor_interval=1,
                 initial_temperature=1.0,
                 initial_alpha=5.0,
                 alpha_threshold=10.0,
                 n_action_samples=10,
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
        self.alpha_learning_rate = alpha_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.n_critics = n_critics
        self.bootstrap = bootstrap
        self.update_actor_interval = update_actor_interval
        self.initial_temperature = initial_temperature
        self.initial_alpha = initial_alpha
        self.alpha_threshold = alpha_threshold
        self.n_action_samples = n_action_samples
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.q_func_type = q_func_type
        self.n_epochs = n_epochs
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        from .torch.cql_impl import CQLImpl
        self.impl = CQLImpl(observation_shape=observation_shape,
                            action_size=action_size,
                            actor_learning_rate=self.actor_learning_rate,
                            critic_learning_rate=self.critic_learning_rate,
                            temp_learning_rate=self.temp_learning_rate,
                            alpha_learning_rate=self.alpha_learning_rate,
                            gamma=self.gamma,
                            tau=self.tau,
                            n_critics=self.n_critics,
                            bootstrap=self.bootstrap,
                            initial_temperature=self.initial_temperature,
                            initial_alpha=self.initial_alpha,
                            alpha_threshold=self.alpha_threshold,
                            n_action_samples=self.n_action_samples,
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
        if total_step % self.update_actor_interval == 0:
            actor_loss = self.impl.update_actor(batch.observations)
            temp_loss, temp = self.impl.update_temp(batch.observations)
            alpha_loss, alpha = self.impl.update_alpha(batch.observations,
                                                       batch.actions)
            self.impl.update_critic_target()
            self.impl.update_actor_target()
        else:
            actor_loss = None
            temp_loss = None
            temp = None
            alpha_loss = None
            alpha = None

        return critic_loss, actor_loss, temp_loss, temp, alpha_loss, alpha

    def _get_loss_labels(self):
        return [
            'critic_loss', 'actor_loss', 'temp_loss', 'temp', 'alpha_loss',
            'alpha'
        ]


class DiscreteCQL(DoubleDQN):
    """ Discrete version of Conservative Q-Learning algorithm.

    Discrete version of CQL is a DoubleDQN-based data-driven deep reinforcement
    learning algorithm (the original paper uses DQN), which achieves
    state-of-the-art performance in offline RL problems.

    CQL mitigates overestimation error by minimizing action-values under the
    current policy and maximizing values under data distribution for
    underestimation issue.

    .. math::

        L(\\theta) = \\mathbb{E}_{s_t \sim D}
            [\\log{\\sum_a \exp{Q_{\\theta}(s_t, a)}}
             - \mathbb{E}_{a \sim D} [Q_{\\theta}(s, a)]]
            + L_{DoubleDQN}(\\theta)

    References:
        * `Kumar et al., Conservative Q-Learning for Offline Reinforcement
          Learning. <https://arxiv.org/abs/2006.04779>`_

    Args:
        learning_rate (float): learning rate.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        eps (float): :math:`\epsilon` for Adam optimizer.
        target_update_interval (int): interval to synchronize the target
            network.
        use_batch_norm (bool): flag to insert batch normalization layers
        q_func_type (str): type of Q function. Available options are
            `['mean', 'qr', 'iqn', 'fqf']`.
        n_epochs (int): the number of epochs to train.
        use_gpu (bool or d3rlpy.gpu.Device): flag to use GPU or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model for data
            augmentation.
        impl (d3rlpy.algos.dqn.IDQNImpl): algorithm implementation.

    Attributes:
        learning_rate (float): learning rate.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        eps (float): :math:`\epsilon` for Adam optimizer.
        target_update_interval (int): interval to synchronize the target
            network.
        use_batch_norm (bool): flag to insert batch normalization layers
        q_func_type (str): type of Q function.
        n_epochs (int): the number of epochs to train.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.dqn.IDQNImpl): algorithm implementation.

    """
    def create_impl(self, observation_shape, action_size):
        from .torch.cql_impl import DiscreteCQLImpl
        self.impl = DiscreteCQLImpl(observation_shape=observation_shape,
                                    action_size=action_size,
                                    learning_rate=self.learning_rate,
                                    gamma=self.gamma,
                                    n_critics=self.n_critics,
                                    bootstrap=self.bootstrap,
                                    eps=self.eps,
                                    use_batch_norm=self.use_batch_norm,
                                    q_func_type=self.q_func_type,
                                    use_gpu=self.use_gpu,
                                    scaler=self.scaler)
