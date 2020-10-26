import numpy as np

from d3rlpy.dataset import compute_lambda_return
from .base import AlgoBase
from .torch.awr_impl import AWRImpl, DiscreteAWRImpl


class AWR(AlgoBase):
    """ Advantage-Weighted Regression algorithm.

    AWR is an actor-critic algorithm that trains via supervised regression way,
    and has shown strong performance in online and offline settings.

    The value function is trained as a supervised regression problem.

    .. math::

        L(\\theta) = \\mathbb{E}_{s_t, R_t \\sim D} [(R_t - V(s_t|\\theta))^2]

    where :math:`R_t` is approximated using TD(:math:`\\lambda`) to mitigate
    high variance issue.

    The policy function is also trained as a supervised regression problem.

    .. math::

        J(\\phi) = \\mathbb{E}_{s_t, a_t, R_t \\sim D}
            [\\log \\pi(a_t|s_t, \\phi)
                \\exp (\\frac{1}{B} (R_t - V(s_t|\\theta)))]

    where :math:`B` is a constant factor.

    References:
        * `Peng et al., Advantage-Weighted Regression: Simple and Scalable
          Off-Policy Reinforcement Learning
          <https://arxiv.org/abs/1910.00177>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for value function.
        batch_size (int): batch size per iteration.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        batch_size_per_update (int): mini-batch size.
        n_actor_updates (int): actor gradient steps per iteration.
        n_critic_updates (int): critic gradient steps per iteration.
        lam (float): :math:`\\lambda`  for TD(:math:`\\lambda`).
        beta (float): :math:`B` for weight scale.
        max_weight (float): :math:`w_{\\text{max}}` for weight clipping.
        momentum (float): momentum for stochastic gradient descent.
        use_batch_norm (bool): flag to insert batch normalization layers.
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
        impl (d3rlpy.algos.torch.awr_impl.AWRImpl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for value function.
        batch_size (int): batch size per iteration.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        batch_size_per_update (int): mini-batch size.
        n_actor_updates (int): actor gradient steps per iteration.
        n_critic_updates (int): critic gradient steps per iteration.
        lam (float): :math:`\\lambda`  for TD(:math:`\\lambda`).
        beta (float): :math:`B` for weight scale.
        max_weight (float): :math:`w_{\\text{max}}` for weight clipping.
        momentum (float): momentum for stochastic gradient descent.
        use_batch_norm (bool): flag to insert batch normalization layers.
        n_epochs (int): the number of epochs to train.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        encoder_params (dict): optional arguments for encoder setup.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.awr_impl.AWRImpl): algorithm implementation.
        eval_results_ (dict): evaluation results.

    """
    def __init__(self,
                 *,
                 actor_learning_rate=5e-5,
                 critic_learning_rate=1e-4,
                 batch_size=2048,
                 n_frames=1,
                 gamma=0.99,
                 batch_size_per_update=256,
                 n_actor_updates=1000,
                 n_critic_updates=200,
                 lam=0.95,
                 beta=0.05,
                 max_weight=20.0,
                 momentum=0.9,
                 use_batch_norm=False,
                 n_epochs=1000,
                 use_gpu=False,
                 scaler=None,
                 augmentation=[],
                 n_augmentations=1,
                 encoder_params={},
                 dynamics=None,
                 impl=None,
                 **kwargs):
        # batch_size in AWR has different semantic from Q learning algorithms.
        super().__init__(n_epochs=n_epochs,
                         batch_size=batch_size,
                         n_frames=n_frames,
                         scaler=scaler,
                         augmentation=augmentation,
                         dynamics=dynamics,
                         use_gpu=use_gpu)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.batch_size_per_update = batch_size_per_update
        self.n_actor_updates = n_actor_updates
        self.n_critic_updates = n_critic_updates
        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.max_weight = max_weight
        self.use_batch_norm = use_batch_norm
        self.momentum = momentum
        self.n_augmentations = n_augmentations
        self.encoder_params = encoder_params
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = AWRImpl(observation_shape=observation_shape,
                            action_size=action_size,
                            actor_learning_rate=self.actor_learning_rate,
                            critic_learning_rate=self.critic_learning_rate,
                            use_batch_norm=self.use_batch_norm,
                            momentum=self.momentum,
                            use_gpu=self.use_gpu,
                            scaler=self.scaler,
                            augmentation=self.augmentation,
                            n_augmentations=self.n_augmentations,
                            encoder_params=self.encoder_params)
        self.impl.build()

    def _compute_lambda_returns(self, batch):
        # compute TD(lambda)
        lambda_returns = []
        for transition in batch.transitions:
            lambda_return = compute_lambda_return(transition=transition,
                                                  algo=self,
                                                  gamma=self.gamma,
                                                  lam=self.lam,
                                                  n_frames=self.n_frames)
            lambda_returns.append(lambda_return)
        return np.array(lambda_returns).reshape((-1, 1))

    def _compute_advantages(self, returns, batch):
        baselines = self.predict_value(batch.observations).reshape((-1, 1))
        advantages = returns - baselines
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages)
        return (advantages - adv_mean) / (adv_std + 1e-5)

    def _compute_clipped_weights(self, advantages):
        weights = np.exp(advantages / self.beta)
        return np.minimum(weights, self.max_weight)

    def predict_value(self, x, *args, **kwargs):
        """ Returns predicted state values.

        Args:
            x (numpy.ndarray): observations.

        Returns:
            numpy.ndarray: predicted state values.

        """
        return self.impl.predict_value(x)

    def update(self, epoch, itr, batch):
        # compute lmabda return
        lambda_returns = self._compute_lambda_returns(batch)

        # calcuate advantage
        advantages = self._compute_advantages(lambda_returns, batch)

        # compute weights
        clipped_weights = self._compute_clipped_weights(advantages)

        n_steps_per_batch = self.batch_size // self.batch_size_per_update

        # update critic
        critic_loss_history = []
        for i in range(self.n_critic_updates // n_steps_per_batch):
            for j in range(n_steps_per_batch):
                head_index = j * self.batch_size_per_update
                tail_index = head_index + self.batch_size_per_update
                observations = batch.observations[head_index:tail_index]
                returns = lambda_returns[head_index:tail_index]
                critic_loss = self.impl.update_critic(observations, returns)
                critic_loss_history.append(critic_loss)
        critic_loss_mean = np.mean(critic_loss_history)

        # update actor
        actor_loss_history = []
        for i in range(self.n_actor_updates // n_steps_per_batch):
            for j in range(n_steps_per_batch):
                head_index = j * self.batch_size_per_update
                tail_index = head_index + self.batch_size_per_update
                observations = batch.observations[head_index:tail_index]
                actions = batch.actions[head_index:tail_index]
                weights = clipped_weights[head_index:tail_index]
                actor_loss = self.impl.update_actor(observations, actions,
                                                    weights)
                actor_loss_history.append(actor_loss)
        actor_loss_mean = np.mean(actor_loss_history)

        return critic_loss_mean, actor_loss_mean, np.mean(clipped_weights)

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss', 'weights']


class DiscreteAWR(AWR):
    """ Discrete veriosn of Advantage-Weighted Regression algorithm.

    AWR is an actor-critic algorithm that trains via supervised regression way,
    and has shown strong performance in online and offline settings.

    The value function is trained as a supervised regression problem.

    .. math::

        L(\\theta) = \\mathbb{E}_{s_t, R_t \\sim D} [(R_t - V(s_t|\\theta))^2]

    where :math:`R_t` is approximated using TD(:math:`\\lambda`) to mitigate
    high variance issue.

    The policy function is also trained as a supervised regression problem.

    .. math::

        J(\\phi) = \\mathbb{E}_{s_t, a_t, R_t \\sim D}
            [\\log \\pi(a_t|s_t, \\phi)
                \\exp (\\frac{1}{B} (R_t - V(s_t|\\theta)))]

    where :math:`B` is a constant factor.

    References:
        * `Peng et al., Advantage-Weighted Regression: Simple and Scalable
          Off-Policy Reinforcement Learning
          <https://arxiv.org/abs/1910.00177>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for value function.
        batch_size (int): batch size per iteration.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        batch_size_per_update (int): mini-batch size.
        n_actor_updates (int): actor gradient steps per iteration.
        n_critic_updates (int): critic gradient steps per iteration.
        lam (float): :math:`\\lambda`  for TD(:math:`\\lambda`).
        beta (float): :math:`B` for weight scale.
        max_weight (float): :math:`w_{\\text{max}}` for weight clipping.
        momentum (float): momentum for stochastic gradient descent.
        use_batch_norm (bool): flag to insert batch normalization layers.
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
        impl (d3rlpy.algos.torch.awr_impl.DiscreteAWRImpl):
            algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for value function.
        batch_size (int): batch size per iteration.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        batch_size_per_update (int): mini-batch size.
        n_actor_updates (int): actor gradient steps per iteration.
        n_critic_updates (int): critic gradient steps per iteration.
        lam (float): :math:`\\lambda`  for TD(:math:`\\lambda`).
        beta (float): :math:`B` for weight scale.
        max_weight (float): :math:`w_{\\text{max}}` for weight clipping.
        momentum (float): momentum for stochastic gradient descent.
        use_batch_norm (bool): flag to insert batch normalization layers.
        n_epochs (int): the number of epochs to train.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        augmentation (d3rlpy.augmentation.AugmentationPipeline):
            augmentation pipeline.
        n_augmentations (int): the number of data augmentations to update.
        encoder_params (dict): optional arguments for encoder setup.
        dynamics (d3rlpy.dynamics.base.DynamicsBase): dynamics model.
        impl (d3rlpy.algos.torch.awr_impl.DiscreteAWRImpl):
            algorithm implementation.
        eval_results_ (dict): evaluation results.

    """
    def create_impl(self, observation_shape, action_size):
        self.impl = DiscreteAWRImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            use_batch_norm=self.use_batch_norm,
            momentum=self.momentum,
            use_gpu=self.use_gpu,
            scaler=self.scaler,
            augmentation=self.augmentation,
            n_augmentations=self.n_augmentations,
            encoder_params=self.encoder_params)
        self.impl.build()
