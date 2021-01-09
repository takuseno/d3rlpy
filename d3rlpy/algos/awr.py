from typing import Any, List, Optional, Union, Sequence
from abc import abstractmethod

import numpy as np

from .base import AlgoBase, DataGenerator
from .torch.awr_impl import AWRBaseImpl, AWRImpl, DiscreteAWRImpl
from ..augmentation import AugmentationPipeline
from ..dataset import compute_lambda_return, TransitionMiniBatch
from ..models.optimizers import OptimizerFactory, SGDFactory
from ..models.encoders import EncoderFactory
from ..gpu import Device
from ..argument_utility import check_encoder, check_use_gpu, check_augmentation
from ..argument_utility import ScalerArg, EncoderArg, UseGPUArg
from ..argument_utility import AugmentationArg


class _AWRBase(AlgoBase):

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _batch_size_per_update: int
    _n_actor_updates: int
    _n_critic_updates: int
    _lam: float
    _beta: float
    _max_weight: float
    _augmentation: AugmentationPipeline
    _use_gpu: Optional[Device]
    _impl: Optional[AWRBaseImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 5e-5,
        critic_learning_rate: float = 1e-4,
        actor_optim_factory: OptimizerFactory = SGDFactory(momentum=0.9),
        critic_optim_factory: OptimizerFactory = SGDFactory(momentum=0.9),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        batch_size: int = 2048,
        n_frames: int = 1,
        gamma: float = 0.99,
        batch_size_per_update: int = 256,
        n_actor_updates: int = 1000,
        n_critic_updates: int = 200,
        lam: float = 0.95,
        beta: float = 1.0,
        max_weight: float = 20.0,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        augmentation: AugmentationArg = None,
        generator: Optional[DataGenerator] = None,
        impl: Optional[AWRImpl] = None,
        **kwargs: Any
    ):
        # batch_size in AWR has different semantic from Q learning algorithms.
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=1,
            gamma=gamma,
            scaler=scaler,
            generator=generator,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._batch_size_per_update = batch_size_per_update
        self._n_actor_updates = n_actor_updates
        self._n_critic_updates = n_critic_updates
        self._lam = lam
        self._beta = beta
        self._max_weight = max_weight
        self._augmentation = check_augmentation(augmentation)
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    @abstractmethod
    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        pass

    def _compute_lambda_returns(self, batch: TransitionMiniBatch) -> np.ndarray:
        # compute TD(lambda)
        lambda_returns = []
        for transition in batch.transitions:
            lambda_return = compute_lambda_return(
                transition=transition,
                algo=self,
                gamma=self._gamma,
                lam=self._lam,
                n_frames=self._n_frames,
            )
            lambda_returns.append(lambda_return)
        return np.array(lambda_returns).reshape((-1, 1))

    def _compute_advantages(
        self, returns: np.ndarray, batch: TransitionMiniBatch
    ) -> np.ndarray:
        baselines = self.predict_value(batch.observations).reshape((-1, 1))
        advantages = returns - baselines
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages)
        return (advantages - adv_mean) / (adv_std + 1e-5)

    def _compute_clipped_weights(self, advantages: np.ndarray) -> np.ndarray:
        weights = np.exp(advantages / self._beta)
        return np.minimum(weights, self._max_weight)

    def predict_value(  # pylint: disable=signature-differs
        self, x: Union[np.ndarray, List[Any]], *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Returns predicted state values.

        Args:
            x: observations.

        Returns:
            predicted state values.

        """
        assert self._impl is not None
        return self._impl.predict_value(x)

    def update(
        self, epoch: int, total_step: int, batch: TransitionMiniBatch
    ) -> List[Optional[float]]:
        assert self._impl is not None

        # compute lmabda return
        lambda_returns = self._compute_lambda_returns(batch)

        # calcuate advantage
        advantages = self._compute_advantages(lambda_returns, batch)

        # compute weights
        clipped_weights = self._compute_clipped_weights(advantages)

        n_steps_per_batch = self.batch_size // self._batch_size_per_update

        # update critic
        critic_loss_history = []
        for _ in range(self._n_critic_updates // n_steps_per_batch):
            for j in range(n_steps_per_batch):
                head_index = j * self._batch_size_per_update
                tail_index = head_index + self._batch_size_per_update
                observations = batch.observations[head_index:tail_index]
                returns = lambda_returns[head_index:tail_index]
                critic_loss = self._impl.update_critic(observations, returns)
                critic_loss_history.append(critic_loss)
        critic_loss_mean = np.mean(critic_loss_history)

        # update actor
        actor_loss_history = []
        for _ in range(self._n_actor_updates // n_steps_per_batch):
            for j in range(n_steps_per_batch):
                head_index = j * self._batch_size_per_update
                tail_index = head_index + self._batch_size_per_update
                observations = batch.observations[head_index:tail_index]
                actions = batch.actions[head_index:tail_index]
                weights = clipped_weights[head_index:tail_index]
                actor_loss = self._impl.update_actor(
                    observations, actions, weights
                )
                actor_loss_history.append(actor_loss)
        actor_loss_mean = np.mean(actor_loss_history)

        return [critic_loss_mean, actor_loss_mean, np.mean(clipped_weights)]

    def _get_loss_labels(self) -> List[str]:
        return ["critic_loss", "actor_loss", "weights"]


class AWR(_AWRBase):
    r"""Advantage-Weighted Regression algorithm.

    AWR is an actor-critic algorithm that trains via supervised regression way,
    and has shown strong performance in online and offline settings.

    The value function is trained as a supervised regression problem.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, R_t \sim D} [(R_t - V(s_t|\theta))^2]

    where :math:`R_t` is approximated using TD(:math:`\lambda`) to mitigate
    high variance issue.

    The policy function is also trained as a supervised regression problem.

    .. math::

        J(\phi) = \mathbb{E}_{s_t, a_t, R_t \sim D}
            [\log \pi(a_t|s_t, \phi)
                \exp (\frac{1}{B} (R_t - V(s_t|\theta)))]

    where :math:`B` is a constant factor.

    References:
        * `Peng et al., Advantage-Weighted Regression: Simple and Scalable
          Off-Policy Reinforcement Learning
          <https://arxiv.org/abs/1910.00177>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for value function.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        batch_size (int): batch size per iteration.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        batch_size_per_update (int): mini-batch size.
        n_actor_updates (int): actor gradient steps per iteration.
        n_critic_updates (int): critic gradient steps per iteration.
        lam (float): :math:`\lambda`  for TD(:math:`\lambda`).
        beta (float): :math:`B` for weight scale.
        max_weight (float): :math:`w_{\text{max}}` for weight clipping.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        generator (d3rlpy.algos.base.DataGenerator): dynamic dataset generator
            (e.g. model-based RL).
        impl (d3rlpy.algos.torch.awr_impl.AWRImpl): algorithm implementation.

    """

    _impl: Optional[AWRImpl]

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = AWRImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            augmentation=self._augmentation,
        )
        self._impl.build()


class DiscreteAWR(_AWRBase):
    r"""Discrete veriosn of Advantage-Weighted Regression algorithm.

    AWR is an actor-critic algorithm that trains via supervised regression way,
    and has shown strong performance in online and offline settings.

    The value function is trained as a supervised regression problem.

    .. math::

        L(\theta) = \mathbb{E}_{s_t, R_t \sim D} [(R_t - V(s_t|\theta))^2]

    where :math:`R_t` is approximated using TD(:math:`\lambda`) to mitigate
    high variance issue.

    The policy function is also trained as a supervised regression problem.

    .. math::

        J(\phi) = \mathbb{E}_{s_t, a_t, R_t \sim D}
            [\log \pi(a_t|s_t, \phi)
                \exp (\frac{1}{B} (R_t - V(s_t|\theta)))]

    where :math:`B` is a constant factor.

    References:
        * `Peng et al., Advantage-Weighted Regression: Simple and Scalable
          Off-Policy Reinforcement Learning
          <https://arxiv.org/abs/1910.00177>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for value function.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        batch_size (int): batch size per iteration.
        n_frames (int): the number of frames to stack for image observation.
        gamma (float): discount factor.
        batch_size_per_update (int): mini-batch size.
        n_actor_updates (int): actor gradient steps per iteration.
        n_critic_updates (int): critic gradient steps per iteration.
        lam (float): :math:`\lambda`  for TD(:math:`\lambda`).
        beta (float): :math:`B` for weight scale.
        max_weight (float): :math:`w_{\text{max}}` for weight clipping.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        generator (d3rlpy.algos.base.DataGenerator): dynamic dataset generator
            (e.g. model-based RL).
        impl (d3rlpy.algos.torch.awr_impl.DiscreteAWRImpl):
            algorithm implementation.

    """

    _impl: Optional[DiscreteAWRImpl]

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DiscreteAWRImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            augmentation=self._augmentation,
        )
        self._impl.build()
