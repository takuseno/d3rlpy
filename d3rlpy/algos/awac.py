from typing import Any, List, Optional, Sequence
from .base import AlgoBase, DataGenerator
from .torch.awac_impl import AWACImpl
from ..dataset import TransitionMiniBatch
from ..models.optimizers import OptimizerFactory, AdamFactory
from ..models.encoders import EncoderFactory
from ..models.q_functions import QFunctionFactory
from ..augmentation import AugmentationPipeline
from ..gpu import Device
from ..argument_utility import ScalerArg, ActionScalerArg
from ..argument_utility import check_encoder, EncoderArg
from ..argument_utility import check_use_gpu, UseGPUArg
from ..argument_utility import check_q_func, QFuncArg
from ..argument_utility import check_augmentation, AugmentationArg
from ..constants import IMPL_NOT_INITIALIZED_ERROR


class AWAC(AlgoBase):
    r"""Advantage Weighted Actor-Critic algorithm.

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
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
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
        target_reduction_type (str): ensemble reduction method at target value
            estimation. The available options are
            ``['min', 'max', 'mean', 'mix', 'none']``.
        update_actor_interval (int): interval to update policy function.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        generator (d3rlpy.algos.base.DataGenerator): dynamic dataset generator
            (e.g. model-based RL).
        impl (d3rlpy.algos.torch.sac_impl.SACImpl): algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _lam: float
    _n_action_samples: int
    _max_weight: float
    _share_encoder: bool
    _target_reduction_type: str
    _update_actor_interval: int
    _use_gpu: Optional[Device]
    _augmentation: AugmentationPipeline
    _impl: Optional[AWACImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(weight_decay=1e-4),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 1024,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        lam: float = 1.0,
        n_action_samples: int = 1,
        max_weight: float = 20.0,
        n_critics: int = 2,
        bootstrap: bool = False,
        share_encoder: bool = False,
        target_reduction_type: str = "min",
        update_actor_interval: int = 1,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        augmentation: AugmentationArg = None,
        generator: Optional[DataGenerator] = None,
        impl: Optional[AWACImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            bootstrap=bootstrap,
            n_critics=n_critics,
            scaler=scaler,
            action_scaler=action_scaler,
            generator=generator,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._max_weight = max_weight
        self._share_encoder = share_encoder
        self._target_reduction_type = target_reduction_type
        self._update_actor_interval = update_actor_interval
        self._augmentation = check_augmentation(augmentation)
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = AWACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            lam=self._lam,
            n_action_samples=self._n_action_samples,
            max_weight=self._max_weight,
            n_critics=self._n_critics,
            bootstrap=self._bootstrap,
            share_encoder=self._share_encoder,
            target_reduction_type=self._target_reduction_type,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            augmentation=self._augmentation,
        )
        self._impl.build()

    def update(
        self, epoch: int, total_step: int, batch: TransitionMiniBatch
    ) -> List[Optional[float]]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        critic_loss = self._impl.update_critic(
            batch.observations,
            batch.actions,
            batch.next_rewards,
            batch.next_observations,
            batch.terminals,
            batch.n_steps,
            batch.get_additional_data("mask"),
        )
        # delayed policy update
        if total_step % self._update_actor_interval == 0:
            actor_loss, mean_std = self._impl.update_actor(
                batch.observations, batch.actions
            )
            self._impl.update_critic_target()
            self._impl.update_actor_target()
        else:
            actor_loss, mean_std = None, None
        return [critic_loss, actor_loss, mean_std]

    def get_loss_labels(self) -> List[str]:
        return ["critic_loss", "actor_loss", "mean_std"]
