from typing import Any, List, Optional, Sequence
from .base import AlgoBase, DataGenerator
from .torch.bear_impl import BEARImpl
from ..dataset import TransitionMiniBatch
from ..optimizers import OptimizerFactory, AdamFactory
from ..encoders import EncoderFactory
from ..q_functions import QFunctionFactory
from ..gpu import Device
from ..augmentation import AugmentationPipeline
from ..argument_utility import check_encoder, EncoderArg
from ..argument_utility import check_use_gpu, UseGPUArg
from ..argument_utility import check_augmentation, AugmentationArg
from ..argument_utility import check_q_func, QFuncArg
from ..argument_utility import ScalerArg


class BEAR(AlgoBase):
    r"""Bootstrapping Error Accumulation Reduction algorithm.

    BEAR is a SAC-based data-driven deep reinforcement learning algorithm.

    BEAR constrains the support of the policy function within data distribution
    by minimizing Maximum Mean Discreptancy (MMD) between the policy function
    and the approximated beahvior policy function :math:`\pi_\beta(a|s)`
    which is optimized through L2 loss.

    .. math::

        L(\beta) = \mathbb{E}_{s_t, a_t \sim D, a \sim
            \pi_\beta(\cdot|s_t)} [(a - a_t)^2]

    The policy objective is a combination of SAC's objective and MMD penalty.

    .. math::

        J(\phi) = J_{SAC}(\phi) - \mathbb{E}_{s_t \sim D} \alpha (
            \text{MMD}(\pi_\beta(\cdot|s_t), \pi_\phi(\cdot|s_t))
            - \epsilon)

    where MMD is computed as follows.

    .. math::

        \text{MMD}(x, y) = \frac{1}{N^2} \sum_{i, i'} k(x_i, x_{i'})
            - \frac{2}{NM} \sum_{i, j} k(x_i, y_j)
            + \frac{1}{M^2} \sum_{j, j'} k(y_j, y_{j'})

    where :math:`k(x, y)` is a gaussian kernel
    :math:`k(x, y) = \exp{((x - y)^2 / (2 \sigma^2))}`.

    :math:`\alpha` is also adjustable through dual gradient decsent where
    :math:`\alpha` becomes smaller if MMD is smaller than the threshold
    :math:`\epsilon`.

    References:
        * `Kumar et al., Stabilizing Off-Policy Q-Learning via Bootstrapping
          Error Reduction. <https://arxiv.org/abs/1906.00949>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        imitator_learning_rate (float): learning rate for behavior policy
            function.
        temp_learning_rate (float): learning rate for temperature parameter.
        alpha_learning_rate (float): learning rate for :math:`\alpha`.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        imitator_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the behavior policy.
        temp_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        alpha_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            optimizer factory for :math:`\alpha`.
        actor_encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory for the critic.
        imitator_encoder_factory (d3rlpy.encoders.EncoderFactory or str):
            encoder factory for the behavior policy.
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
        update_actor_interval (int): interval to update policy function.
        initial_temperature (float): initial temperature value.
        initial_alpha (float): initial :math:`\alpha` value.
        alpha_threshold (float): threshold value described as
            :math:`\epsilon`.
        lam (float): weight for critic ensemble.
        n_action_samples (int): the number of action samples to estimate
            action-values.
        mmd_sigma (float): :math:`\sigma` for gaussian kernel in MMD
            calculation.
        rl_start_epoch (int): epoch to start to update policy function and Q
            functions. If this is large, RL training would be more stabilized.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device iD or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The avaiable options are `['pixel', 'min_max', 'standard']`.
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        generator (d3rlpy.algos.base.DataGenerator): dynamic dataset generator
            (e.g. model-based RL).
        impl (d3rlpy.algos.torch.bear_impl.BEARImpl): algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _imitator_learning_rate: float
    _temp_learning_rate: float
    _alpha_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _imitator_optim_factory: OptimizerFactory
    _temp_optim_factory: OptimizerFactory
    _alpha_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _imitator_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _bootstrap: bool
    _share_encoder: bool
    _update_actor_interval: int
    _initial_temperature: float
    _initial_alpha: float
    _alpha_threshold: float
    _lam: float
    _n_action_samples: int
    _mmd_sigma: float
    _rl_start_epoch: int
    _augmentation: AugmentationPipeline
    _use_gpu: Optional[Device]
    _impl: Optional[BEARImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        imitator_learning_rate: float = 1e-3,
        temp_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 1e-3,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        imitator_optim_factory: OptimizerFactory = AdamFactory(),
        temp_optim_factory: OptimizerFactory = AdamFactory(),
        alpha_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        imitator_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        bootstrap: bool = False,
        share_encoder: bool = False,
        update_actor_interval: int = 1,
        initial_temperature: float = 1.0,
        initial_alpha: float = 1.0,
        alpha_threshold: float = 0.05,
        lam: float = 0.75,
        n_action_samples: int = 4,
        mmd_sigma: float = 20.0,
        rl_start_epoch: int = 0,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        augmentation: AugmentationArg = None,
        generator: Optional[DataGenerator] = None,
        impl: Optional[BEARImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            generator=generator,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._imitator_learning_rate = imitator_learning_rate
        self._temp_learning_rate = temp_learning_rate
        self._alpha_learning_rate = alpha_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._imitator_optim_factory = imitator_optim_factory
        self._temp_optim_factory = temp_optim_factory
        self._alpha_optim_factory = alpha_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._imitator_encoder_factory = check_encoder(imitator_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._bootstrap = bootstrap
        self._share_encoder = share_encoder
        self._update_actor_interval = update_actor_interval
        self._initial_temperature = initial_temperature
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._mmd_sigma = mmd_sigma
        self._rl_start_epoch = rl_start_epoch
        self._augmentation = check_augmentation(augmentation)
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = BEARImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            imitator_learning_rate=self._imitator_learning_rate,
            temp_learning_rate=self._temp_learning_rate,
            alpha_learning_rate=self._alpha_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            imitator_optim_factory=self._imitator_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            alpha_optim_factory=self._alpha_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            imitator_encoder_factory=self._imitator_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            bootstrap=self._bootstrap,
            share_encoder=self._share_encoder,
            initial_temperature=self._initial_temperature,
            initial_alpha=self._initial_alpha,
            alpha_threshold=self._alpha_threshold,
            lam=self._lam,
            n_action_samples=self._n_action_samples,
            mmd_sigma=self._mmd_sigma,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            augmentation=self._augmentation,
        )
        self._impl.build()

    def update(
        self, epoch: int, total_step: int, batch: TransitionMiniBatch
    ) -> List[Optional[float]]:
        assert self._impl is not None

        imitator_loss = self._impl.update_imitator(
            batch.observations, batch.actions
        )
        if epoch >= self._rl_start_epoch:
            critic_loss = self._impl.update_critic(
                batch.observations,
                batch.actions,
                batch.next_rewards,
                batch.next_observations,
                batch.terminals,
                batch.n_steps,
            )
            if total_step % self._update_actor_interval == 0:
                actor_loss = self._impl.update_actor(batch.observations)
                temp_loss, temp = self._impl.update_temp(batch.observations)
                alpha_loss, alpha = self._impl.update_alpha(batch.observations)
                self._impl.update_actor_target()
                self._impl.update_critic_target()
            else:
                actor_loss = None
                temp_loss = None
                temp = None
                alpha_loss = None
                alpha = None
        else:
            critic_loss = None
            actor_loss = None
            temp_loss = None
            temp = None
            alpha_loss = None
            alpha = None
        return [
            critic_loss,
            actor_loss,
            imitator_loss,
            temp_loss,
            temp,
            alpha_loss,
            alpha,
        ]

    def _get_loss_labels(self) -> List[str]:
        return [
            "critic_loss",
            "actor_loss",
            "imitator_loss",
            "temp_loss",
            "temp",
            "alpha_loss",
            "alpha",
        ]
