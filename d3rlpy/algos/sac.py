from typing import Any, List, Optional, Sequence
from .base import AlgoBase, DataGenerator
from .torch.sac_impl import SACImpl, DiscreteSACImpl
from ..augmentation import AugmentationPipeline
from ..dataset import TransitionMiniBatch
from ..models.encoders import EncoderFactory
from ..models.q_functions import QFunctionFactory
from ..models.optimizers import OptimizerFactory, AdamFactory
from ..gpu import Device
from ..argument_utility import check_encoder, EncoderArg
from ..argument_utility import check_use_gpu, UseGPUArg
from ..argument_utility import check_augmentation, AugmentationArg
from ..argument_utility import check_q_func, QFuncArg
from ..argument_utility import ScalerArg, ActionScalerArg
from ..constants import IMPL_NOT_INITIALIZED_ERROR


class SAC(AlgoBase):
    r"""Soft Actor-Critic algorithm.

    SAC is a DDPG-based maximum entropy RL algorithm, which produces
    state-of-the-art performance in online RL settings.
    SAC leverages twin Q functions proposed in TD3. Additionally,
    `delayed policy update` in TD3 is also implemented, which is not done in
    the paper.

    .. math::

        L(\theta_i) = \mathbb{E}_{s_t,\, a_t,\, r_{t+1},\, s_{t+1} \sim D,\,
                                   a_{t+1} \sim \pi_\phi(\cdot|s_{t+1})} \Big[
            \big(y - Q_{\theta_i}(s_t, a_t)\big)^2\Big]

    .. math::

        y = r_{t+1} + \gamma \Big(\min_j Q_{\theta_j}(s_{t+1}, a_{t+1})
            - \alpha \log \big(\pi_\phi(a_{t+1}|s_{t+1})\big)\Big)

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D,\, a_t \sim \pi_\phi(\cdot|s_t)}
            \Big[\alpha \log (\pi_\phi (a_t|s_t))
              - \min_i Q_{\theta_i}\big(s_t, \pi_\phi(a_t|s_t)\big)\Big]

    The temperature parameter :math:`\alpha` is also automatically adjustable.

    .. math::

        J(\alpha) = \mathbb{E}_{s_t \sim D,\, a_t \sim \pi_\phi(\cdot|s_t)}
            \bigg[-\alpha \Big(\log \big(\pi_\phi(a_t|s_t)\big) + H\Big)\bigg]

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
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
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
        n_critics (int): the number of Q functions for ensemble.
        bootstrap (bool): flag to bootstrap Q functions.
        share_encoder (bool): flag to share encoder network.
        target_reduction_type (str): ensemble reduction method at target value
            estimation. The available options are
            ``['min', 'max', 'mean', 'mix', 'none']``.
        update_actor_interval (int): interval to update policy function.
        initial_temperature (float): initial temperature value.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
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
    _temp_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _temp_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _bootstrap: bool
    _share_encoder: bool
    _target_reduction_type: str
    _update_actor_interval: int
    _initial_temperature: float
    _augmentation: AugmentationPipeline
    _use_gpu: Optional[Device]
    _impl: Optional[SACImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        temp_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        temp_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        bootstrap: bool = False,
        share_encoder: bool = False,
        target_reduction_type: str = "min",
        update_actor_interval: int = 1,
        initial_temperature: float = 1.0,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        augmentation: AugmentationArg = None,
        generator: Optional[DataGenerator] = None,
        impl: Optional[SACImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            generator=generator,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._temp_learning_rate = temp_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._temp_optim_factory = temp_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._bootstrap = bootstrap
        self._share_encoder = share_encoder
        self._target_reduction_type = target_reduction_type
        self._update_actor_interval = update_actor_interval
        self._initial_temperature = initial_temperature
        self._augmentation = check_augmentation(augmentation)
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = SACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            temp_learning_rate=self._temp_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            bootstrap=self._bootstrap,
            share_encoder=self._share_encoder,
            target_reduction_type=self._target_reduction_type,
            initial_temperature=self._initial_temperature,
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
        )

        # delayed policy update
        if total_step % self._update_actor_interval == 0:
            actor_loss = self._impl.update_actor(batch.observations)

            # lagrangian parameter update for SAC temperature
            if self._temp_learning_rate > 0:
                temp_loss, temp = self._impl.update_temp(batch.observations)
            else:
                temp_loss, temp = None, None

            self._impl.update_critic_target()
            self._impl.update_actor_target()
        else:
            actor_loss = None
            temp_loss = None
            temp = None

        return [critic_loss, actor_loss, temp_loss, temp]

    def get_loss_labels(self) -> List[str]:
        return ["critic_loss", "actor_loss", "temp_loss", "temp"]


class DiscreteSAC(AlgoBase):
    r"""Soft Actor-Critic algorithm for discrete action-space.

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
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
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
        generator (d3rlpy.algos.base.DataGenerator): dynamic dataset generator
            (e.g. model-based RL).
        impl (d3rlpy.algos.torch.sac_impl.DiscreteSACImpl):
            algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _temp_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _temp_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _n_critics: int
    _bootstrap: bool
    _share_encoder: bool
    _initial_temperature: float
    _target_update_interval: int
    _augmentation: AugmentationPipeline
    _use_gpu: Optional[Device]
    _impl: Optional[DiscreteSACImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        temp_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(eps=1e-4),
        critic_optim_factory: OptimizerFactory = AdamFactory(eps=1e-4),
        temp_optim_factory: OptimizerFactory = AdamFactory(eps=1e-4),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 64,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_critics: int = 2,
        bootstrap: bool = False,
        share_encoder: bool = False,
        initial_temperature: float = 1.0,
        target_update_interval: int = 8000,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        augmentation: AugmentationArg = None,
        generator: Optional[DataGenerator] = None,
        impl: Optional[DiscreteSACImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=None,
            generator=generator,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._temp_learning_rate = temp_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._temp_optim_factory = temp_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._n_critics = n_critics
        self._bootstrap = bootstrap
        self._share_encoder = share_encoder
        self._initial_temperature = initial_temperature
        self._target_update_interval = target_update_interval
        self._augmentation = check_augmentation(augmentation)
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = DiscreteSACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            temp_learning_rate=self._temp_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            n_critics=self._n_critics,
            bootstrap=self._bootstrap,
            share_encoder=self._share_encoder,
            initial_temperature=self._initial_temperature,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
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
        )

        actor_loss = self._impl.update_actor(batch.observations)

        # lagrangian parameter update for SAC temeprature
        if self._temp_learning_rate > 0:
            temp_loss, temp = self._impl.update_temp(batch.observations)
        else:
            temp_loss, temp = None, None

        if total_step % self._target_update_interval == 0:
            self._impl.update_target()

        return [critic_loss, actor_loss, temp_loss, temp]

    def get_loss_labels(self) -> List[str]:
        return ["critic_loss", "actor_loss", "temp_loss", "temp"]
