import dataclasses
from typing import Dict, Optional

from ..argument_utility import UseGPUArg
from ..base import LearnableConfig, register_learnable
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import Shape, TransitionMiniBatch
from ..models.encoders import EncoderFactory, make_encoder_field
from ..models.optimizers import OptimizerFactory, make_optimizer_field
from ..models.q_functions import QFunctionFactory, make_q_func_field
from .base import AlgoBase
from .torch.plas_impl import PLASImpl, PLASWithPerturbationImpl

__all__ = [
    "PLASConfig",
    "PLAS",
    "PLASWithPerturbationConfig",
    "PLASWithPerturbation",
]


@dataclasses.dataclass(frozen=True)
class PLASConfig(LearnableConfig):
    r"""Config of Policy in Latent Action Space algorithm.

    PLAS is an offline deep reinforcement learning algorithm whose policy
    function is trained in latent space of Conditional VAE.
    Unlike other algorithms, PLAS can achieve good performance by using
    its less constrained policy function.

    .. math::

       a \sim p_\beta (a|s, z=\pi_\phi(s))

    where :math:`\beta` is a parameter of the decoder in Conditional VAE.

    References:
        * `Zhou et al., PLAS: latent action space for offline reinforcement
          learning. <https://arxiv.org/abs/2011.07213>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        imitator_learning_rate (float): learning rate for Conditional VAE.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        imitator_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the conditional VAE.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the critic.
        imitator_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the conditional VAE.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        update_actor_interval (int): interval to update policy function.
        lam (float): weight factor for critic ensemble.
        warmup_steps (int): the number of steps to warmup the VAE.
        beta (float): KL reguralization term for Conditional VAE.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): reward preprocessor.
    """
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    imitator_learning_rate: float = 1e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    imitator_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    imitator_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 100
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    update_actor_interval: int = 1
    lam: float = 0.75
    warmup_steps: int = 500000
    beta: float = 0.5

    def create(self, use_gpu: UseGPUArg = False) -> "PLAS":
        return PLAS(self, use_gpu)

    @staticmethod
    def get_type() -> str:
        return "plas"


class PLAS(AlgoBase):
    _config: PLASConfig
    _impl: Optional[PLASImpl]

    def _create_impl(self, observation_shape: Shape, action_size: int) -> None:
        self._impl = PLASImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._config.actor_learning_rate,
            critic_learning_rate=self._config.critic_learning_rate,
            imitator_learning_rate=self._config.imitator_learning_rate,
            actor_optim_factory=self._config.actor_optim_factory,
            critic_optim_factory=self._config.critic_optim_factory,
            imitator_optim_factory=self._config.imitator_optim_factory,
            actor_encoder_factory=self._config.actor_encoder_factory,
            critic_encoder_factory=self._config.critic_encoder_factory,
            imitator_encoder_factory=self._config.imitator_encoder_factory,
            q_func_factory=self._config.q_func_factory,
            gamma=self._config.gamma,
            tau=self._config.tau,
            n_critics=self._config.n_critics,
            lam=self._config.lam,
            beta=self._config.beta,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
            use_gpu=self._use_gpu,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        if self._grad_step < self._config.warmup_steps:
            imitator_loss = self._impl.update_imitator(batch)
            metrics.update({"imitator_loss": imitator_loss})
        else:
            critic_loss = self._impl.update_critic(batch)
            metrics.update({"critic_loss": critic_loss})
            if self._grad_step % self._config.update_actor_interval == 0:
                actor_loss = self._impl.update_actor(batch)
                metrics.update({"actor_loss": actor_loss})
                self._impl.update_actor_target()
                self._impl.update_critic_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


@dataclasses.dataclass(frozen=True)
class PLASWithPerturbationConfig(PLASConfig):
    r"""Config of Policy in Latent Action Space algorithm with perturbation layer.

    PLAS with perturbation layer enables PLAS to output out-of-distribution
    action.

    References:
        * `Zhou et al., PLAS: latent action space for offline reinforcement
          learning. <https://arxiv.org/abs/2011.07213>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        imitator_learning_rate (float): learning rate for Conditional VAE.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        imitator_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the conditional VAE.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the critic.
        imitator_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the conditional VAE.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        update_actor_interval (int): interval to update policy function.
        lam (float): weight factor for critic ensemble.
        action_flexibility (float): output scale of perturbation layer.
        warmup_steps (int): the number of steps to warmup the VAE.
        beta (float): KL reguralization term for Conditional VAE.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): reward preprocessor.
    """
    action_flexibility: float = 0.05

    def create(self, use_gpu: UseGPUArg = False) -> "PLASWithPerturbation":
        return PLASWithPerturbation(self, use_gpu)

    @staticmethod
    def get_type() -> str:
        return "plas_with_perturbation"


class PLASWithPerturbation(PLAS):
    _config: PLASWithPerturbationConfig
    _impl: Optional[PLASWithPerturbationImpl]

    def _create_impl(self, observation_shape: Shape, action_size: int) -> None:
        self._impl = PLASWithPerturbationImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._config.actor_learning_rate,
            critic_learning_rate=self._config.critic_learning_rate,
            imitator_learning_rate=self._config.imitator_learning_rate,
            actor_optim_factory=self._config.actor_optim_factory,
            critic_optim_factory=self._config.critic_optim_factory,
            imitator_optim_factory=self._config.imitator_optim_factory,
            actor_encoder_factory=self._config.actor_encoder_factory,
            critic_encoder_factory=self._config.critic_encoder_factory,
            imitator_encoder_factory=self._config.imitator_encoder_factory,
            q_func_factory=self._config.q_func_factory,
            gamma=self._config.gamma,
            tau=self._config.tau,
            n_critics=self._config.n_critics,
            lam=self._config.lam,
            beta=self._config.beta,
            action_flexibility=self._config.action_flexibility,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
            use_gpu=self._use_gpu,
        )
        self._impl.build()


register_learnable(PLASConfig)
register_learnable(PLASWithPerturbationConfig)
