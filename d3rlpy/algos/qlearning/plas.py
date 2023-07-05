import dataclasses
from typing import Dict

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ...dataset import Shape
from ...models.builders import (
    create_conditional_vae,
    create_continuous_q_function,
    create_deterministic_policy,
    create_deterministic_residual_policy,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...torch_utility import TorchMiniBatch
from .base import QLearningAlgoBase
from .torch.plas_impl import PLASImpl, PLASWithPerturbationImpl

__all__ = [
    "PLASConfig",
    "PLAS",
    "PLASWithPerturbationConfig",
    "PLASWithPerturbation",
]


@dataclasses.dataclass()
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
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        imitator_learning_rate (float): Learning rate for Conditional VAE.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        imitator_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the conditional VAE.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        imitator_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the conditional VAE.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        n_critics (int): Number of Q functions for ensemble.
        update_actor_interval (int): Interval to update policy function.
        lam (float): Weight factor for critic ensemble.
        warmup_steps (int): Number of steps to warmup the VAE.
        beta (float): KL reguralization term for Conditional VAE.
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

    def create(self, device: DeviceArg = False) -> "PLAS":
        return PLAS(self, device)

    @staticmethod
    def get_type() -> str:
        return "plas"


class PLAS(QLearningAlgoBase[PLASImpl, PLASConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_deterministic_policy(
            observation_shape,
            2 * action_size,
            self._config.actor_encoder_factory,
            device=self._device,
        )
        q_func = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        imitator = create_conditional_vae(
            observation_shape=observation_shape,
            action_size=action_size,
            latent_size=2 * action_size,
            beta=self._config.beta,
            min_logstd=-4.0,
            max_logstd=15.0,
            encoder_factory=self._config.imitator_encoder_factory,
            device=self._device,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.parameters(), lr=self._config.actor_learning_rate
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_func.parameters(), lr=self._config.critic_learning_rate
        )
        imitator_optim = self._config.critic_optim_factory.create(
            imitator.parameters(), lr=self._config.imitator_learning_rate
        )

        self._impl = PLASImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            policy=policy,
            q_func=q_func,
            imitator=imitator,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            imitator_optim=imitator_optim,
            gamma=self._config.gamma,
            tau=self._config.tau,
            lam=self._config.lam,
            device=self._device,
        )

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
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


@dataclasses.dataclass()
class PLASWithPerturbationConfig(PLASConfig):
    r"""Config of Policy in Latent Action Space algorithm with perturbation
    layer.

    PLAS with perturbation layer enables PLAS to output out-of-distribution
    action.

    References:
        * `Zhou et al., PLAS: latent action space for offline reinforcement
          learning. <https://arxiv.org/abs/2011.07213>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        imitator_learning_rate (float): Learning rate for Conditional VAE.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        imitator_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the conditional VAE.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        imitator_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the conditional VAE.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        n_critics (int): Number of Q functions for ensemble.
        update_actor_interval (int): Interval to update policy function.
        lam (float): Weight factor for critic ensemble.
        action_flexibility (float): Output scale of perturbation layer.
        warmup_steps (int): Number of steps to warmup the VAE.
        beta (float): KL reguralization term for Conditional VAE.
    """
    action_flexibility: float = 0.05

    def create(self, device: DeviceArg = False) -> "PLASWithPerturbation":
        return PLASWithPerturbation(self, device)

    @staticmethod
    def get_type() -> str:
        return "plas_with_perturbation"


class PLASWithPerturbation(PLAS):
    _config: PLASWithPerturbationConfig

    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_deterministic_policy(
            observation_shape,
            2 * action_size,
            self._config.actor_encoder_factory,
            device=self._device,
        )
        q_func = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        imitator = create_conditional_vae(
            observation_shape=observation_shape,
            action_size=action_size,
            latent_size=2 * action_size,
            beta=self._config.beta,
            min_logstd=-4.0,
            max_logstd=15.0,
            encoder_factory=self._config.imitator_encoder_factory,
            device=self._device,
        )
        perturbation = create_deterministic_residual_policy(
            observation_shape=observation_shape,
            action_size=action_size,
            scale=self._config.action_flexibility,
            encoder_factory=self._config.actor_encoder_factory,
            device=self._device,
        )

        parameters = list(policy.parameters())
        parameters += list(perturbation.parameters())
        actor_optim = self._config.actor_optim_factory.create(
            params=parameters, lr=self._config.actor_learning_rate
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_func.parameters(), lr=self._config.critic_learning_rate
        )
        imitator_optim = self._config.critic_optim_factory.create(
            imitator.parameters(), lr=self._config.imitator_learning_rate
        )

        self._impl = PLASWithPerturbationImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            policy=policy,
            q_func=q_func,
            imitator=imitator,
            perturbation=perturbation,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            imitator_optim=imitator_optim,
            gamma=self._config.gamma,
            tau=self._config.tau,
            lam=self._config.lam,
            device=self._device,
        )


register_learnable(PLASConfig)
register_learnable(PLASWithPerturbationConfig)
