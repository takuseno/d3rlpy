import dataclasses

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_continuous_q_function,
    create_deterministic_policy,
    create_deterministic_residual_policy,
    create_vae_decoder,
    create_vae_encoder,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.plas_impl import (
    PLASImpl,
    PLASModules,
    PLASWithPerturbationImpl,
    PLASWithPerturbationModules,
)

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
        targ_policy = create_deterministic_policy(
            observation_shape,
            2 * action_size,
            self._config.actor_encoder_factory,
            device=self._device,
        )
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        vae_encoder = create_vae_encoder(
            observation_shape=observation_shape,
            action_size=action_size,
            latent_size=2 * action_size,
            min_logstd=-4.0,
            max_logstd=15.0,
            encoder_factory=self._config.imitator_encoder_factory,
            device=self._device,
        )
        vae_decoder = create_vae_decoder(
            observation_shape=observation_shape,
            action_size=action_size,
            latent_size=2 * action_size,
            encoder_factory=self._config.imitator_encoder_factory,
            device=self._device,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(), lr=self._config.actor_learning_rate
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(), lr=self._config.critic_learning_rate
        )
        vae_optim = self._config.critic_optim_factory.create(
            list(vae_encoder.named_modules())
            + list(vae_decoder.named_modules()),
            lr=self._config.imitator_learning_rate,
        )

        modules = PLASModules(
            policy=policy,
            targ_policy=targ_policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            vae_optim=vae_optim,
        )

        self._impl = PLASImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            lam=self._config.lam,
            beta=self._config.beta,
            warmup_steps=self._config.warmup_steps,
            device=self._device,
        )

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
        targ_policy = create_deterministic_policy(
            observation_shape,
            2 * action_size,
            self._config.actor_encoder_factory,
            device=self._device,
        )
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        vae_encoder = create_vae_encoder(
            observation_shape=observation_shape,
            action_size=action_size,
            latent_size=2 * action_size,
            min_logstd=-4.0,
            max_logstd=15.0,
            encoder_factory=self._config.imitator_encoder_factory,
            device=self._device,
        )
        vae_decoder = create_vae_decoder(
            observation_shape=observation_shape,
            action_size=action_size,
            latent_size=2 * action_size,
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
        targ_perturbation = create_deterministic_residual_policy(
            observation_shape=observation_shape,
            action_size=action_size,
            scale=self._config.action_flexibility,
            encoder_factory=self._config.actor_encoder_factory,
            device=self._device,
        )

        named_modules = list(policy.named_modules())
        named_modules += list(perturbation.named_modules())
        actor_optim = self._config.actor_optim_factory.create(
            named_modules, lr=self._config.actor_learning_rate
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(), lr=self._config.critic_learning_rate
        )
        vae_optim = self._config.critic_optim_factory.create(
            list(vae_encoder.named_modules())
            + list(vae_decoder.named_modules()),
            lr=self._config.imitator_learning_rate,
        )

        modules = PLASWithPerturbationModules(
            policy=policy,
            targ_policy=targ_policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            perturbation=perturbation,
            targ_perturbation=targ_perturbation,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            vae_optim=vae_optim,
        )

        self._impl = PLASWithPerturbationImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            lam=self._config.lam,
            beta=self._config.beta,
            warmup_steps=self._config.warmup_steps,
            device=self._device,
        )


register_learnable(PLASConfig)
register_learnable(PLASWithPerturbationConfig)
