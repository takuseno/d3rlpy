import dataclasses
import math

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_continuous_q_function,
    create_normal_policy,
    create_parameter,
    create_vae_decoder,
    create_vae_encoder,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.bear_impl import BEARImpl, BEARModules

__all__ = ["BEARConfig", "BEAR"]


@dataclasses.dataclass()
class BEARConfig(LearnableConfig):
    r"""Config of Bootstrapping Error Accumulation Reduction algorithm.

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
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        imitator_learning_rate (float): Learning rate for behavior policy
            function.
        temp_learning_rate (float): Learning rate for temperature parameter.
        alpha_learning_rate (float): Learning rate for :math:`\alpha`.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        imitator_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the behavior policy.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for the temperature.
        alpha_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory for :math:`\alpha`.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        imitator_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the behavior policy.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        n_critics (int): Number of Q functions for ensemble.
        initial_temperature (float): Initial temperature value.
        initial_alpha (float): Initial :math:`\alpha` value.
        alpha_threshold (float): Threshold value described as
            :math:`\epsilon`.
        lam (float): Weight for critic ensemble.
        n_action_samples (int): Number of action samples to compute the
            best action.
        n_target_samples (int): Number of action samples to compute
            BCQ-like target value.
        n_mmd_action_samples (int): Number of action samples to compute MMD.
        mmd_kernel (str): MMD kernel function. The available options are
            ``['gaussian', 'laplacian']``.
        mmd_sigma (float): :math:`\sigma` for gaussian kernel in MMD
            calculation.
        vae_kl_weight (float): Constant weight to scale KL term for behavior
            policy training.
        warmup_steps (int): Number of steps to warmup the policy
            function.
    """

    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    imitator_learning_rate: float = 3e-4
    temp_learning_rate: float = 1e-4
    alpha_learning_rate: float = 1e-3
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    imitator_optim_factory: OptimizerFactory = make_optimizer_field()
    temp_optim_factory: OptimizerFactory = make_optimizer_field()
    alpha_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    imitator_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    initial_temperature: float = 1.0
    initial_alpha: float = 1.0
    alpha_threshold: float = 0.05
    lam: float = 0.75
    n_action_samples: int = 100
    n_target_samples: int = 10
    n_mmd_action_samples: int = 4
    mmd_kernel: str = "laplacian"
    mmd_sigma: float = 20.0
    vae_kl_weight: float = 0.5
    warmup_steps: int = 40000

    def create(self, device: DeviceArg = False) -> "BEAR":
        return BEAR(self, device)

    @staticmethod
    def get_type() -> str:
        return "bear"


class BEAR(QLearningAlgoBase[BEARImpl, BEARConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_normal_policy(
            observation_shape,
            action_size,
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
        log_temp = create_parameter(
            (1, 1),
            math.log(self._config.initial_temperature),
            device=self._device,
        )
        log_alpha = create_parameter(
            (1, 1), math.log(self._config.initial_alpha), device=self._device
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(), lr=self._config.actor_learning_rate
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(), lr=self._config.critic_learning_rate
        )
        vae_optim = self._config.imitator_optim_factory.create(
            list(vae_encoder.named_modules())
            + list(vae_decoder.named_modules()),
            lr=self._config.imitator_learning_rate,
        )
        temp_optim = self._config.temp_optim_factory.create(
            log_temp.named_modules(), lr=self._config.temp_learning_rate
        )
        alpha_optim = self._config.alpha_optim_factory.create(
            log_alpha.named_modules(), lr=self._config.actor_learning_rate
        )

        modules = BEARModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            log_temp=log_temp,
            log_alpha=log_alpha,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            vae_optim=vae_optim,
            temp_optim=temp_optim,
            alpha_optim=alpha_optim,
        )

        self._impl = BEARImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            alpha_threshold=self._config.alpha_threshold,
            lam=self._config.lam,
            n_action_samples=self._config.n_action_samples,
            n_target_samples=self._config.n_target_samples,
            n_mmd_action_samples=self._config.n_mmd_action_samples,
            mmd_kernel=self._config.mmd_kernel,
            mmd_sigma=self._config.mmd_sigma,
            vae_kl_weight=self._config.vae_kl_weight,
            warmup_steps=self._config.warmup_steps,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(BEARConfig)
