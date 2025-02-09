import dataclasses

import torch

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_continuous_q_function,
    create_normal_policy,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...models.torch import Parameter
from ...optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
from .base import QLearningAlgoBase
from .functional import FunctionalQLearningAlgoImplBase
from .functional_utils import (
    DeterministicContinuousActionSampler,
    GaussianContinuousActionSampler,
)
from .torch.awac_impl import AWACActorLossFn
from .torch.ddpg_impl import DDPGValuePredictor
from .torch.sac_impl import SACCriticLossFn, SACModules, SACUpdater

__all__ = ["AWACConfig", "AWAC"]


@dataclasses.dataclass()
class AWACConfig(LearnableConfig):
    r"""Config of Advantage Weighted Actor-Critic algorithm.

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
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        lam (float): :math:`\lambda` for weight calculation.
        n_action_samples (int): Number of sampled actions to calculate
            :math:`A^\pi(s_t, a_t)`.
        n_critics (int): Number of Q functions for ensemble.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 1024
    gamma: float = 0.99
    tau: float = 0.005
    lam: float = 1.0
    n_action_samples: int = 1
    n_critics: int = 2

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "AWAC":
        return AWAC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "awac"


class AWAC(QLearningAlgoBase[FunctionalQLearningAlgoImplBase, AWACConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            min_logstd=-6.0,
            max_logstd=0.0,
            use_std_parameter=True,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(),
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )

        dummy_log_temp = Parameter(torch.zeros(1, 1))
        dummy_log_temp.to(self._device)
        modules = SACModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            log_temp=dummy_log_temp,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=None,
        )

        updater = SACUpdater(
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            critic_optim=critic_optim,
            actor_optim=actor_optim,
            critic_loss_fn=SACCriticLossFn(
                q_func_forwarder=q_func_forwarder,
                targ_q_func_forwarder=targ_q_func_forwarder,
                policy=policy,
                log_temp=dummy_log_temp,
                gamma=self._config.gamma,
            ),
            actor_loss_fn=AWACActorLossFn(
                q_func_forwarder=q_func_forwarder,
                policy=policy,
                n_action_samples=self._config.n_action_samples,
                lam=self._config.lam,
                action_size=action_size,
            ),
            tau=self._config.tau,
            compiled=self.compiled,
        )
        exploit_action_sampler = DeterministicContinuousActionSampler(policy)
        explore_action_sampler = GaussianContinuousActionSampler(policy)
        value_predictor = DDPGValuePredictor(q_func_forwarder)

        self._impl = FunctionalQLearningAlgoImplBase(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            updater=updater,
            exploit_action_sampler=exploit_action_sampler,
            explore_action_sampler=explore_action_sampler,
            value_predictor=value_predictor,
            q_function=q_funcs,
            q_function_optim=critic_optim.optim,
            policy=policy,
            policy_optim=actor_optim.optim,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(AWACConfig)
