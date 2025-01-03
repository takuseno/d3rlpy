import dataclasses

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_continuous_q_function,
    create_deterministic_policy,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...optimizers.optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.ddpg_impl import DDPGValuePredictor, DDPGCriticLossFn, DDPGActorLossFn, DDPGUpdater, DDPGModules
from .functional import FunctionalQLearningAlgoImplBase
from .functional_utils import DeterministicContinuousActionSampler

__all__ = ["DDPGConfig", "DDPG"]


@dataclasses.dataclass()
class DDPGConfig(LearnableConfig):
    r"""Config of Deep Deterministic Policy Gradients algorithm.

    DDPG is an actor-critic algorithm that trains a Q function parametrized
    with :math:`\theta` and a policy function parametrized with :math:`\phi`.

    .. math::

        L(\theta) = \mathbb{E}_{s_t,\, a_t,\, r_{t+1},\, s_{t+1} \sim D} \Big[(r_{t+1}
            + \gamma Q_{\theta'}\big(s_{t+1}, \pi_{\phi'}(s_{t+1}))
            - Q_\theta(s_t, a_t)\big)^2\Big]

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D} \Big[Q_\theta\big(s_t, \pi_\phi(s_t)\big)\Big]

    where :math:`\theta'` and :math:`\phi` are the target network parameters.
    There target network parameters are updated every iteration.

    .. math::

        \theta' \gets \tau \theta + (1 - \tau) \theta'

        \phi' \gets \tau \phi + (1 - \tau) \phi'

    References:
        * `Silver et al., Deterministic policy gradient algorithms.
          <http://proceedings.mlr.press/v32/silver14.html>`_
        * `Lillicrap et al., Continuous control with deep reinforcement
          learning. <https://arxiv.org/abs/1509.02971>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q function.
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
        n_critics (int): Number of Q functions for ensemble.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    batch_size: int = 256
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    tau: float = 0.005
    n_critics: int = 1

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DDPG":
        return DDPG(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "ddpg"


class DDPG(QLearningAlgoBase[FunctionalQLearningAlgoImplBase, DDPGConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_deterministic_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_policy = create_deterministic_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
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

        modules = DDPGModules(
            policy=policy,
            targ_policy=targ_policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
        )

        updater = DDPGUpdater(
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            policy=policy,
            targ_policy=targ_policy,
            critic_optim=critic_optim,
            actor_optim=actor_optim,
            critic_loss_fn=DDPGCriticLossFn(
                q_func_forwarder=q_func_forwarder,
                targ_q_func_forwarder=targ_q_func_forwarder,
                targ_policy=targ_policy,
                gamma=self._config.gamma,
            ),
            actor_loss_fn=DDPGActorLossFn(
                q_func_forwarder=q_func_forwarder,
                policy=policy,
            ),
            tau=self._config.tau,
            compiled=self.compiled,
        )
        action_sampler = DeterministicContinuousActionSampler(policy)
        value_predictor = DDPGValuePredictor(q_func_forwarder)

        self._impl = FunctionalQLearningAlgoImplBase(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            updater=updater,
            exploit_action_sampler=action_sampler,
            explore_action_sampler=action_sampler,
            value_predictor=value_predictor,
            q_function=q_funcs,
            q_function_optim=critic_optim.optim,
            policy=policy,
            policy_optim=actor_optim.optim,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(DDPGConfig)
