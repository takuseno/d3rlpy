import dataclasses

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_continuous_q_function,
    create_normal_policy,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...optimizers.optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.crr_impl import CRRImpl, CRRModules

__all__ = ["CRRConfig", "CRR"]


@dataclasses.dataclass()
class CRRConfig(LearnableConfig):
    r"""Config of Critic Reguralized Regression algorithm.

    CRR is a simple offline RL method similar to AWAC.

    The policy is trained as a supervised regression.

    .. math::

        J(\phi) = \mathbb{E}_{s_t, a_t \sim D}
            [\log \pi_\phi(a_t|s_t) f(Q_\theta, \pi_\phi, s_t, a_t)]

    where :math:`f` is a filter function which has several options. The first
    option is ``binary`` function.

    .. math::

        f := \mathbb{1} [A_\theta(s, a) > 0]

    The other is ``exp`` function.

    .. math::

        f := \exp(A(s, a) / \beta)

    The :math:`A(s, a)` is an average function which also has several options.
    The first option is ``mean``.

    .. math::

        A(s, a) = Q_\theta (s, a) - \frac{1}{m} \sum^m_j Q(s, a_j)

    The other one is ``max``.

    .. math::

        A(s, a) = Q_\theta (s, a) - \max^m_j Q(s, a_j)

    where :math:`a_j \sim \pi_\phi(s)`.

    In evaluation, the action is determined by Critic Weighted Policy (CWP).
    In CWP, the several actions are sampled from the policy function, and the
    final action is re-sampled from the estimated action-value distribution.

    References:
        * `Wang et al., Critic Reguralized Regression.
          <https://arxiv.org/abs/2006.15134>`_

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
        beta (float): Temperature value defined as :math:`\beta` above.
        n_action_samples (int): Number of sampled actions to calculate
            :math:`A(s, a)` and for CWP.
        advantage_type (str): Advantage function type. The available options
            are ``['mean', 'max']``.
        weight_type (str): Filter function type. The available options
            are ``['binary', 'exp']``.
        max_weight (float): Maximum weight for cross-entropy loss.
        n_critics (int): Number of Q functions for ensemble.
        target_update_type (str): Target update type. The available options are
            ``['hard', 'soft']``.
        tau (float): Target network synchronization coefficiency used with
            ``soft`` target update.
        update_actor_interval (int): Interval to update policy function used
            with ``hard`` target update.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    batch_size: int = 100
    gamma: float = 0.99
    beta: float = 1.0
    n_action_samples: int = 4
    advantage_type: str = "mean"
    weight_type: str = "exp"
    max_weight: float = 20.0
    n_critics: int = 1
    target_update_type: str = "hard"
    tau: float = 5e-3
    target_update_interval: int = 100
    update_actor_interval: int = 1
    compile_graph: bool = False

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "CRR":
        return CRR(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "crr"


class CRR(QLearningAlgoBase[CRRImpl, CRRConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        compiled = self._config.compile_graph and "cuda" in self._device

        policy = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_policy = create_normal_policy(
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
            compiled=compiled,
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(),
            lr=self._config.critic_learning_rate,
            compiled=compiled,
        )

        modules = CRRModules(
            policy=policy,
            targ_policy=targ_policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
        )

        self._impl = CRRImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            beta=self._config.beta,
            n_action_samples=self._config.n_action_samples,
            advantage_type=self._config.advantage_type,
            weight_type=self._config.weight_type,
            max_weight=self._config.max_weight,
            tau=self._config.tau,
            target_update_type=self._config.target_update_type,
            target_update_interval=self._config.target_update_interval,
            compile_graph=compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(CRRConfig)
