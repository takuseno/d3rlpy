import dataclasses
from typing import Dict, Optional

from ..base import DeviceArg, LearnableConfig, register_learnable
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import Shape
from ..models.encoders import EncoderFactory, make_encoder_field
from ..models.optimizers import OptimizerFactory, make_optimizer_field
from ..models.q_functions import QFunctionFactory, make_q_func_field
from ..torch_utility import TorchMiniBatch
from .base import AlgoBase
from .torch.crr_impl import CRRImpl

__all__ = ["CRRConfig", "CRR"]


@dataclasses.dataclass(frozen=True)
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
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        beta (float): temperature value defined as :math:`\beta` above.
        n_action_samples (int): the number of sampled actions to calculate
            :math:`A(s, a)` and for CWP.
        advantage_type (str): advantage function type. The available options
            are ``['mean', 'max']``.
        weight_type (str): filter function type. The available options
            are ``['binary', 'exp']``.
        max_weight (float): maximum weight for cross-entropy loss.
        n_critics (int): the number of Q functions for ensemble.
        target_update_type (str): target update type. The available options are
            ``['hard', 'soft']``.
        tau (float): target network synchronization coefficiency used with
            ``soft`` target update.
        update_actor_interval (int): interval to update policy function used
            with ``hard`` target update.
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): reward preprocessor.
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

    def create(self, device: DeviceArg = False) -> "CRR":
        return CRR(self, device)

    @staticmethod
    def get_type() -> str:
        return "crr"


class CRR(AlgoBase):
    _config: CRRConfig
    _impl: Optional[CRRImpl]

    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        self._impl = CRRImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._config.actor_learning_rate,
            critic_learning_rate=self._config.critic_learning_rate,
            actor_optim_factory=self._config.actor_optim_factory,
            critic_optim_factory=self._config.critic_optim_factory,
            actor_encoder_factory=self._config.actor_encoder_factory,
            critic_encoder_factory=self._config.critic_encoder_factory,
            q_func_factory=self._config.q_func_factory,
            gamma=self._config.gamma,
            beta=self._config.beta,
            n_action_samples=self._config.n_action_samples,
            advantage_type=self._config.advantage_type,
            weight_type=self._config.weight_type,
            max_weight=self._config.max_weight,
            n_critics=self._config.n_critics,
            tau=self._config.tau,
            device=self._device,
        )
        self._impl.build()

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        critic_loss = self._impl.update_critic(batch)
        actor_loss = self._impl.update_actor(batch)

        if self._config.target_update_type == "hard":
            if self._grad_step % self._config.target_update_interval == 0:
                self._impl.sync_critic_target()
                self._impl.sync_actor_target()
        elif self._config.target_update_type == "soft":
            self._impl.update_critic_target()
            self._impl.update_actor_target()
        else:
            raise ValueError(
                f"invalid target_update_type: {self._config.target_update_type}"
            )

        return {"critic_loss": critic_loss, "actor_loss": actor_loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(CRRConfig)
