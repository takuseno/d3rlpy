import dataclasses
from typing import Dict, Optional

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ...dataset import Shape
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...torch_utility import TorchMiniBatch
from .base import QLearningAlgoBase
from .torch.awac_impl import AWACImpl

__all__ = ["AWACConfig", "AWAC"]


@dataclasses.dataclass(frozen=True)
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
        tau (float): target network synchronization coefficiency.
        lam (float): :math:`\lambda` for weight calculation.
        n_action_samples (int): the number of sampled actions to calculate
            :math:`A^\pi(s_t, a_t)`.
        n_critics (int): the number of Q functions for ensemble.
        update_actor_interval (int): interval to update policy function.
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
    batch_size: int = 1024
    gamma: float = 0.99
    tau: float = 0.005
    lam: float = 1.0
    n_action_samples: int = 1
    n_critics: int = 2
    update_actor_interval: int = 1

    def create(self, device: DeviceArg = False) -> "AWAC":
        return AWAC(self, device)

    @staticmethod
    def get_type() -> str:
        return "awac"


class AWAC(QLearningAlgoBase):
    _config: AWACConfig
    _impl: Optional[AWACImpl]

    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        self._impl = AWACImpl(
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
            tau=self._config.tau,
            lam=self._config.lam,
            n_action_samples=self._config.n_action_samples,
            n_critics=self._config.n_critics,
            device=self._device,
        )
        self._impl.build()

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        critic_loss = self._impl.update_critic(batch)
        metrics.update({"critic_loss": critic_loss})

        # delayed policy update
        if self._grad_step % self._config.update_actor_interval == 0:
            actor_loss = self._impl.update_actor(batch)
            metrics.update({"actor_loss": actor_loss})
            self._impl.update_critic_target()
            self._impl.update_actor_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(AWACConfig)
