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
from .torch.td3_impl import TD3Impl

__all__ = ["TD3Config", "TD3"]


@dataclasses.dataclass()
class TD3Config(LearnableConfig):
    r"""Config of Twin Delayed Deep Deterministic Policy Gradients algorithm.

    TD3 is an improved DDPG-based algorithm.
    Major differences from DDPG are as follows.

    * TD3 has twin Q functions to reduce overestimation bias at TD learning.
      The number of Q functions can be designated by `n_critics`.
    * TD3 adds noise to target value estimation to avoid overfitting with the
      deterministic policy.
    * TD3 updates the policy function after several Q function updates in order
      to reduce variance of action-value estimation. The interval of the policy
      function update can be designated by `update_actor_interval`.

    .. math::

        L(\theta_i) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma \min_j Q_{\theta_j'}(s_{t+1}, \pi_{\phi'}(s_{t+1}) +
            \epsilon) - Q_{\theta_i}(s_t, a_t))^2]

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D}
            [\min_i Q_{\theta_i}(s_t, \pi_\phi(s_t))]

    where :math:`\epsilon \sim clip (N(0, \sigma), -c, c)`

    References:
        * `Fujimoto et al., Addressing Function Approximation Error in
          Actor-Critic Methods. <https://arxiv.org/abs/1802.09477>`_

    Args:
        actor_learning_rate (float): learning rate for a policy function.
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
        n_critics (int): the number of Q functions for ensemble.
        target_smoothing_sigma (float): standard deviation for target noise.
        target_smoothing_clip (float): clipping range for target noise.
        update_actor_interval (int): interval to update policy function
            described as `delayed policy update` in the paper.
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
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    target_smoothing_sigma: float = 0.2
    target_smoothing_clip: float = 0.5
    update_actor_interval: int = 2

    def create(self, device: DeviceArg = False) -> "TD3":
        return TD3(self, device)

    @staticmethod
    def get_type() -> str:
        return "td3"


class TD3(QLearningAlgoBase):
    _config: TD3Config
    _impl: Optional[TD3Impl]

    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        self._impl = TD3Impl(
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
            n_critics=self._config.n_critics,
            target_smoothing_sigma=self._config.target_smoothing_sigma,
            target_smoothing_clip=self._config.target_smoothing_clip,
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


register_learnable(TD3Config)
