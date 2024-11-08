import dataclasses
from typing import Callable, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing_extensions import Self

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace, LoggingStrategy
from ...dataset import ReplayBufferBase
from ...logging import FileAdapterFactory, LoggerAdapterFactory
from ...metrics import EvaluatorProtocol
from ...models.builders import (
    create_continuous_q_function,
    create_deterministic_policy,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...optimizers.optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
from ..utility import build_scalers_with_transition_picker
from .base import QLearningAlgoBase
from .torch.ddpg_impl import DDPGModules
from .torch.prdc_impl import PRDCImpl

__all__ = ["PRDCConfig", "PRDC"]


@dataclasses.dataclass()
class PRDCConfig(LearnableConfig):
    r"""Config of PRDC algorithm.

    PRDC is an simple offline RL algorithm built on top of TD3.
    PRDC introduces Dataset Constraint (DC)-reguralized policy objective
    function.

    .. math::

        J(\phi) = \mathbb{E}_{s \sim D}
            [\lambda Q(s, \pi(s)) - d^\beta_D(s, \pi(s))]

    where

    .. math::

        \lambda = \frac{\alpha}{\frac{1}{N} \sum_(s_i, a_i) |Q(s_i, a_i)|}

    and `d^\beta_\mathcal{D}(s,\pi(s))` is the DC loss, defined as

    .. math::

        d^\beta_\mathcal{D}(s,\pi(s)) = \min_{\hat{s}, \hat{a} \sim D}
            [\| (\beta s) \oplus \pi(s) - (\beta \hat{s}) \oplus \hat{a} \|]

    References:
        * `Ran et al., Policy Regularization with Dataset Constraint for Offline
          Reinforcement Learning Learning.
          <https://arxiv.org/abs/2306.06569>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for a policy function.
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
        n_critics (int): Number of Q functions for ensemble.
        target_smoothing_sigma (float): Standard deviation for target noise.
        target_smoothing_clip (float): Clipping range for target noise.
        alpha (float): :math:`\alpha` value.
        beta (float): :math:`\beta` value.
        update_actor_interval (int): Interval to update policy function
            described as `delayed policy update` in the paper.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
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
    alpha: float = 2.5
    beta: float = 2.0
    update_actor_interval: int = 2

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "PRDC":
        return PRDC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "prdc"


class PRDC(QLearningAlgoBase[PRDCImpl, PRDCConfig]):
    _nbsr = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)

    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        assert not self._config.compile_graph, (
            "PRDC doesn't support compile_graph option because there is "
            "non-CUDA operation in the update logic."
        )

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

        self._impl = PRDCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            target_smoothing_sigma=self._config.target_smoothing_sigma,
            target_smoothing_clip=self._config.target_smoothing_clip,
            alpha=self._config.alpha,
            beta=self._config.beta,
            update_actor_interval=self._config.update_actor_interval,
            compiled=self.compiled,
            nbsr=self._nbsr,
            device=self._device,
        )

    def fit(
        self,
        dataset: ReplayBufferBase,
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logging_steps: int = 500,
        logging_strategy: LoggingStrategy = LoggingStrategy.EPOCH,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> list[tuple[int, dict[str, float]]]:
        observation_list = []
        action_list = []
        for episode in dataset.buffer.episodes:
            for i in range(episode.transition_count):
                transition = dataset.transition_picker(episode, i)
                observation_list.append(
                    np.reshape(transition.observation, (1, -1))
                )
                action_list.append(np.reshape(transition.action, (1, -1)))
        observations = np.concatenate(observation_list, axis=0)
        actions = np.concatenate(action_list, axis=0)

        build_scalers_with_transition_picker(self, dataset)
        if self.observation_scaler and self.observation_scaler.built:
            observations = self.observation_scaler.transform_numpy(observations)

        if self.action_scaler and self.action_scaler.built:
            actions = self.action_scaler.transform_numpy(actions)

        self._nbsr.fit(
            np.concatenate(
                [np.multiply(observations, self._config.beta), actions],
                axis=1,
            )
        )

        return super().fit(
            dataset=dataset,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            logging_steps=logging_steps,
            logging_strategy=logging_strategy,
            experiment_name=experiment_name,
            with_timestamp=with_timestamp,
            logger_adapter=logger_adapter,
            show_progress=show_progress,
            save_interval=save_interval,
            evaluators=evaluators,
            callback=callback,
            epoch_callback=epoch_callback,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(PRDCConfig)
