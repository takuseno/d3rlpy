from typing import Any, List, Optional, Sequence, Tuple, cast

import numpy as np

from ..argument_utility import (
    ActionScalerArg,
    AugmentationArg,
    EncoderArg,
    QFuncArg,
    ScalerArg,
    UseGPUArg,
    check_augmentation,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from ..augmentation import AugmentationPipeline
from ..constants import DYNAMICS_NOT_GIVEN_ERROR, IMPL_NOT_INITIALIZED_ERROR
from ..dataset import Transition, TransitionMiniBatch
from ..dynamics import DynamicsBase
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .base import AlgoBase
from .torch.sac_impl import SACImpl


class MOPO(AlgoBase):
    r"""Model-based Offline Policy Optimization.

    MOPO is a model-based RL approach for offline policy optimization.
    MOPO leverages the probablistic ensemble dynamics model to generate
    new dynamics data with uncertainty penalties.
    The ensemble dynamics model consists of :math:`N` probablistic models
    :math:`\{T_{\theta_i}\}_{i=1}^N`.
    At each epoch, new transitions are generated via randomly picked dynamics
    model :math:`T_\theta`.

    .. math::
        s_{t+1}, r_{t+1} \sim T_\theta(s_t, a_t)

    where :math:`s_t \sim D` for the first step, otherwise :math:`s_t` is the
    previous generated observation, and :math:`a_t \sim \pi(\cdot|s_t)`.
    The generated :math:`r_{t+1}` would be far from the ground truth if the
    actions sampled from the policy function is out-of-distribution.
    Thus, the uncertainty penalty reguralizes this bias.

    .. math::
        \tilde{r_{t+1}} = r_{t+1} - \lambda \max_{i=1}^N
            || \Sigma_i (s_t, a_t) ||

    where :math:`\Sigma(s_t, a_t)` is the estimated variance.
    Finally, the generated transitions
    :math:`(s_t, a_t, \tilde{r_{t+1}}, s_{t+1})` are appended to dataset
    :math:`D`.
    This generation process starts with randomly sampled
    ``n_initial_transitions`` transitions till ``horizon`` steps.

    Note:
        Currently, MOPO only supports vector observations.

    References:
        * `Yu et al., MOPO: Model-based Offline Policy Optimization.
          <https://arxiv.org/abs/2005.13239>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float): learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        target_reduction_type (str): ensemble reduction method at target value
            estimation. The available options are
            ``['min', 'max', 'mean', 'mix', 'none']``.
        update_actor_interval (int): interval to update policy function.
        initial_temperature (float): initial temperature value.
        dynamics (d3rlpy.dynamics.DynamicsBase): dynamics object.
        n_ensembles (int): the number of dynamics models for ensemble.
        rollout_interval (int): the number of steps before rollout.
        horizon (int): the rollout step length.
        n_initial_transitions (int): the number of initial transitions for
            rollout.
        lam (float): :math:`\lambda` for uncertainty penalties.
        real_ratio (float): the real of dataset samples in a mini-batch.
        generated_maxlen (int): the maximum number of generated samples.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        augmentation (d3rlpy.augmentation.AugmentationPipeline or list(str)):
            augmentation pipeline.
        impl (d3rlpy.algos.torch.sac_impl.SACImpl): algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _temp_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _temp_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _target_reduction_type: str
    _update_actor_interval: int
    _initial_temperature: float
    _dynamics: Optional[DynamicsBase]
    _n_ensembles: int
    _rollout_interval: int
    _horizon: int
    _n_initial_transitions: int
    _lam: float
    _augmentation: AugmentationPipeline
    _use_gpu: Optional[Device]
    _impl: Optional[SACImpl]

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        temp_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        temp_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 2,
        target_reduction_type: str = "min",
        update_actor_interval: int = 1,
        initial_temperature: float = 1.0,
        dynamics: Optional[DynamicsBase] = None,
        n_ensembles: int = 5,
        rollout_interval: int = 100000,
        horizon: int = 5,
        n_initial_transitions: int = 400,
        lam: float = 1.0,
        real_ratio: float = 0.5,
        generated_maxlen: int = 100000,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        augmentation: AugmentationArg = None,
        impl: Optional[SACImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._temp_learning_rate = temp_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._temp_optim_factory = temp_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._target_reduction_type = target_reduction_type
        self._update_actor_interval = update_actor_interval
        self._initial_temperature = initial_temperature
        self._dynamics = dynamics
        self._n_ensembles = n_ensembles
        self._rollout_interval = rollout_interval
        self._horizon = horizon
        self._n_initial_transitions = n_initial_transitions
        self._lam = lam
        self._augmentation = check_augmentation(augmentation)
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = SACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            temp_learning_rate=self._temp_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            target_reduction_type=self._target_reduction_type,
            initial_temperature=self._initial_temperature,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            augmentation=self._augmentation,
        )
        self._impl.build()

    def update(
        self, epoch: int, total_step: int, batch: TransitionMiniBatch
    ) -> List[Optional[float]]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        critic_loss = self._impl.update_critic(
            batch.observations,
            batch.actions,
            batch.next_rewards,
            batch.next_observations,
            batch.terminals,
            batch.n_steps,
            batch.masks,
        )

        # delayed policy update
        if total_step % self._update_actor_interval == 0:
            actor_loss = self._impl.update_actor(batch.observations)

            # lagrangian parameter update for SAC temperature
            if self._temp_learning_rate > 0:
                temp_loss, temp = self._impl.update_temp(batch.observations)
            else:
                temp_loss, temp = None, None

            self._impl.update_critic_target()
            self._impl.update_actor_target()
        else:
            actor_loss = None
            temp_loss = None
            temp = None

        return [critic_loss, actor_loss, temp_loss, temp]

    def get_loss_labels(self) -> List[str]:
        return ["critic_loss", "actor_loss", "temp_loss", "temp"]

    def generate_new_data(
        self, epoch: int, total_step: int, transitions: List[Transition]
    ) -> Optional[List[Transition]]:
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert self._dynamics, DYNAMICS_NOT_GIVEN_ERROR

        if total_step % self._rollout_interval != 0:
            return None

        # uniformly sample transitions
        init_transitions: List[Transition] = []
        indices = np.random.randint(
            len(transitions), size=self._n_initial_transitions
        )
        for i in indices:
            init_transitions.append(transitions[i])

        rets: List[Transition] = []

        # rollout
        batch = TransitionMiniBatch(init_transitions)
        observations = batch.observations
        actions = self.sample_action(observations)
        rewards = batch.rewards
        prev_transitions: List[Transition] = []
        for _ in range(self._horizon):
            # predict next state
            pred = self._dynamics.predict(observations, actions, True)
            pred = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], pred)
            next_observations, next_rewards, variances = pred

            # regularize by uncertainty
            next_rewards -= self._lam * variances

            # sample policy action
            next_actions = self.sample_action(next_observations)

            # append new transitions
            new_transitions = []
            for i in range(self._n_initial_transitions):
                transition = Transition(
                    observation_shape=self._impl.observation_shape,
                    action_size=self._impl.action_size,
                    observation=observations[i],
                    action=actions[i],
                    reward=float(rewards[i][0]),
                    next_observation=next_observations[i],
                    next_action=next_actions[i],
                    next_reward=float(next_rewards[i][0]),
                    terminal=0.0,
                )

                if prev_transitions:
                    prev_transitions[i].next_transition = transition
                    transition.prev_transition = prev_transitions[i]

                new_transitions.append(transition)

            prev_transitions = new_transitions
            rets += new_transitions
            observations = next_observations.copy()
            actions = next_actions.copy()
            rewards = next_rewards.copy()

        return rets
