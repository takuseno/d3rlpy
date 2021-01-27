from typing import Any, Callable, List, Optional, Union

import numpy as np
import gym
from tqdm import trange
from typing_extensions import Protocol

from ..dataset import TransitionMiniBatch
from ..envs import BatchEnv
from ..logger import D3RLPyLogger
from ..preprocessing import Scaler, ActionScaler
from ..preprocessing.stack import StackedObservation, BatchStackedObservation
from ..metrics.scorer import evaluate_on_environment
from .buffers import Buffer, BatchBuffer
from .explorers import Explorer


class _AlgoProtocol(Protocol):
    def update(
        self, epoch: int, total_step: int, batch: TransitionMiniBatch
    ) -> List[Optional[float]]:
        ...

    def build_with_env(self, env: gym.Env) -> None:
        ...

    def save_params(self, logger: D3RLPyLogger) -> None:
        ...

    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    def get_loss_labels(self) -> List[str]:
        ...

    def save_model(self, fname: str) -> None:
        ...

    @property
    def action_size(self) -> Optional[int]:
        ...

    @property
    def scaler(self) -> Optional[Scaler]:
        ...

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        ...

    @property
    def n_frames(self) -> int:
        ...

    @property
    def n_steps(self) -> int:
        ...

    @property
    def gamma(self) -> float:
        ...

    @property
    def batch_size(self) -> int:
        ...

    @property
    def impl(self) -> Optional[Any]:
        ...


def _setup_algo(algo: _AlgoProtocol, env: gym.Env) -> None:
    # initialize scaler
    if algo.scaler:
        algo.scaler.fit_with_env(env)

    # initialize action scaler
    if algo.action_scaler:
        algo.action_scaler.fit_with_env(env)

    # setup algorithm
    if algo.impl is None:
        algo.build_with_env(env)


def train_single_env(
    algo: _AlgoProtocol,
    env: gym.Env,
    buffer: Buffer,
    explorer: Optional[Explorer] = None,
    n_steps: int = 1000000,
    n_steps_per_epoch: int = 10000,
    update_interval: int = 1,
    update_start_step: int = 0,
    eval_env: Optional[gym.Env] = None,
    eval_epsilon: float = 0.0,
    save_metrics: bool = True,
    save_interval: int = 1,
    experiment_name: Optional[str] = None,
    with_timestamp: bool = True,
    logdir: str = "d3rlpy_logs",
    verbose: bool = True,
    show_progress: bool = True,
    tensorboard: bool = True,
    timelimit_aware: bool = True,
) -> None:
    """Start training loop of online deep reinforcement learning.

    Args:
        algo: algorithm object.
        env: gym-like environment.
        buffer : replay buffer.
        explorer: action explorer.
        n_steps: the number of total steps to train.
        n_steps_per_epoch: the number of steps per epoch.
        update_interval: the number of steps per update.
        update_start_step: the steps before starting updates.
        eval_env: gym-like environment. If None, evaluation is skipped.
        eval_epsilon: :math:`\\epsilon`-greedy factor during evaluation.
        save_metrics: flag to record metrics. If False, the log
            directory is not created and the model parameters are not saved.
        save_interval: the number of epochs before saving models.
        experiment_name: experiment name for logging. If not passed,
            the directory name will be ``{class name}_online_{timestamp}``.
        with_timestamp: flag to add timestamp string to the last of
            directory name.
        logdir: root directory name to save logs.
        verbose: flag to show logged information on stdout.
        show_progress: flag to show progress bar for iterations.
        tensorboard: flag to save logged information in tensorboard
            (additional to the csv data)
        timelimit_aware: flag to turn ``terminal`` flag ``False`` when
            ``TimeLimit.truncated`` flag is ``True``, which is designed to
            incorporate with ``gym.wrappers.TimeLimit``.

    """
    # initialize algorithm parameters
    _setup_algo(algo, env)

    # setup logger
    if experiment_name is None:
        experiment_name = algo.__class__.__name__ + "_online"

    logger = D3RLPyLogger(
        experiment_name,
        save_metrics=save_metrics,
        root_dir=logdir,
        verbose=verbose,
        tensorboard=tensorboard,
        with_timestamp=with_timestamp,
    )

    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    # prepare stacked observation
    if is_image:
        stacked_frame = StackedObservation(observation_shape, algo.n_frames)

    # save hyperparameters
    algo.save_params(logger)

    # switch based on show_progress flag
    xrange = trange if show_progress else range

    # setup evaluation scorer
    eval_scorer: Optional[Callable[..., float]]
    if eval_env:
        eval_scorer = evaluate_on_environment(eval_env, epsilon=eval_epsilon)
    else:
        eval_scorer = None

    # start training loop
    observation, reward, terminal = env.reset(), 0.0, False
    clip_episode = False
    for total_step in xrange(n_steps):
        with logger.measure_time("step"):
            # stack observation if necessary
            if is_image:
                stacked_frame.append(observation)
                fed_observation = stacked_frame.eval()
            else:
                observation = observation.astype("f4")
                fed_observation = observation

            # sample exploration action
            with logger.measure_time("inference"):
                if explorer:
                    x = fed_observation.reshape((1,) + fed_observation.shape)
                    action = explorer.sample(algo, x, total_step)[0]
                else:
                    action = algo.sample_action([fed_observation])[0]

            # store observation
            buffer.append(
                observation=observation,
                action=action,
                reward=reward,
                terminal=terminal,
                clip_episode=clip_episode,
            )

            # get next observation
            if clip_episode:
                observation, reward, terminal = env.reset(), 0.0, False
                clip_episode = False
                # for image observation
                if is_image:
                    stacked_frame.clear()
            else:
                with logger.measure_time("environment_step"):
                    observation, reward, terminal, info = env.step(action)

                # special case for TimeLimit wrapper
                if timelimit_aware and "TimeLimit.truncated" in info:
                    clip_episode = True
                    terminal = False
                else:
                    clip_episode = terminal

            # psuedo epoch count
            epoch = total_step // n_steps_per_epoch

            if total_step > update_start_step and len(buffer) > algo.batch_size:
                if total_step % update_interval == 0:
                    # sample mini-batch
                    with logger.measure_time("sample_batch"):
                        batch = buffer.sample(
                            batch_size=algo.batch_size,
                            n_frames=algo.n_frames,
                            n_steps=algo.n_steps,
                            gamma=algo.gamma,
                        )

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = algo.update(
                            epoch=epoch,
                            total_step=total_step // update_interval,
                            batch=batch,
                        )

                    # record metrics
                    for name, val in zip(algo.get_loss_labels(), loss):
                        if val:
                            logger.add_metric(name, val)

        if epoch > 0 and total_step % n_steps_per_epoch == 0:
            # evaluation
            if eval_scorer:
                logger.add_metric("evaluation", eval_scorer(algo))

        if epoch % save_interval == 0:
            # save metrics
            logger.commit(epoch, total_step)
            logger.save_model(total_step, algo)


def train_batch_env(
    algo: _AlgoProtocol,
    env: BatchEnv,
    buffer: BatchBuffer,
    explorer: Optional[Explorer] = None,
    n_epochs: int = 1000,
    n_steps_per_epoch: int = 1000,
    n_updates_per_epoch: int = 1000,
    eval_interval: int = 10,
    eval_env: Optional[gym.Env] = None,
    eval_epsilon: float = 0.0,
    save_metrics: bool = True,
    save_interval: int = 1,
    experiment_name: Optional[str] = None,
    with_timestamp: bool = True,
    logdir: str = "d3rlpy_logs",
    verbose: bool = True,
    show_progress: bool = True,
    tensorboard: bool = True,
    timelimit_aware: bool = True,
) -> None:
    """Start training loop of online deep reinforcement learning.

    Args:
        algo: algorithm object.
        env: gym-like environment.
        buffer : replay buffer.
        explorer: action explorer.
        n_epochs: the number of epochs to train.
        n_steps_per_epoch: the number of steps per epoch.
        n_updates_per_epoch: the number of updates per epoch.
        eval_interval: the number of epochs before evaluation.
        eval_env: gym-like environment. If None, evaluation is skipped.
        eval_epsilon: :math:`\\epsilon`-greedy factor during evaluation.
        save_metrics: flag to record metrics. If False, the log
            directory is not created and the model parameters are not saved.
        save_interval: the number of epochs before saving models.
        experiment_name: experiment name for logging. If not passed,
            the directory name will be ``{class name}_online_{timestamp}``.
        with_timestamp: flag to add timestamp string to the last of
            directory name.
        logdir: root directory name to save logs.
        verbose: flag to show logged information on stdout.
        show_progress: flag to show progress bar for iterations.
        tensorboard: flag to save logged information in tensorboard
            (additional to the csv data)
        timelimit_aware: flag to turn ``terminal`` flag ``False`` when
            ``TimeLimit.truncated`` flag is ``True``, which is designed to
            incorporate with ``gym.wrappers.TimeLimit``.

    """
    # initialize algorithm parameters
    _setup_algo(algo, env)

    # setup logger
    if experiment_name is None:
        experiment_name = algo.__class__.__name__ + "_online"

    logger = D3RLPyLogger(
        experiment_name,
        save_metrics=save_metrics,
        root_dir=logdir,
        verbose=verbose,
        tensorboard=tensorboard,
        with_timestamp=with_timestamp,
    )

    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    # prepare stacked observation
    if is_image:
        stacked_frame = BatchStackedObservation(
            observation_shape, algo.n_frames, len(env)
        )

    # save hyperparameters
    algo.save_params(logger)

    # switch based on show_progress flag
    xrange = trange if show_progress else range

    # setup evaluation scorer
    eval_scorer: Optional[Callable[..., float]]
    if eval_env:
        eval_scorer = evaluate_on_environment(eval_env, epsilon=eval_epsilon)
    else:
        eval_scorer = None

    # start training loop
    observation = env.reset()
    reward, terminal = np.zeros(len(env)), np.zeros(len(env))
    clip_episode = np.zeros(len(env))
    for epoch in range(n_epochs):
        for step in xrange(n_steps_per_epoch):

            total_step = len(env) * (epoch * n_steps_per_epoch + step)

            # stack observation if necessary
            if is_image:
                stacked_frame.append(observation)
                fed_observation = stacked_frame.eval()
            else:
                observation = observation.astype("f4")
                fed_observation = observation

            # sample exploration action
            with logger.measure_time("inference"):
                if explorer:
                    action = explorer.sample(algo, fed_observation, total_step)
                else:
                    action = algo.sample_action(fed_observation)

            # store observation
            buffer.append(
                observations=observation,
                actions=action,
                rewards=reward,
                terminals=terminal,
                clip_episodes=clip_episode,
            )

            # get next observation
            with logger.measure_time("environment_step"):
                observation, reward, terminal, infos = env.step(action)

            # special case for TimeLimit wrapper
            for i, info in enumerate(infos):
                if timelimit_aware and "TimeLimit.truncated" in info:
                    clip_episode[i] = 1.0
                    terminal[i] = 0.0
                else:
                    clip_episode[i] = terminal[i]

                if clip_episode[i] and is_image:
                    stacked_frame.clear_by_index(i)

        for step in range(n_updates_per_epoch):
            # sample mini-batch
            with logger.measure_time("sample_batch"):
                batch = buffer.sample(
                    batch_size=algo.batch_size,
                    n_frames=algo.n_frames,
                    n_steps=algo.n_steps,
                    gamma=algo.gamma,
                )

            # update parameters
            with logger.measure_time("algorithm_update"):
                loss = algo.update(
                    epoch=epoch,
                    total_step=epoch * n_updates_per_epoch + step,
                    batch=batch,
                )

            # record metrics
            for name, val in zip(algo.get_loss_labels(), loss):
                if val:
                    logger.add_metric(name, val)

        if epoch % eval_interval == 0:
            # evaluation
            if eval_scorer:
                logger.add_metric("evaluation", eval_scorer(algo))

        if epoch % save_interval == 0:
            # save metrics
            logger.commit(epoch, total_step)
            logger.save_model(total_step, algo)

    # finish all process
    env.close()
