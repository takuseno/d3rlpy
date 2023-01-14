from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import gym
import numpy as np
from tqdm.auto import trange

from ..base import ImplBase, LearnableBase, save_config
from ..constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
)
from ..dataset import ReplayBuffer, create_fifo_replay_buffer
from ..logger import LOG, D3RLPyLogger
from ..metrics.scorer import evaluate_on_environment
from .explorers import Explorer

__all__ = ["AlgoImplBase", "AlgoBase"]


def _assert_action_space(algo: LearnableBase, env: gym.Env) -> None:
    if isinstance(env.action_space, gym.spaces.Box):
        assert (
            algo.get_action_type() == ActionSpace.CONTINUOUS
        ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR
    elif isinstance(env.action_space, gym.spaces.discrete.Discrete):
        assert (
            algo.get_action_type() == ActionSpace.DISCRETE
        ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
    else:
        action_space = type(env.action_space)
        raise ValueError(f"The action-space is not supported: {action_space}")


class AlgoImplBase(ImplBase):
    @abstractmethod
    def save_policy(self, fname: str) -> None:
        pass

    @abstractmethod
    def predict_best_action(
        self, x: Union[np.ndarray, List[Any]]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        pass

    @abstractmethod
    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        pass

    def copy_policy_from(self, impl: "AlgoImplBase") -> None:
        raise NotImplementedError

    def copy_policy_optim_from(self, impl: "AlgoImplBase") -> None:
        raise NotImplementedError

    def copy_q_function_from(self, impl: "AlgoImplBase") -> None:
        raise NotImplementedError

    def copy_q_function_optim_from(self, impl: "AlgoImplBase") -> None:
        raise NotImplementedError

    def reset_optimizer_states(self) -> None:
        raise NotImplementedError


def _setup_algo(algo: "AlgoBase", env: gym.Env) -> None:
    # initialize observation scaler
    if algo.observation_scaler:
        LOG.debug(
            "Fitting observation scaler...",
            observation_scaler=algo.observation_scaler.get_type(),
        )
        algo.observation_scaler.fit_with_env(env)

    # initialize action scaler
    if algo.action_scaler:
        LOG.debug(
            "Fitting action scaler...",
            action_scler=algo.action_scaler.get_type(),
        )
        algo.action_scaler.fit_with_env(env)

    # setup algorithm
    if algo.impl is None:
        LOG.debug("Building model...")
        algo.build_with_env(env)
        LOG.debug("Model has been built.")
    else:
        LOG.warning("Skip building models since they're already built.")


class AlgoBase(LearnableBase):

    _impl: Optional[AlgoImplBase]

    def save_policy(self, fname: str) -> None:
        """Save the greedy-policy computational graph as TorchScript or ONNX.

        The format will be automatically detected by the file name.

        .. code-block:: python

            # save as TorchScript
            algo.save_policy('policy.pt')

            # save as ONNX
            algo.save_policy('policy.onnx')

        The artifacts saved with this method will work without d3rlpy.
        This method is especially useful to deploy the learned policy to
        production environments or embedding systems.

        See also

            * https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html (for Python).
            * https://pytorch.org/tutorials/advanced/cpp_export.html (for C++).
            * https://onnx.ai (for ONNX)

        Args:
            fname: destination file path.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        self._impl.save_policy(fname)

    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        """Returns greedy actions.

        .. code-block:: python

            # 100 observations with shape of (10,)
            x = np.random.random((100, 10))

            actions = algo.predict(x)
            # actions.shape == (100, action size) for continuous control
            # actions.shape == (100,) for discrete control

        Args:
            x: observations

        Returns:
            greedy actions

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        return self._impl.predict_best_action(x)

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Returns predicted action-values.

        .. code-block:: python

            # 100 observations with shape of (10,)
            x = np.random.random((100, 10))

            # for continuous control
            # 100 actions with shape of (2,)
            actions = np.random.random((100, 2))

            # for discrete control
            # 100 actions in integer values
            actions = np.random.randint(2, size=100)

            values = algo.predict_value(x, actions)
            # values.shape == (100,)

            values, stds = algo.predict_value(x, actions, with_std=True)
            # stds.shape  == (100,)

        Args:
            x: observations
            action: actions
            with_std: flag to return standard deviation of ensemble
                estimation. This deviation reflects uncertainty for the given
                observations. This uncertainty will be more accurate if you
                enable ``bootstrap`` flag and increase ``n_critics`` value.

        Returns:
            predicted action-values

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        return self._impl.predict_value(x, action, with_std)

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        """Returns sampled actions.

        The sampled actions are identical to the output of `predict` method if
        the policy is deterministic.

        Args:
            x: observations.

        Returns:
            sampled actions.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        return self._impl.sample_action(x)

    def fit_online(
        self,
        env: gym.Env,
        buffer: Optional[ReplayBuffer] = None,
        explorer: Optional[Explorer] = None,
        n_steps: int = 1000000,
        n_steps_per_epoch: int = 10000,
        update_interval: int = 1,
        update_start_step: int = 0,
        random_steps: int = 0,
        eval_env: Optional[gym.Env] = None,
        eval_epsilon: float = 0.0,
        save_metrics: bool = True,
        save_interval: int = 1,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        timelimit_aware: bool = True,
        callback: Optional[Callable[["AlgoBase", int, int], None]] = None,
    ) -> None:
        """Start training loop of online deep reinforcement learning.

        Args:
            env: gym-like environment.
            buffer : replay buffer.
            explorer: action explorer.
            n_steps: the number of total steps to train.
            n_steps_per_epoch: the number of steps per epoch.
            update_interval: the number of steps per update.
            update_start_step: the steps before starting updates.
            random_steps: the steps for the initial random explortion.
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
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            timelimit_aware: flag to turn ``terminal`` flag ``False`` when
                ``TimeLimit.truncated`` flag is ``True``, which is designed to
                incorporate with ``gym.wrappers.TimeLimit``.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called at the end of epochs.

        """

        # create default replay buffer
        if buffer is None:
            buffer = create_fifo_replay_buffer(1000000)

        # check action-space
        _assert_action_space(self, env)

        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__ + "_online"
        logger = D3RLPyLogger(
            experiment_name,
            save_metrics=save_metrics,
            root_dir=logdir,
            verbose=verbose,
            tensorboard_dir=tensorboard_dir,
            with_timestamp=with_timestamp,
        )

        # initialize algorithm parameters
        _setup_algo(self, env)

        # save hyperparameters
        save_config(self, logger)

        # switch based on show_progress flag
        xrange = trange if show_progress else range

        # setup evaluation scorer
        eval_scorer: Optional[Callable[..., float]]
        if eval_env:
            eval_scorer = evaluate_on_environment(
                eval_env, epsilon=eval_epsilon
            )
        else:
            eval_scorer = None

        # start training loop
        observation = env.reset()
        rollout_return = 0.0
        for total_step in xrange(1, n_steps + 1):
            with logger.measure_time("step"):
                # sample exploration action
                with logger.measure_time("inference"):
                    if total_step < random_steps:
                        action = env.action_space.sample()
                    elif explorer:
                        x = observation.reshape((1,) + observation.shape)
                        action = explorer.sample(self, x, total_step)[0]
                    else:
                        action = self.sample_action([observation])[0]

                # step environment
                with logger.measure_time("environment_step"):
                    next_observation, reward, terminal, info = env.step(action)
                    rollout_return += reward

                # special case for TimeLimit wrapper
                if timelimit_aware and "TimeLimit.truncated" in info:
                    clip_episode = True
                    terminal = False
                else:
                    clip_episode = terminal

                # store observation
                buffer.append(observation, action, reward)

                # reset if terminated
                if clip_episode:
                    buffer.clip_episode(terminal)
                    observation = env.reset()
                    logger.add_metric("rollout_return", rollout_return)
                    rollout_return = 0.0
                else:
                    observation = next_observation

                # psuedo epoch count
                epoch = total_step // n_steps_per_epoch

                if (
                    total_step > update_start_step
                    and buffer.transition_count > self.batch_size
                ):
                    if total_step % update_interval == 0:
                        # sample mini-batch
                        with logger.measure_time("sample_batch"):
                            batch = buffer.sample_transition_batch(
                                self.batch_size
                            )

                        # update parameters
                        with logger.measure_time("algorithm_update"):
                            loss = self.update(batch)

                        # record metrics
                        for name, val in loss.items():
                            logger.add_metric(name, val)

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            if epoch > 0 and total_step % n_steps_per_epoch == 0:
                # evaluation
                if eval_scorer:
                    logger.add_metric("evaluation", eval_scorer(self))

                if epoch % save_interval == 0:
                    logger.save_model(total_step, self)

                # save metrics
                logger.commit(epoch, total_step)

        # clip the last episode
        buffer.clip_episode(False)

        # close logger
        logger.close()

    def collect(
        self,
        env: gym.Env,
        buffer: Optional[ReplayBuffer] = None,
        explorer: Optional[Explorer] = None,
        deterministic: bool = False,
        n_steps: int = 1000000,
        show_progress: bool = True,
        timelimit_aware: bool = True,
    ) -> ReplayBuffer:
        """Collects data via interaction with environment.

        If ``buffer`` is not given, ``ReplayBuffer`` will be internally created.

        Args:
            env: gym-like environment.
            buffer : replay buffer.
            explorer: action explorer.
            deterministic: flag to collect data with the greedy policy.
            n_steps: the number of total steps to train.
            show_progress: flag to show progress bar for iterations.
            timelimit_aware: flag to turn ``terminal`` flag ``False`` when
                ``TimeLimit.truncated`` flag is ``True``, which is designed to
                incorporate with ``gym.wrappers.TimeLimit``.

        Returns:
            replay buffer with the collected data.

        """
        # create default replay buffer
        if buffer is None:
            buffer = create_fifo_replay_buffer(1000000)

        # check action-space
        _assert_action_space(self, env)

        # initialize algorithm parameters
        _setup_algo(self, env)

        # switch based on show_progress flag
        xrange = trange if show_progress else range

        # start training loop
        observation = env.reset()
        for total_step in xrange(1, n_steps + 1):
            # sample exploration action
            if deterministic:
                action = self.predict([observation])[0]
            else:
                if explorer:
                    x = observation.reshape((1,) + observation.shape)
                    action = explorer.sample(self, x, total_step)[0]
                else:
                    action = self.sample_action([observation])[0]

            # step environment
            next_observation, reward, terminal, info = env.step(action)

            # special case for TimeLimit wrapper
            if timelimit_aware and "TimeLimit.truncated" in info:
                clip_episode = True
                terminal = False
            else:
                clip_episode = terminal

            # store observation
            buffer.append(observation, action, reward)

            # reset if terminated
            if clip_episode:
                buffer.clip_episode(terminal)
                observation = env.reset()
            else:
                observation = next_observation

        # clip the last episode
        buffer.clip_episode(False)

        return buffer

    def copy_policy_from(self, algo: "AlgoBase") -> None:
        """Copies policy parameters from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithm
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_policy_from(cql)

        Args:
            algo: algorithm object.

        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, AlgoImplBase)
        self._impl.copy_policy_from(algo.impl)

    def copy_policy_optim_from(self, algo: "AlgoBase") -> None:
        """Copies policy optimizer states from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithm
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_policy_optim_from(cql)

        Args:
            algo: algorithm object.

        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, AlgoImplBase)
        self._impl.copy_policy_optim_from(algo.impl)

    def copy_q_function_from(self, algo: "AlgoBase") -> None:
        """Copies Q-function parameters from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithmn
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_q_function_from(cql)

        Args:
            algo: algorithm object.

        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, AlgoImplBase)
        self._impl.copy_q_function_from(algo.impl)

    def copy_q_function_optim_from(self, algo: "AlgoBase") -> None:
        """Copies Q-function optimizer states from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithm
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_policy_optim_from(cql)

        Args:
            algo: algorithm object.

        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, AlgoImplBase)
        self._impl.copy_q_function_optim_from(algo.impl)

    def reset_optimizer_states(self) -> None:
        """Resets optimizer states.

        This is especially useful when fine-tuning policies with setting inital
        optimizer states.

        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        self._impl.reset_optimizer_states()
