from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import gym
import numpy as np

from ..base import ImplBase, LearnableBase
from ..constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
)
from ..envs import BatchEnv
from ..models.torch.policies import Policy
from ..models.torch.q_functions.ensemble_q_function import EnsembleQFunction
from ..online.buffers import (
    BatchBuffer,
    BatchReplayBuffer,
    Buffer,
    ReplayBuffer,
)
from ..online.explorers import Explorer
from ..online.iterators import (
    AlgoProtocol,
    collect,
    train_batch_env,
    train_single_env,
)
from ..torch_utility import hard_sync


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
    def save_policy(self, fname: str, as_onnx: bool) -> None:
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

    @property
    def policy(self) -> Policy:
        raise NotImplementedError

    def copy_policy_from(self, impl: "AlgoImplBase") -> None:
        if not isinstance(impl.policy, type(self.policy)):
            raise ValueError(
                f"Invalid policy type: expected={type(self.policy)},"
                f"actual={type(impl.policy)}"
            )
        hard_sync(self.policy, impl.policy)

    @property
    def q_function(self) -> EnsembleQFunction:
        raise NotImplementedError

    def copy_q_function_from(self, impl: "AlgoImplBase") -> None:
        q_func = self.q_function.q_funcs[0]
        if not isinstance(impl.q_function.q_funcs[0], type(q_func)):
            raise ValueError(
                f"Invalid Q-function type: expected={type(q_func)},"
                f"actual={type(impl.q_function.q_funcs[0])}"
            )
        hard_sync(self.q_function, impl.q_function)


class AlgoBase(LearnableBase):

    _impl: Optional[AlgoImplBase]

    def save_policy(self, fname: str, as_onnx: bool = False) -> None:
        """Save the greedy-policy computational graph as TorchScript or ONNX.

        .. code-block:: python

            # save as TorchScript
            algo.save_policy('policy.pt')

            # save as ONNX
            algo.save_policy('policy.onnx', as_onnx=True)

        The artifacts saved with this method will work without d3rlpy.
        This method is especially useful to deploy the learned policy to
        production environments or embedding systems.

        See also

            * https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html (for Python).
            * https://pytorch.org/tutorials/advanced/cpp_export.html (for C++).
            * https://onnx.ai (for ONNX)

        Args:
            fname: destination file path.
            as_onnx: flag to save as ONNX format.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        self._impl.save_policy(fname, as_onnx)

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
        buffer: Optional[Buffer] = None,
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
        tensorboard_dir: Optional[str] = None,
        timelimit_aware: bool = True,
        callback: Optional[Callable[[AlgoProtocol, int, int], None]] = None,
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
            buffer = ReplayBuffer(1000000, env=env)

        # check action-space
        _assert_action_space(self, env)

        train_single_env(
            algo=self,
            env=env,
            buffer=buffer,
            explorer=explorer,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            update_interval=update_interval,
            update_start_step=update_start_step,
            eval_env=eval_env,
            eval_epsilon=eval_epsilon,
            save_metrics=save_metrics,
            save_interval=save_interval,
            experiment_name=experiment_name,
            with_timestamp=with_timestamp,
            logdir=logdir,
            verbose=verbose,
            show_progress=show_progress,
            tensorboard_dir=tensorboard_dir,
            timelimit_aware=timelimit_aware,
            callback=callback,
        )

    def fit_batch_online(
        self,
        env: BatchEnv,
        buffer: Optional[BatchBuffer] = None,
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
        tensorboard_dir: Optional[str] = None,
        timelimit_aware: bool = True,
        callback: Optional[Callable[[AlgoProtocol, int, int], None]] = None,
    ) -> None:
        """Start training loop of batch online deep reinforcement learning.

        Args:
            env: gym-like environment.
            buffer : replay buffer.
            explorer: action explorer.
            n_epochs: the number of epochs to train.
            n_steps_per_epoch: the number of steps per epoch.
            update_interval: the number of steps per update.
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
            buffer = BatchReplayBuffer(1000000, env=env)

        # check action-space
        _assert_action_space(self, env)

        train_batch_env(
            algo=self,
            env=env,
            buffer=buffer,
            explorer=explorer,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_updates_per_epoch=n_updates_per_epoch,
            eval_interval=eval_interval,
            eval_env=eval_env,
            eval_epsilon=eval_epsilon,
            save_metrics=save_metrics,
            save_interval=save_interval,
            experiment_name=experiment_name,
            with_timestamp=with_timestamp,
            logdir=logdir,
            verbose=verbose,
            show_progress=show_progress,
            tensorboard_dir=tensorboard_dir,
            timelimit_aware=timelimit_aware,
            callback=callback,
        )

    def collect(
        self,
        env: gym.Env,
        buffer: Optional[Buffer] = None,
        explorer: Optional[Explorer] = None,
        deterministic: bool = False,
        n_steps: int = 1000000,
        show_progress: bool = True,
        timelimit_aware: bool = True,
    ) -> Buffer:
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
            buffer = ReplayBuffer(1000000, env=env)

        # check action-space
        _assert_action_space(self, env)

        collect(
            algo=self,
            env=env,
            buffer=buffer,
            explorer=explorer,
            deterministic=deterministic,
            n_steps=n_steps,
            show_progress=show_progress,
            timelimit_aware=timelimit_aware,
        )

        return buffer

    def copy_policy_from(self, algo: "AlgoBase") -> None:
        """Copies policy parameters from the given algorithm.

        .. code-block:: python

            # pretrain with static dataset
            cql = d3rlpy.algos.CQL()
            cql.fit(dataset, n_steps=100000)

            # transfer to online algorithmn
            sac = d3rlpy.algos.SAC()
            sac.create_impl(cql.observation_shape, cql.action_size)
            sac.copy_policy_from(cql)

        Args:
            algo: algorithm object.

        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, AlgoImplBase)
        self._impl.copy_policy_from(algo.impl)

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
