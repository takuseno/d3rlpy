from abc import abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import gym

from ..base import ImplBase, LearnableBase
from ..envs import BatchEnv
from ..dataset import Transition
from ..online.iterators import train_single_env, train_batch_env
from ..online.buffers import (
    Buffer,
    BatchBuffer,
    ReplayBuffer,
    BatchReplayBuffer,
)
from ..online.explorers import Explorer
from ..argument_utility import ScalerArg, ActionScalerArg
from ..constants import IMPL_NOT_INITIALIZED_ERROR


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


class DataGenerator:
    def generate(
        self, algo: "AlgoBase", transitions: List[Transition]
    ) -> List[Transition]:
        pass


class AlgoBase(LearnableBase):

    _generator: Optional[DataGenerator]
    _impl: Optional[AlgoImplBase]

    def __init__(
        self,
        batch_size: int,
        n_frames: int,
        n_steps: int,
        gamma: float,
        bootstrap: bool,
        n_critics: int,
        scaler: ScalerArg,
        action_scaler: ActionScalerArg,
        generator: Optional[DataGenerator],
    ):
        super().__init__(
            batch_size,
            n_frames,
            n_steps,
            gamma,
            bootstrap,
            n_critics,
            scaler,
            action_scaler,
        )
        self._generator = generator

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
        tensorboard: bool = True,
        timelimit_aware: bool = True,
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
            tensorboard: flag to save logged information in tensorboard
                (additional to the csv data)
            timelimit_aware: flag to turn ``terminal`` flag ``False`` when
                ``TimeLimit.truncated`` flag is ``True``, which is designed to
                incorporate with ``gym.wrappers.TimeLimit``.

        """

        # create default replay buffer
        if buffer is None:
            buffer = ReplayBuffer(1000000, env=env)

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
            tensorboard=tensorboard,
            timelimit_aware=timelimit_aware,
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
        tensorboard: bool = True,
        timelimit_aware: bool = True,
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
            tensorboard: flag to save logged information in tensorboard
                (additional to the csv data)
            timelimit_aware: flag to turn ``terminal`` flag ``False`` when
                ``TimeLimit.truncated`` flag is ``True``, which is designed to
                incorporate with ``gym.wrappers.TimeLimit``.

        """

        # create default replay buffer
        if buffer is None:
            buffer = BatchReplayBuffer(1000000, env=env)

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
            tensorboard=tensorboard,
            timelimit_aware=timelimit_aware,
        )

    def _generate_new_data(
        self, transitions: List[Transition]
    ) -> List[Transition]:
        new_data = []
        if self._generator:
            new_data += self._generator.generate(self, transitions)
        return new_data
