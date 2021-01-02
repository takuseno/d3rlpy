import numpy as np
import gym

from abc import abstractmethod
from typing import List, Optional, Union
from ..base import ImplBase, LearnableBase
from ..dataset import Transition
from ..dynamics.base import DynamicsBase
from ..online.iterators import train
from ..online.buffers import Buffer
from ..online.explorers import Explorer
from ..preprocessing import Scaler


class AlgoImplBase(ImplBase):
    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def save_policy(self, fname: str, as_onnx: bool) -> None:
        pass

    @abstractmethod
    def predict_best_action(self, x: Union[np.ndarray, list]) -> np.ndarray:
        pass

    @abstractmethod
    def predict_value(
        self,
        x: Union[np.ndarray, list],
        action: Union[np.ndarray, list],
        with_std: bool,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def sample_action(self, x: Union[np.ndarray, list]) -> np.ndarray:
        pass


class AlgoBase(LearnableBase):

    _dynamics: Optional[DynamicsBase]
    _impl: Optional[AlgoImplBase]

    def __init__(
        self,
        batch_size: int,
        n_frames: int,
        n_steps: int,
        gamma: float,
        scaler: Optional[Scaler],
        dynamics: Optional[DynamicsBase],
    ):
        super().__init__(batch_size, n_frames, n_steps, gamma, scaler)
        self._dynamics = dynamics

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
            fname (str): destination file path.
            as_onnx (bool): flag to save as ONNX format.

        """
        assert self._impl is not None
        self._impl.save_policy(fname, as_onnx)

    def predict(self, x: Union[np.ndarray, list]) -> np.ndarray:
        """Returns greedy actions.

        .. code-block:: python

            # 100 observations with shape of (10,)
            x = np.random.random((100, 10))

            actions = algo.predict(x)
            # actions.shape == (100, action size) for continuous control
            # actions.shape == (100,) for discrete control

        Args:
            x (numpy.ndarray): observations

        Returns:
            numpy.ndarray: greedy actions

        """
        assert self._impl is not None
        return self._impl.predict_best_action(x)

    def predict_value(
        self,
        x: Union[np.ndarray, list],
        action: Union[np.ndarray, list],
        with_std: bool = False,
    ) -> np.ndarray:
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
            x (numpy.ndarray): observations
            action (numpy.ndarray): actions
            with_std (bool): flag to return standard deviation of ensemble
                estimation. This deviation reflects uncertainty for the given
                observations. This uncertainty will be more accurate if you
                enable `bootstrap` flag and increase `n_critics` value.

        Returns:
            numpy.ndarray: predicted action-values

        """
        assert self._impl is not None
        return self._impl.predict_value(x, action, with_std)

    def sample_action(self, x: Union[np.ndarray, list]) -> np.ndarray:
        """Returns sampled actions.

        The sampled actions are identical to the output of `predict` method if
        the policy is deterministic.

        Args:
            x (numpy.ndarray): observations.

        Returns:
            numpy.ndarray: sampled actions.

        """
        assert self._impl is not None
        return self._impl.sample_action(x)

    def fit_online(
        self,
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
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard: bool = True,
    ) -> None:
        """Start training loop of online deep reinforcement learning.

        This method is a convenient alias to ``d3rlpy.online.iterators.train``.

        Args:
            env (gym.Env): gym-like environment.
            buffer (d3rlpy.online.buffers.Buffer): replay buffer.
            explorer (d3rlpy.online.explorers.Explorer): action explorer.
            n_steps (int): the number of total steps to train.
            n_steps_per_epoch (int): the number of steps per epoch.
            update_interval (int): the number of steps per update.
            update_start_step (int): the steps before starting updates.
            eval_env (gym.Env): gym-like environment. If None, evaluation is
                skipped.
            eval_epsilon (float): :math:`\\epsilon`-greedy factor during
                evaluation.
            save_metrics (bool): flag to record metrics. If False, the log
                directory is not created and the model parameters are not saved.
            experiment_name (str): experiment name for logging. If not passed,
                the directory name will be `{class name}_online_{timestamp}`.
            with_timestamp (bool): flag to add timestamp string to the last of
                directory name.
            logdir (str): root directory name to save logs.
            verbose (bool): flag to show logged information on stdout.
            show_progress (bool): flag to show progress bar for iterations.
            tensorboard (bool): flag to save logged information in tensorboard
                (additional to the csv data)

        """
        train(
            env=env,
            algo=self,
            buffer=buffer,
            explorer=explorer,
            n_steps=n_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            update_interval=update_interval,
            update_start_step=update_start_step,
            eval_env=eval_env,
            eval_epsilon=eval_epsilon,
            save_metrics=save_metrics,
            experiment_name=experiment_name,
            with_timestamp=with_timestamp,
            logdir=logdir,
            verbose=verbose,
            show_progress=show_progress,
            tensorboard=tensorboard,
        )

    def _generate_new_data(
        self, transitions: List[Transition]
    ) -> List[Transition]:
        new_data = []
        if self._dynamics:
            new_data += self._dynamics.generate(self, transitions)
        return new_data
