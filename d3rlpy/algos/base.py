from abc import ABCMeta, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import gym
from tqdm import trange

from ..base import ImplBase, LearnableBase
from ..logger import D3RLPyLogger
from ..dataset import Transition
from ..online.buffers import Buffer
from ..online.explorers import Explorer
from ..preprocessing.stack import StackedObservation
from ..metrics.scorer import evaluate_on_environment
from ..argument_utility import ScalerArg


class AlgoImplBase(ImplBase):
    @abstractmethod
    def build(self) -> None:
        pass

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


class DataGenerator(metaclass=ABCMeta):
    @abstractmethod
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
        scaler: ScalerArg,
        generator: Optional[DataGenerator],
    ):
        super().__init__(batch_size, n_frames, n_steps, gamma, scaler)
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
            fname (str): destination file path.
            as_onnx (bool): flag to save as ONNX format.

        """
        assert self._impl is not None
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
            x (numpy.ndarray): observations

        Returns:
            numpy.ndarray: greedy actions

        """
        assert self._impl is not None
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

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
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
        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__ + "_online"

        logger = D3RLPyLogger(
            experiment_name,
            save_metrics=save_metrics,
            root_dir=logdir,
            verbose=verbose,
            tensorboard=tensorboard,
            with_timestamp=with_timestamp,
        )

        self._active_logger = logger

        observation_shape = env.observation_space.shape
        is_image = len(observation_shape) == 3

        # prepare stacked observation
        if is_image:
            stacked_frame = StackedObservation(observation_shape, self.n_frames)
            n_channels = observation_shape[0]
            image_size = observation_shape[1:]
            observation_shape = (n_channels * self.n_frames, *image_size)

        # setup algorithm
        if self._impl is None:
            self.build_with_env(env)

        # save hyperparameters
        self._save_params(logger)
        batch_size = self.batch_size

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
        observation, reward, terminal = env.reset(), 0.0, False
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
                        action = explorer.sample(
                            self, fed_observation, total_step
                        )
                    else:
                        action = self.sample_action([fed_observation])[0]

                # store observation
                buffer.append(observation, action, reward, terminal)

                # get next observation
                if terminal:
                    observation, reward, terminal = env.reset(), 0.0, False
                    # for image observation
                    if is_image:
                        stacked_frame.clear()
                else:
                    with logger.measure_time("environment_step"):
                        observation, reward, terminal, _ = env.step(action)

                # psuedo epoch count
                epoch = total_step // n_steps_per_epoch

                if total_step > update_start_step and len(buffer) > batch_size:
                    if total_step % update_interval == 0:
                        # sample mini-batch
                        with logger.measure_time("sample_batch"):
                            batch = buffer.sample(
                                batch_size=batch_size,
                                n_frames=self._n_frames,
                                n_steps=self._n_steps,
                                gamma=self._gamma,
                            )

                        # update parameters
                        with logger.measure_time("algorithm_update"):
                            loss = self.update(epoch, total_step, batch)

                        # record metrics
                        for name, val in zip(self._get_loss_labels(), loss):
                            if val:
                                logger.add_metric(name, val)

            if epoch > 0 and total_step % n_steps_per_epoch == 0:
                # evaluation
                if eval_scorer:
                    logger.add_metric("evaluation", eval_scorer(self))

                # save metrics
                logger.commit(epoch, total_step)
                logger.save_model(total_step, self)

    def _generate_new_data(
        self, transitions: List[Transition]
    ) -> List[Transition]:
        new_data = []
        if self._generator:
            new_data += self._generator.generate(self, transitions)
        return new_data
