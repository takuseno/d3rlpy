import dataclasses
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, cast

import gym
import numpy as np
from gym.spaces import Discrete
from tqdm.auto import tqdm

from .argument_utility import UseGPUArg, check_use_gpu
from .constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
)
from .dataset import (
    DatasetInfo,
    Episode,
    ReplayBuffer,
    Shape,
    TransitionMiniBatch,
    TransitionPickerProtocol,
)
from .gpu import Device
from .logger import LOG, D3RLPyLogger
from .preprocessing import (
    ActionScaler,
    ObservationScaler,
    RewardScaler,
    make_action_scaler_field,
    make_observation_scaler_field,
    make_reward_scaler_field,
)
from .serializable_config import DynamicConfig, generate_config_registration

__all__ = [
    "ImplBase",
    "save_config",
    "LearnableBase",
    "LearnableConfig",
    "LearnableConfigWithShape",
    "register_learnable",
]


class ImplBase(metaclass=ABCMeta):
    @abstractmethod
    def save_model(self, fname: str) -> None:
        pass

    @abstractmethod
    def load_model(self, fname: str) -> None:
        pass

    @property
    @abstractmethod
    def observation_shape(self) -> Shape:
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        pass


@dataclasses.dataclass(frozen=True)
class LearnableConfig(DynamicConfig):
    batch_size: int = 256
    gamma: float = 0.99
    observation_scaler: Optional[
        ObservationScaler
    ] = make_observation_scaler_field()
    action_scaler: Optional[ActionScaler] = make_action_scaler_field()
    reward_scaler: Optional[RewardScaler] = make_reward_scaler_field()

    def create(self, use_gpu: UseGPUArg = False) -> "LearnableBase":
        raise NotImplementedError


register_learnable, make_learnable_field = generate_config_registration(
    LearnableConfig
)


@dataclasses.dataclass(frozen=True)
class LearnableConfigWithShape(DynamicConfig):
    observation_shape: Shape
    action_size: int
    config: LearnableConfig = make_learnable_field()

    def create(self, use_gpu: UseGPUArg = False) -> "LearnableBase":
        algo = self.config.create(use_gpu)
        algo.create_impl(self.observation_shape, self.action_size)
        return algo


def save_config(alg: "LearnableBase", logger: D3RLPyLogger) -> None:
    assert alg.impl
    config = LearnableConfigWithShape(
        observation_shape=alg.impl.observation_shape,
        action_size=alg.impl.action_size,
        config=alg.config,
    )
    logger.add_params(config.serialize_to_dict())


def evaluate(
    alg: "LearnableBase",
    scorers: Dict[
        str, Callable[[Any, List[Episode], TransitionPickerProtocol], float]
    ],
    episodes: List[Episode],
    transition_picker: TransitionPickerProtocol,
    logger: D3RLPyLogger,
) -> None:
    for name, scorer in scorers.items():
        # evaluate with test data
        test_score = scorer(alg, episodes, transition_picker)
        # save metrics
        logger.add_metric(name, test_score)


class LearnableBase:
    _config: LearnableConfig
    _use_gpu: Optional[Device]
    _impl: Optional[ImplBase]
    _grad_step: int

    def __init__(
        self,
        config: LearnableConfig,
        use_gpu: UseGPUArg,
        impl: Optional[ImplBase] = None,
    ):
        self._config = config
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl
        self._grad_step = 0

    def save_model(self, fname: str) -> None:
        """Saves neural network parameters.

        .. code-block:: python

            algo.save_model('model.pt')

        Args:
            fname: destination file path.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        self._impl.save_model(fname)

    def load_model(self, fname: str) -> None:
        """Load neural network parameters.

        .. code-block:: python

            algo.load_model('model.pt')

        Args:
            fname: source file path.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        self._impl.load_model(fname)

    @classmethod
    def from_json(
        cls, fname: str, use_gpu: UseGPUArg = False
    ) -> "LearnableBase":
        config = LearnableConfigWithShape.deserialize_from_file(fname)
        return config.create(use_gpu)

    def fit(
        self,
        dataset: ReplayBuffer,
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[List[Episode]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[
                str,
                Callable[[Any, List[Episode], TransitionPickerProtocol], float],
            ]
        ] = None,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
    ) -> List[Tuple[int, Dict[str, float]]]:
        """Trains with the given dataset.

        .. code-block:: python

            algo.fit(episodes, n_steps=1000000)

        Args:
            dataset: ReplayBuffer object.
            n_steps: the number of steps to train.
            n_steps_per_epoch: the number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            save_metrics: flag to record metrics in files. If False,
                the log directory is not created and the model parameters are
                not saved during training.
            experiment_name: experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: flag to add timestamp string to the last of
                directory name.
            logdir: root directory name to save logs.
            verbose: flag to show logged information on stdout.
            show_progress: flag to show progress bar for iterations.
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            eval_episodes: list of episodes to test.
            save_interval: interval to save parameters.
            scorers: list of scorer functions used with `eval_episodes`.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.

        Returns:
            list of result tuples (epoch, metrics) per epoch.

        """
        results = list(
            self.fitter(
                dataset,
                n_steps,
                n_steps_per_epoch,
                save_metrics,
                experiment_name,
                with_timestamp,
                logdir,
                verbose,
                show_progress,
                tensorboard_dir,
                eval_episodes,
                save_interval,
                scorers,
                callback,
            )
        )
        return results

    def fitter(
        self,
        dataset: ReplayBuffer,
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard_dir: Optional[str] = None,
        eval_episodes: Optional[List[Episode]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[
                str,
                Callable[[Any, List[Episode], TransitionPickerProtocol], float],
            ]
        ] = None,
        callback: Optional[Callable[["LearnableBase", int, int], None]] = None,
    ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
             iteration algo methods and properties can be changed or queried.

        .. code-block:: python

            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)

        Args:
            dataset: offline dataset to train.
            n_steps: the number of steps to train.
            n_steps_per_epoch: the number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            save_metrics: flag to record metrics in files. If False,
                the log directory is not created and the model parameters are
                not saved during training.
            experiment_name: experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: flag to add timestamp string to the last of
                directory name.
            logdir: root directory name to save logs.
            verbose: flag to show logged information on stdout.
            show_progress: flag to show progress bar for iterations.
            tensorboard_dir: directory to save logged information in
                tensorboard (additional to the csv data).  if ``None``, the
                directory will not be created.
            eval_episodes: list of episodes to test.
            save_interval: interval to save parameters.
            scorers: list of scorer functions used with `eval_episodes`.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.

        Returns:
            iterator yielding current epoch and metrics dict.

        """
        dataset_info = DatasetInfo.from_episodes(dataset.episodes)
        LOG.info("dataset info", dataset_info=dataset_info)

        # check action space
        if self.get_action_type() == ActionSpace.BOTH:
            pass
        elif dataset_info.action_space == ActionSpace.DISCRETE:
            assert (
                self.get_action_type() == ActionSpace.DISCRETE
            ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
        else:
            assert (
                self.get_action_type() == ActionSpace.CONTINUOUS
            ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR

        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__
        logger = D3RLPyLogger(
            experiment_name,
            save_metrics=save_metrics,
            root_dir=logdir,
            verbose=verbose,
            tensorboard_dir=tensorboard_dir,
            with_timestamp=with_timestamp,
        )

        # initialize observation scaler
        if self._config.observation_scaler:
            LOG.debug(
                "Fitting observation scaler...",
                observation_scaler=self._config.observation_scaler.get_type(),
            )
            self._config.observation_scaler.fit(dataset.episodes)

        # initialize action scaler
        if self._config.action_scaler:
            LOG.debug(
                "Fitting action scaler...",
                action_scaler=self._config.action_scaler.get_type(),
            )
            self._config.action_scaler.fit(dataset.episodes)

        # initialize reward scaler
        if self._config.reward_scaler:
            LOG.debug(
                "Fitting reward scaler...",
                reward_scaler=self._config.reward_scaler.get_type(),
            )
            self._config.reward_scaler.fit(dataset.episodes)

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            action_size = dataset_info.action_size
            observation_shape = dataset.sample_transition().observation_shape
            self.create_impl(observation_shape, action_size)
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # save hyperparameters
        save_config(self, logger)

        # training loop
        n_epochs = n_steps // n_steps_per_epoch
        total_step = 0
        for epoch in range(1, n_epochs + 1):

            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(n_steps_per_epoch),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epochs}",
            )

            for itr in range_gen:
                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        batch = dataset.sample_transition_batch(
                            self._config.batch_size
                        )

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            if scorers and eval_episodes:
                evaluate(
                    alg=self,
                    scorers=scorers,
                    episodes=eval_episodes,
                    transition_picker=dataset.transition_picker,
                    logger=logger,
                )

            # save metrics
            metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        logger.close()

    def create_impl(self, observation_shape: Shape, action_size: int) -> None:
        """Instantiate implementation objects with the dataset shapes.

        This method will be used internally when `fit` method is called.

        Args:
            observation_shape: observation shape.
            action_size: dimension of action-space.

        """
        if self._impl:
            LOG.warn("Parameters will be reinitialized.")
        self._create_impl(observation_shape, action_size)

    def _create_impl(self, observation_shape: Shape, action_size: int) -> None:
        raise NotImplementedError

    def build_with_dataset(self, dataset: ReplayBuffer) -> None:
        """Instantiate implementation object with MDPDataset object.

        Args:
            dataset: dataset.

        """
        dataset_info = DatasetInfo.from_episodes(dataset.episodes)
        observation_shape = dataset.sample_transition().observation_shape
        self.create_impl(observation_shape, dataset_info.action_size)

    def build_with_env(self, env: gym.Env) -> None:
        """Instantiate implementation object with OpenAI Gym object.

        Args:
            env: gym-like environment.

        """
        observation_shape = env.observation_space.shape
        if isinstance(env.action_space, Discrete):
            action_size = cast(int, env.action_space.n)
        else:
            action_size = cast(int, env.action_space.shape[0])
        self.create_impl(observation_shape, action_size)

    def update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: mini-batch data.

        Returns:
            dictionary of metrics.

        """
        loss = self._update(batch)
        self._grad_step += 1
        return loss

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        raise NotImplementedError

    def get_action_type(self) -> ActionSpace:
        """Returns action type (continuous or discrete).

        Returns:
            action type.

        """
        raise NotImplementedError

    @property
    def config(self) -> LearnableConfig:
        return self._config

    @property
    def batch_size(self) -> int:
        """Batch size to train.

        Returns:
            int: batch size.

        """
        return self._config.batch_size

    @property
    def gamma(self) -> float:
        """Discount factor.

        Returns:
            float: discount factor.

        """
        return self._config.gamma

    @property
    def observation_scaler(self) -> Optional[ObservationScaler]:
        """Preprocessing observation scaler.

        Returns:
            Optional[ObservationScaler]: preprocessing observation scaler.

        """
        return self._config.observation_scaler

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        """Preprocessing action scaler.

        Returns:
            Optional[ActionScaler]: preprocessing action scaler.

        """
        return self._config.action_scaler

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        """Preprocessing reward scaler.

        Returns:
            Optional[RewardScaler]: preprocessing reward scaler.

        """
        return self._config.reward_scaler

    @property
    def impl(self) -> Optional[ImplBase]:
        """Implementation object.

        Returns:
            Optional[ImplBase]: implementation object.

        """
        return self._impl

    @property
    def observation_shape(self) -> Optional[Shape]:
        """Observation shape.

        Returns:
            Optional[Sequence[int]]: observation shape.

        """
        if self._impl:
            return self._impl.observation_shape
        return None

    @property
    def action_size(self) -> Optional[int]:
        """Action size.

        Returns:
            Optional[int]: action size.

        """
        if self._impl:
            return self._impl.action_size
        return None

    @property
    def grad_step(self) -> int:
        """Total gradient step counter.

        This value will keep counting after ``fit`` and ``fit_online``
        methods finish.

        Returns:
            total gradient step counter.

        """
        return self._grad_step

    def set_grad_step(self, grad_step: int) -> None:
        """Set total gradient step counter.

        This method can be used to restart the middle of training with an
        arbitrary gradient step counter, which has effects on periodic
        functions such as the target update.

        Args:
            grad_step: total gradient step counter.

        """
        self._grad_step = grad_step
