import dataclasses
from abc import abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Generic, Optional, TypeVar, Union

import gym
import numpy as np
import torch
from tqdm.auto import tqdm
from typing_extensions import Self

from ...base import ImplBase, LearnableBase, LearnableConfig, save_config
from ...constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ...dataset import (
    DatasetInfo,
    Observation,
    ReplayBuffer,
    TrajectoryMiniBatch,
)
from ...logger import LOG, D3RLPyLogger
from ...metrics import evaluate_transformer_with_environment
from ...torch_utility import TorchTrajectoryMiniBatch
from ..utility import (
    assert_action_space_with_dataset,
    build_scalers_with_dataset,
)
from .inputs import TorchTransformerInput, TransformerInput

__all__ = [
    "TransformerAlgoImplBase",
    "StatefulTransformerWrapper",
    "TransformerConfig",
    "TransformerAlgoBase",
]


class TransformerAlgoImplBase(ImplBase):
    @abstractmethod
    def predict(self, inpt: TorchTransformerInput) -> torch.Tensor:
        ...


@dataclasses.dataclass()
class TransformerConfig(LearnableConfig):
    context_size: int = 20


TTransformerImpl = TypeVar("TTransformerImpl", bound=TransformerAlgoImplBase)
TTransformerConfig = TypeVar("TTransformerConfig", bound=TransformerConfig)


class StatefulTransformerWrapper(Generic[TTransformerImpl, TTransformerConfig]):
    _algo: "TransformerAlgoBase[TTransformerImpl, TTransformerConfig]"
    _target_return: float
    _return_rest: float
    _observations: Deque[Observation]
    _actions: Deque[Union[np.ndarray, int]]
    _rewards: Deque[float]
    _returns_to_go: Deque[float]
    _timesteps: Deque[int]
    _timestep: int

    def __init__(
        self,
        algo: "TransformerAlgoBase[TTransformerImpl, TTransformerConfig]",
        target_return: float,
    ):
        assert algo.impl, "algo must be built."
        self._algo = algo
        self._target_return = target_return
        self._return_rest = target_return

        context_size = algo.config.context_size
        self._observations = deque([], maxlen=context_size)
        self._actions = deque([self._get_pad_action()], maxlen=context_size)
        self._rewards = deque([], maxlen=context_size)
        self._returns_to_go = deque([], maxlen=context_size)
        self._timesteps = deque([], maxlen=context_size)
        self._timestep = 0

    def predict(self, x: Observation, reward: float) -> Union[np.ndarray, int]:
        self._observations.append(x)
        self._rewards.append(reward)
        self._returns_to_go.append(self._return_rest - reward)
        self._timesteps.append(self._timestep)
        inpt = TransformerInput(
            observations=np.array(self._observations),
            actions=np.array(self._actions),
            rewards=np.array(self._rewards).reshape((-1, 1)),
            returns_to_go=np.array(self._returns_to_go).reshape((-1, 1)),
            timesteps=np.array(self._timesteps),
        )
        action = self._algo.predict(inpt)
        self._actions[-1] = action
        self._actions.append(self._get_pad_action())
        self._timestep += 1
        return action

    def reset(self) -> None:
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()
        self._returns_to_go.clear()
        self._timesteps.clear()
        self._actions.append(self._get_pad_action())
        self._timestep = 0
        self._return_rest = self._target_return

    @property
    def algo(
        self,
    ) -> "TransformerAlgoBase[TTransformerImpl, TTransformerConfig]":
        return self._algo

    def _get_pad_action(self) -> Union[int, np.ndarray]:
        assert self._algo.impl
        if self._algo.get_action_type() == ActionSpace.CONTINUOUS:
            pad_action = np.zeros(self._algo.impl.action_size, dtype=np.float32)
        else:
            pad_action = 0
        return pad_action


class TransformerAlgoBase(
    Generic[TTransformerImpl, TTransformerConfig],
    LearnableBase[TTransformerImpl, TTransformerConfig],
):
    def predict(self, inpt: TransformerInput) -> np.ndarray:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        with torch.no_grad():
            torch_inpt = TorchTransformerInput.from_numpy(
                inpt=inpt,
                context_size=self._config.context_size,
                device=self._device,
                observation_scaler=self._config.observation_scaler,
                action_scaler=self._config.action_scaler,
                reward_scaler=self._config.reward_scaler,
            )
            action = self._impl.predict(torch_inpt)

            if self._config.action_scaler:
                action = self._config.action_scaler.reverse_transform(action)

        return action.cpu().detach().numpy()

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
        eval_env: Optional[gym.Env[Any, Any]] = None,
        eval_target_return: Optional[float] = None,
        save_interval: int = 1,
        callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> None:
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
            eval_env: evaluation environment.
            eval_target_return: evaluation return target.
            save_interval: interval to save parameters.
            callback: callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.

        Returns:
            iterator yielding current epoch and metrics dict.

        """
        dataset_info = DatasetInfo.from_episodes(dataset.episodes)
        LOG.info("dataset info", dataset_info=dataset_info)

        # check action space
        assert_action_space_with_dataset(self, dataset_info)

        # initialize scalers
        build_scalers_with_dataset(self, dataset)

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

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            action_size = dataset_info.action_size
            observation_shape = (
                dataset.sample_transition().observation_signature.shape[0]
            )
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
                        batch = dataset.sample_trajectory_batch(
                            self._config.batch_size,
                            length=self._config.context_size,
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

            if eval_env:
                assert eval_target_return is not None
                eval_score = evaluate_transformer_with_environment(
                    algo=self.as_stateful_wrapper(eval_target_return),
                    env=eval_env,
                )
                logger.add_metric("environment", eval_score)

            # save metrics
            logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

        logger.close()

    def update(self, batch: TrajectoryMiniBatch) -> Dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: mini-batch data.

        Returns:
            dictionary of metrics.

        """
        torch_batch = TorchTrajectoryMiniBatch.from_batch(
            batch=batch,
            device=self._device,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
        )
        loss = self.inner_update(torch_batch)
        self._grad_step += 1
        return loss

    @abstractmethod
    def inner_update(self, batch: TorchTrajectoryMiniBatch) -> Dict[str, float]:
        """Update parameters with PyTorch mini-batch.

        Args:
            batch: PyTorch mini-batch data.

        Returns:
            dictionary of metrics.

        """
        raise NotImplementedError

    def as_stateful_wrapper(
        self, target_return: float
    ) -> StatefulTransformerWrapper[TTransformerImpl, TTransformerConfig]:
        """Returns a wrapped Transformer algorithm for stateful decision making.

        Args:
            target_return: target environment return.

        Returns:
            StatefulTransformerWrapper object.

        """
        return StatefulTransformerWrapper(self, target_return)
