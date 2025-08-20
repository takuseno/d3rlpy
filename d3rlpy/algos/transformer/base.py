import dataclasses
from abc import abstractmethod
from collections import defaultdict, deque
from typing import (
    Callable,
    Dict,
    Generic,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
import torch
from tqdm.auto import tqdm
from typing_extensions import Self

from ...base import ImplBase, LearnableBase, LearnableConfig, save_config
from ...constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace, LoggingStrategy
from ...dataset import ReplayBuffer, TrajectoryMiniBatch, is_tuple_shape
from ...logging import (
    LOG,
    D3RLPyLogger,
    FileAdapterFactory,
    LoggerAdapterFactory,
)
from ...metrics import evaluate_transformer_with_environment, EvaluatorProtocol
from ...torch_utility import TorchTrajectoryMiniBatch, eval_api, train_api
from ...types import GymEnv, NDArray, Observation, TorchObservation, Float32NDArray
from ..utility import (
    assert_action_space_with_dataset,
    build_scalers_with_trajectory_slicer,
)
from .action_samplers import (
    IdentityTransformerActionSampler,
    SoftmaxTransformerActionSampler,
    TransformerActionSampler,
)
from .inputs import TorchTransformerInput, TransformerInput

__all__ = [
    "TransformerAlgoImplBase",
    "StatefulTransformerWrapper",
    "TransformerConfig",
    "TransformerAlgoBase",
]


class TransformerAlgoImplBase(ImplBase):
    @eval_api
    def predict(self, inpt: TorchTransformerInput) -> torch.Tensor:
        return self.inner_predict(inpt)

    @abstractmethod
    def inner_predict(self, inpt: TorchTransformerInput) -> torch.Tensor:
        raise NotImplementedError

    @train_api
    def update(
        self, batch: TorchTrajectoryMiniBatch, grad_step: int
    ) -> dict[str, float]:
        return self.inner_update(batch, grad_step)

    @abstractmethod
    def inner_update(
        self, batch: TorchTrajectoryMiniBatch, grad_step: int
    ) -> dict[str, float]:
        raise NotImplementedError


@dataclasses.dataclass()
class TransformerConfig(LearnableConfig):
    context_size: int = 20
    max_timestep: int = 1000


TTransformerImpl = TypeVar("TTransformerImpl", bound=TransformerAlgoImplBase)
TTransformerConfig = TypeVar("TTransformerConfig", bound=TransformerConfig)


class StatefulTransformerWrapper(Generic[TTransformerImpl, TTransformerConfig]):
    r"""A stateful wrapper for inference of Transformer-based algorithms.

    This wrapper class provides a similar interface of Q-learning-based
    algoritms, which is especially useful when you evaluate Transformer-based
    algorithms such as Decision Transformer.

    .. code-block:: python

        from d3rlpy.algos import DecisionTransformerConfig
        from d3rlpy.algos import StatefulTransformerWrapper

        dt = DecisionTransformerConfig().create()
        dt.create_impl(<observation_shape>, <action_size>)
        # initialize wrapper with a target return of 1000
        wrapper = StatefulTransformerWrapper(dt, target_return=1000)
        # shortcut is also available
        wrapper = dt.as_stateful_wrapper(target_return=1000)

        # predict next action to achieve the return of 1000 in the end
        action = wrapper.predict(<observation>, <reward>)

        # clear stateful information
        wrapper.reset()

    Args:
        algo (TransformerAlgoBase): Transformer-based algorithm.
        target_return (float): Target return.
        action_sampler (d3rlpy.algos.TransformerActionSampler): Action sampler.
    """

    _algo: "TransformerAlgoBase[TTransformerImpl, TTransformerConfig]"
    _target_return: float
    _action_sampler: TransformerActionSampler
    _return_rest: float
    _observations: deque[Observation]
    _actions: deque[Union[NDArray, int]]
    _rewards: deque[float]
    _returns_to_go: deque[float]
    _timesteps: deque[int]
    _timestep: int

    def __init__(
        self,
        algo: "TransformerAlgoBase[TTransformerImpl, TTransformerConfig]",
        target_return: float,
        action_sampler: TransformerActionSampler,
    ):
        assert algo.impl, "algo must be built."
        self._algo = algo
        self._target_return = target_return
        self._action_sampler = action_sampler
        self._return_rest = target_return

        context_size = algo.config.context_size
        self._observations = deque([], maxlen=context_size)
        self._embeddings = deque([], maxlen=context_size)
        self._actions = deque([self._get_pad_action()], maxlen=context_size)
        self._rewards = deque([], maxlen=context_size)
        self._returns_to_go = deque([], maxlen=context_size)
        self._timesteps = deque([], maxlen=context_size)
        self._timestep = 1

    def predict(self, x: Observation, embedding: Float32NDArray, reward: float) -> Union[NDArray, int]:
        r"""Returns action.

        Args:
            x: Observation.
            reward: Last reward.

        Returns:
            Action.
        """
        self._observations.append(x)
        self._embeddings.append(embedding)
        self._rewards.append(reward)
        self._returns_to_go.append(self._return_rest - reward)
        self._timesteps.append(self._timestep)

        numpy_observations: Observation
        if isinstance(x, np.ndarray):
            numpy_observations = np.array(self._observations)
        else:
            numpy_observations = [
                np.array([o[i] for o in self._observations])
                for i in range(len(x))
            ]

        inpt = TransformerInput(
            observations=numpy_observations,
            actions=np.array(self._actions),
            rewards=np.array(self._rewards).reshape((-1, 1)),
            returns_to_go=np.array(self._returns_to_go).reshape((-1, 1)),
            timesteps=np.array(self._timesteps),
            embeddings=None if embedding is None else np.array(self._embeddings),
        )
        action = self._action_sampler(self._algo.predict(inpt))
        self._actions[-1] = action
        self._actions.append(self._get_pad_action())
        self._timestep = min(self._timestep + 1, self._algo.config.max_timestep)
        self._return_rest -= reward
        return action

    def reset(self) -> None:
        """Clears stateful information."""
        self._observations.clear()
        self._embeddings.clear()
        self._actions.clear()
        self._rewards.clear()
        self._returns_to_go.clear()
        self._timesteps.clear()
        self._actions.append(self._get_pad_action())
        self._timestep = 1
        self._return_rest = self._target_return

    @property
    def algo(
        self,
    ) -> "TransformerAlgoBase[TTransformerImpl, TTransformerConfig]":
        return self._algo

    def _get_pad_action(self) -> Union[int, NDArray]:
        assert self._algo.impl
        pad_action: Union[int, NDArray]
        if self._algo.get_action_type() == ActionSpace.CONTINUOUS:
            pad_action = np.zeros(self._algo.impl.action_size, dtype=np.float32)
        else:
            pad_action = 0
        return pad_action


class TransformerAlgoBase(
    Generic[TTransformerImpl, TTransformerConfig],
    LearnableBase[TTransformerImpl, TTransformerConfig],
):
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

        Visit https://d3rlpy.readthedocs.io/en/stable/tutorials/after_training_policies.html#export-policies-as-torchscript for the further usage.

        Args:
            fname: Destination file path.
        """  # noqa: E501
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        context_size = self._config.context_size
        dummy_x = []
        if is_tuple_shape(self._impl.observation_shape):
            dummy_x.extend(
                [
                    torch.rand(context_size, *shape, device=self._device)
                    for shape in self._impl.observation_shape
                ]
            )
            num_observations = len(self._impl.observation_shape)
        else:
            dummy_x.append(
                torch.rand(
                    context_size,
                    *self._impl.observation_shape,
                    device=self._device,
                )
            )
            num_observations = 1
        # action
        if self.get_action_type() == ActionSpace.CONTINUOUS:
            dummy_x.append(
                torch.rand(
                    context_size, self._impl.action_size, device=self._device
                )
            )
        else:
            dummy_x.append(torch.rand(context_size, 1, device=self._device))
        # return_to_go
        dummy_x.append(torch.rand(context_size, 1, device=self._device))
        # timesteps
        dummy_x.append(torch.arange(context_size, device=self._device))

        # workaround until version 1.6
        self._impl.modules.freeze()

        # local function to select best actions
        def _func(*x: Sequence[torch.Tensor]) -> torch.Tensor:
            assert self._impl

            # add batch dimension
            x = [v.view(1, *v.shape) for v in x]  # type: ignore

            observations: TorchObservation = x[:-3]
            actions = x[-3]
            returns_to_go = x[-2]
            timesteps = x[-1]

            if len(observations) == 1:
                observations = observations[0]

            if self._config.observation_scaler:
                observations = self._config.observation_scaler.transform(
                    observations
                )
            if self._config.action_scaler:
                actions = self._config.action_scaler.transform(actions)

            inpt = TorchTransformerInput(
                observations=observations,
                actions=actions,
                rewards=torch.zeros_like(returns_to_go),
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                masks=torch.zeros_like(returns_to_go),
                length=self._config.context_size,
            )

            action = self._impl.predict(inpt)

            if self._config.action_scaler:
                action = self._config.action_scaler.reverse_transform(action)

            if self.get_action_type() == ActionSpace.DISCRETE:
                action = action.argmax()

            return action

        traced_script = torch.jit.trace(_func, dummy_x, check_trace=False)

        if fname.endswith(".onnx"):
            # currently, PyTorch cannot directly export function as ONNX.
            torch.onnx.export(
                traced_script,
                dummy_x,
                fname,
                export_params=True,
                opset_version=11,
                input_names=[
                    f"observation_{i}" for i in range(num_observations)
                ]
                + ["action", "return_to_go", "timestep"],
                output_names=["output_0"],
            )
        elif fname.endswith(".pt"):
            traced_script.save(fname)
        else:
            raise ValueError(
                f"invalid format type: {fname}."
                " .pt and .onnx extensions are currently supported."
            )

        # workaround until version 1.6
        self._impl.modules.unfreeze()

    def predict(self, inpt: TransformerInput) -> NDArray:
        """Returns action.

        This is for internal use. For evaluation, use
        ``StatefulTransformerWrapper`` instead.

        Args:
            inpt: Sequence input.

        Returns:
            Action.
        """
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

        return action.cpu().detach().numpy()  # type: ignore

    def fit(
        self,
        dataset: ReplayBuffer,
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        logging_steps: int = 500,
        logging_strategy: LoggingStrategy = LoggingStrategy.EPOCH,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        eval_env: Optional[GymEnv] = None,
        eval_target_return: Optional[float] = None,
        eval_action_sampler: Optional[TransformerActionSampler] = None,
        save_interval: int = 1,
        evaluators: Optional[Dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> None:
        """Trains with given dataset.

        Args:
            dataset: Offline dataset to train.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            eval_env: Evaluation environment.
            eval_target_return: Evaluation return target.
            eval_action_sampler: Action sampler used in evaluation.
            save_interval: Interval to save parameters.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
        """
        LOG.info("dataset info", dataset_info=dataset.dataset_info)

        # check action space
        assert_action_space_with_dataset(self, dataset.dataset_info)

        # initialize scalers
        build_scalers_with_trajectory_slicer(self, dataset)

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            action_size = dataset.dataset_info.action_size
            observation_shape = (
                dataset.sample_transition().observation_signature.shape
            )
            if len(observation_shape) == 1:
                observation_shape = observation_shape[0]  # type: ignore
            self.create_impl(observation_shape, action_size)
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__
        self.logger = D3RLPyLogger(
            algo=self,
            adapter_factory=logger_adapter,
            experiment_name=experiment_name,
            n_steps_per_epoch=n_steps_per_epoch,
            with_timestamp=with_timestamp,
        )

        # save hyperparameters
        save_config(self, self.logger)

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
                with self.logger.measure_time("step"):
                    # pick transitions
                    with self.logger.measure_time("sample_batch"):
                        batch = dataset.sample_trajectory_batch(
                            self._config.batch_size,
                            length=self._config.context_size,
                        )

                    # update parameters
                    with self.logger.measure_time("algorithm_update"):
                        loss = self.update(batch)

                    # record metrics
                    for name, val in loss.items():
                        self.logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                if (
                    logging_strategy == LoggingStrategy.STEPS
                    and total_step % logging_steps == 0
                ):
                    metrics = self.logger.commit(epoch, total_step)

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # call epoch_callback if given
            if epoch_callback:
                epoch_callback(self, epoch, total_step)

            if evaluators:
                for name, evaluator in evaluators.items():
                    test_score = evaluator(self, dataset)
                    self.logger.add_metric(name, test_score)

            if eval_env:
                assert eval_target_return is not None
                eval_score = evaluate_transformer_with_environment(
                    algo=self.as_stateful_wrapper(
                        target_return=eval_target_return,
                        action_sampler=eval_action_sampler,
                    ),
                    env=eval_env,
                )
                self.logger.add_metric("environment", eval_score)

            # save metrics
            if logging_strategy == LoggingStrategy.EPOCH:
                self.logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                self.logger.save_model(total_step, self)

        self.logger.close()

    def update(self, batch: TrajectoryMiniBatch) -> dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: Mini-batch data.

        Returns:
            Dictionary of metrics.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        torch_batch = TorchTrajectoryMiniBatch.from_batch(
            batch=batch,
            device=self._device,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
        )

        if self._config.transform:
            torch_batch = self._config.transform(torch_batch)

        loss = self._impl.update(torch_batch, self._grad_step)
        self._grad_step += 1
        return loss

    def as_stateful_wrapper(
        self,
        target_return: float,
        action_sampler: Optional[TransformerActionSampler] = None,
    ) -> StatefulTransformerWrapper[TTransformerImpl, TTransformerConfig]:
        """Returns a wrapped Transformer algorithm for stateful decision making.

        Args:
            target_return: Target environment return.
            action_sampler: Action sampler.

        Returns:
            StatefulTransformerWrapper object.
        """
        if action_sampler is None:
            if self.get_action_type() == ActionSpace.CONTINUOUS:
                action_sampler = IdentityTransformerActionSampler()
            else:
                action_sampler = SoftmaxTransformerActionSampler()
        return StatefulTransformerWrapper(self, target_return, action_sampler)
