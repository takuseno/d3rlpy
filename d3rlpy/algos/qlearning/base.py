from abc import abstractmethod
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm, trange
from typing_extensions import Self

from ...base import ImplBase, LearnableBase, LearnableConfig, save_config
from ...constants import (
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
    LoggingStrategy,
)
from ...dataset import (
    ReplayBufferBase,
    TransitionMiniBatch,
    check_non_1d_array,
    create_fifo_replay_buffer,
    is_tuple_shape,
)
from ...logging import (
    LOG,
    D3RLPyLogger,
    FileAdapterFactory,
    LoggerAdapterFactory,
)
from ...metrics import EvaluatorProtocol, evaluate_qlearning_with_environment
from ...models.torch import Policy
from ...torch_utility import (
    TorchMiniBatch,
    convert_to_torch,
    convert_to_torch_recursively,
    eval_api,
    hard_sync,
    sync_optimizer_state,
    train_api,
)
from ...types import GymEnv, NDArray, Observation, TorchObservation
from ..utility import (
    assert_action_space_with_dataset,
    assert_action_space_with_env,
    build_scalers_with_env,
    build_scalers_with_transition_picker,
)
from .explorers import Explorer

__all__ = [
    "QLearningAlgoImplBase",
    "QLearningAlgoBase",
    "TQLearningImpl",
    "TQLearningConfig",
]


class QLearningAlgoImplBase(ImplBase):
    @train_api
    def update(self, batch: TorchMiniBatch, grad_step: int) -> Dict[str, float]:
        return self.inner_update(batch, grad_step)

    @abstractmethod
    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        pass

    @eval_api
    def predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    @abstractmethod
    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        pass

    @eval_api
    def sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_sample_action(x)

    @abstractmethod
    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        pass

    @eval_api
    def predict_value(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        return self.inner_predict_value(x, action)

    @abstractmethod
    def inner_predict_value(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        pass

    @property
    def policy(self) -> Policy:
        raise NotImplementedError

    def copy_policy_from(self, impl: "QLearningAlgoImplBase") -> None:
        if not isinstance(impl.policy, type(self.policy)):
            raise ValueError(
                f"Invalid policy type: expected={type(self.policy)},"
                f"actual={type(impl.policy)}"
            )
        hard_sync(self.policy, impl.policy)

    @property
    def policy_optim(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def copy_policy_optim_from(self, impl: "QLearningAlgoImplBase") -> None:
        if not isinstance(impl.policy_optim, type(self.policy_optim)):
            raise ValueError(
                "Invalid policy optimizer type: "
                f"expected={type(self.policy_optim)},"
                f"actual={type(impl.policy_optim)}"
            )
        sync_optimizer_state(self.policy_optim, impl.policy_optim)

    @property
    def q_function(self) -> nn.ModuleList:
        raise NotImplementedError

    def copy_q_function_from(self, impl: "QLearningAlgoImplBase") -> None:
        q_func = self.q_function[0]
        if not isinstance(impl.q_function[0], type(q_func)):
            raise ValueError(
                f"Invalid Q-function type: expected={type(q_func)},"
                f"actual={type(impl.q_function[0])}"
            )
        hard_sync(self.q_function, impl.q_function)

    @property
    def q_function_optim(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def copy_q_function_optim_from(self, impl: "QLearningAlgoImplBase") -> None:
        if not isinstance(impl.q_function_optim, type(self.q_function_optim)):
            raise ValueError(
                "Invalid Q-function optimizer type: "
                f"expected={type(self.q_function_optim)}",
                f"actual={type(impl.q_function_optim)}",
            )
        sync_optimizer_state(self.q_function_optim, impl.q_function_optim)

    def reset_optimizer_states(self) -> None:
        self.modules.reset_optimizer_states()


TQLearningImpl = TypeVar("TQLearningImpl", bound=QLearningAlgoImplBase)
TQLearningConfig = TypeVar("TQLearningConfig", bound=LearnableConfig)


class QLearningAlgoBase(
    Generic[TQLearningImpl, TQLearningConfig],
    LearnableBase[TQLearningImpl, TQLearningConfig],
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
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        if is_tuple_shape(self._impl.observation_shape):
            dummy_x = [
                torch.rand(1, *shape, device=self._device)
                for shape in self._impl.observation_shape
            ]
            num_inputs = len(self._impl.observation_shape)
        else:
            dummy_x = torch.rand(
                1, *self._impl.observation_shape, device=self._device
            )
            num_inputs = 1

        # workaround until version 1.6
        self._impl.modules.freeze()

        # local function to select best actions
        def _func(*x: Sequence[torch.Tensor]) -> torch.Tensor:
            assert self._impl

            observation: TorchObservation = x
            if len(observation) == 1:
                observation = observation[0]

            if self._config.observation_scaler:
                observation = self._config.observation_scaler.transform(
                    observation
                )

            action = self._impl.predict_best_action(observation)

            if self._config.action_scaler:
                action = self._config.action_scaler.reverse_transform(action)

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
                input_names=[f"input_{i}" for i in range(num_inputs)],
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

    def predict(self, x: Observation) -> NDArray:
        """Returns greedy actions.

        .. code-block:: python

            # 100 observations with shape of (10,)
            x = np.random.random((100, 10))

            actions = algo.predict(x)
            # actions.shape == (100, action size) for continuous control
            # actions.shape == (100,) for discrete control

        Args:
            x: Observations

        Returns:
            Greedy actions
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert check_non_1d_array(x), "Input must have batch dimension."

        torch_x = convert_to_torch_recursively(x, self._device)

        with torch.no_grad():
            if self._config.observation_scaler:
                torch_x = self._config.observation_scaler.transform(torch_x)

            action = self._impl.predict_best_action(torch_x)

            if self._config.action_scaler:
                action = self._config.action_scaler.reverse_transform(action)

        return action.cpu().detach().numpy()  # type: ignore

    def predict_value(self, x: Observation, action: NDArray) -> NDArray:
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

        Args:
            x: Observations
            action: Actions

        Returns:
            Predicted action-values
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert check_non_1d_array(x), "Input must have batch dimension."

        torch_x = convert_to_torch_recursively(x, self._device)

        torch_action = convert_to_torch(action, self._device)

        with torch.no_grad():
            if self._config.observation_scaler:
                torch_x = self._config.observation_scaler.transform(torch_x)

            if self.get_action_type() == ActionSpace.CONTINUOUS:
                if self._config.action_scaler:
                    torch_action = self._config.action_scaler.transform(
                        torch_action
                    )
            elif self.get_action_type() == ActionSpace.DISCRETE:
                torch_action = torch_action.long()
            else:
                raise ValueError("invalid action type")

            value = self._impl.predict_value(torch_x, torch_action)

        return value.cpu().detach().numpy()  # type: ignore

    def sample_action(self, x: Observation) -> NDArray:
        """Returns sampled actions.

        The sampled actions are identical to the output of `predict` method if
        the policy is deterministic.

        Args:
            x: Observations.

        Returns:
            Sampled actions.
        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert check_non_1d_array(x), "Input must have batch dimension."

        torch_x = convert_to_torch_recursively(x, self._device)

        with torch.no_grad():
            if self._config.observation_scaler:
                torch_x = self._config.observation_scaler.transform(torch_x)

            action = self._impl.sample_action(torch_x)

            # transform action back to the original range
            if self._config.action_scaler:
                action = self._config.action_scaler.reverse_transform(action)

        return action.cpu().detach().numpy()  # type: ignore

    def fit(
        self,
        dataset: ReplayBufferBase,
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logging_steps: int = 500,
        logging_strategy: LoggingStrategy = LoggingStrategy.EPOCH,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[Dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> List[Tuple[int, Dict[str, float]]]:
        """Trains with given dataset.

        .. code-block:: python

            algo.fit(episodes, n_steps=1000000)

        Args:
            dataset: ReplayBuffer object.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logging_steps: Number of steps to log metrics. This will be ignored
                if logging_strategy is EPOCH.
            logging_strategy: Logging strategy to use.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            save_interval: Interval to save parameters.
            evaluators: List of evaluators.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
            epoch_callback: Callable function that takes
                ``(algo, epoch, total_step)``, which is called at the end of
                every epoch.

        Returns:
            List of result tuples (epoch, metrics) per epoch.
        """
        results = list(
            self.fitter(
                dataset=dataset,
                n_steps=n_steps,
                n_steps_per_epoch=n_steps_per_epoch,
                experiment_name=experiment_name,
                with_timestamp=with_timestamp,
                logging_steps=logging_steps,
                logging_strategy=logging_strategy,
                logger_adapter=logger_adapter,
                show_progress=show_progress,
                save_interval=save_interval,
                evaluators=evaluators,
                callback=callback,
                epoch_callback=epoch_callback,
            )
        )
        return results

    def fitter(
        self,
        dataset: ReplayBufferBase,
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        logging_steps: int = 500,
        logging_strategy: LoggingStrategy = LoggingStrategy.EPOCH,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        save_interval: int = 1,
        evaluators: Optional[Dict[str, EvaluatorProtocol]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        epoch_callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
        iteration algo methods and properties can be changed or queried.

        .. code-block:: python

            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)

        Args:
            dataset: Offline dataset to train.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logging_steps: Number of steps to log metrics. This will be ignored
                if logging_strategy is EPOCH.
            logging_strategy: Logging strategy to use.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            save_interval: Interval to save parameters.
            evaluators: List of evaluators.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
            epoch_callback: Callable function that takes
                ``(algo, epoch, total_step)``, which is called at the end of
                every epoch.

        Returns:
            Iterator yielding current epoch and metrics dict.
        """
        LOG.info("dataset info", dataset_info=dataset.dataset_info)

        # check action space
        assert_action_space_with_dataset(self, dataset.dataset_info)

        # initialize scalers
        build_scalers_with_transition_picker(self, dataset)

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
                        batch = dataset.sample_transition_batch(
                            self._config.batch_size
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

            # save metrics
            if logging_strategy == LoggingStrategy.EPOCH:
                metrics = self.logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                self.logger.save_model(total_step, self)

            yield epoch, metrics

        self.logger.close()

    def fit_online(
        self,
        env: GymEnv,
        buffer: Optional[ReplayBufferBase] = None,
        explorer: Optional[Explorer] = None,
        n_steps: int = 1000000,
        n_steps_per_epoch: int = 10000,
        update_interval: int = 1,
        n_updates: int = 1,
        update_start_step: int = 0,
        random_steps: int = 0,
        eval_env: Optional[GymEnv] = None,
        eval_epsilon: float = 0.0,
        eval_n_trials: int = 10,
        save_interval: int = 1,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logging_steps: int = 500,
        logging_strategy: LoggingStrategy = LoggingStrategy.EPOCH,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        callback: Optional[Callable[[Self, int, int], None]] = None,
    ) -> None:
        """Start training loop of online deep reinforcement learning.

        Args:
            env: Gym-like environment.
            buffer : Replay buffer.
            explorer: Action explorer.
            n_steps: Number of total steps to train.
            n_steps_per_epoch: Number of steps per epoch.
            update_interval: Number of steps per update.
            n_updates: Number of gradient steps at a time. The combination of
                ``update_interval`` and ``n_updates`` controls Update-To-Data
                (UTD) ratio.
            update_start_step: Steps before starting updates.
            random_steps: Steps for the initial random explortion.
            eval_env: Gym-like environment. If None, evaluation is skipped.
            eval_epsilon: :math:`\\epsilon`-greedy factor during evaluation.
            save_interval: Number of epochs before saving models.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be ``{class name}_online_{timestamp}``.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logging_steps: Number of steps to log metrics. This will be ignored
                if logging_strategy is EPOCH.
            logging_strategy: Logging strategy to use.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called at the end of epochs.
        """

        # create default replay buffer
        if buffer is None:
            buffer = create_fifo_replay_buffer(1000000, env=env)

        # check action-space
        assert_action_space_with_env(self, env)

        # initialize algorithm parameters
        build_scalers_with_env(self, env)

        # setup algorithm
        if self.impl is None:
            LOG.debug("Building model...")
            self.build_with_env(env)
            LOG.debug("Model has been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__ + "_online"
        logger = D3RLPyLogger(
            algo=self,
            adapter_factory=logger_adapter,
            experiment_name=experiment_name,
            n_steps_per_epoch=n_steps_per_epoch,
            with_timestamp=with_timestamp,
        )

        # save hyperparameters
        save_config(self, logger)

        # switch based on show_progress flag
        xrange = trange if show_progress else range

        # start training loop
        observation, _ = env.reset()
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
                        action = self.sample_action(
                            np.expand_dims(observation, axis=0)
                        )[0]

                # step environment
                with logger.measure_time("environment_step"):
                    (
                        next_observation,
                        reward,
                        terminal,
                        truncated,
                        _,
                    ) = env.step(action)
                    rollout_return += float(reward)

                clip_episode = terminal or truncated

                # store observation
                buffer.append(observation, action, float(reward))

                # reset if terminated
                if clip_episode:
                    buffer.clip_episode(terminal)
                    observation, _ = env.reset()
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
                        for _ in range(n_updates):  # controls UTD ratio
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

                        if (
                            logging_strategy == LoggingStrategy.STEPS
                            and total_step % logging_steps == 0
                        ):
                            logger.commit(epoch, total_step)

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            if epoch > 0 and total_step % n_steps_per_epoch == 0:
                # evaluation
                if eval_env:
                    eval_score = evaluate_qlearning_with_environment(
                        self,
                        eval_env,
                        n_trials=eval_n_trials,
                        epsilon=eval_epsilon,
                    )
                    logger.add_metric("evaluation", eval_score)

                if epoch % save_interval == 0:
                    logger.save_model(total_step, self)

                # save metrics
                if logging_strategy == LoggingStrategy.EPOCH:
                    logger.commit(epoch, total_step)

        # clip the last episode
        buffer.clip_episode(False)

        # close logger
        logger.close()

    def collect(
        self,
        env: GymEnv,
        buffer: Optional[ReplayBufferBase] = None,
        explorer: Optional[Explorer] = None,
        deterministic: bool = False,
        n_steps: int = 1000000,
        show_progress: bool = True,
    ) -> ReplayBufferBase:
        """Collects data via interaction with environment.

        If ``buffer`` is not given, ``ReplayBuffer`` will be internally created.

        Args:
            env: Fym-like environment.
            buffer: Replay buffer.
            explorer: Action explorer.
            deterministic: Flag to collect data with the greedy policy.
            n_steps: Number of total steps to train.
            show_progress: Flag to show progress bar for iterations.

        Returns:
            Replay buffer with the collected data.
        """
        # create default replay buffer
        if buffer is None:
            buffer = create_fifo_replay_buffer(1000000, env=env)

        # check action-space
        assert_action_space_with_env(self, env)

        # initialize algorithm parameters
        build_scalers_with_env(self, env)

        # setup algorithm
        if self.impl is None:
            LOG.debug("Building model...")
            self.build_with_env(env)
            LOG.debug("Model has been built.")
        else:
            LOG.warning("Skip building models since they're already built.")

        # switch based on show_progress flag
        xrange = trange if show_progress else range

        # start training loop
        observation, _ = env.reset()
        for total_step in xrange(1, n_steps + 1):
            # sample exploration action
            if deterministic:
                action = self.predict(np.expand_dims(observation, axis=0))[0]
            else:
                if explorer:
                    x = observation.reshape((1,) + observation.shape)
                    action = explorer.sample(self, x, total_step)[0]
                else:
                    action = self.sample_action(
                        np.expand_dims(observation, axis=0)
                    )[0]

            # step environment
            next_observation, reward, terminal, truncated, _ = env.step(action)

            clip_episode = terminal or truncated

            # store observation
            buffer.append(observation, action, float(reward))

            # reset if terminated
            if clip_episode:
                buffer.clip_episode(terminal)
                observation, _ = env.reset()
            else:
                observation = next_observation

        # clip the last episode
        buffer.clip_episode(False)

        return buffer

    def update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: Mini-batch data.

        Returns:
            Dictionary of metrics.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        torch_batch = TorchMiniBatch.from_batch(
            batch=batch,
            gamma=self._config.gamma,
            compute_returns_to_go=self.need_returns_to_go,
            device=self._device,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
        )
        loss = self._impl.update(torch_batch, self._grad_step)
        self._grad_step += 1
        return loss

    @property
    def need_returns_to_go(self) -> bool:
        return False

    def copy_policy_from(
        self, algo: "QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]"
    ) -> None:
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
            algo: Algorithm object.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, QLearningAlgoImplBase)
        self._impl.copy_policy_from(algo.impl)

    def copy_policy_optim_from(
        self, algo: "QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]"
    ) -> None:
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
            algo: Algorithm object.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, QLearningAlgoImplBase)
        self._impl.copy_policy_optim_from(algo.impl)

    def copy_q_function_from(
        self, algo: "QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]"
    ) -> None:
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
            algo: Algorithm object.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, QLearningAlgoImplBase)
        self._impl.copy_q_function_from(algo.impl)

    def copy_q_function_optim_from(
        self, algo: "QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig]"
    ) -> None:
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
            algo: Algorithm object.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert isinstance(algo.impl, QLearningAlgoImplBase)
        self._impl.copy_q_function_optim_from(algo.impl)

    def reset_optimizer_states(self) -> None:
        """Resets optimizer states.

        This is especially useful when fine-tuning policies with setting inital
        optimizer states.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        self._impl.reset_optimizer_states()
