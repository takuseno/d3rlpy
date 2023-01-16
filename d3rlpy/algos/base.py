from abc import abstractmethod
from collections import defaultdict
from typing import Callable, Dict, Generator, List, Optional, Tuple, cast

import gym
import numpy as np
import torch
from tqdm.auto import tqdm, trange

from ..base import ImplBase, LearnableBase, save_config
from ..constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    IMPL_NOT_INITIALIZED_ERROR,
    ActionSpace,
)
from ..dataset import (
    DatasetInfo,
    Episode,
    Observation,
    ReplayBuffer,
    Shape,
    TransitionMiniBatch,
    check_non_1d_array,
    create_fifo_replay_buffer,
    is_tuple_shape,
)
from ..logger import LOG, D3RLPyLogger
from ..metrics import EvaluatorProtocol, evaluate_with_environment
from ..models.torch import EnsembleQFunction, Policy
from ..torch_utility import (
    TorchMiniBatch,
    convert_to_torch,
    convert_to_torch_recursively,
    eval_api,
    freeze,
    get_state_dict,
    hard_sync,
    map_location,
    reset_optimizer_states,
    set_state_dict,
    sync_optimizer_state,
    to_cpu,
    to_cuda,
    unfreeze,
)
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
    _observation_shape: Shape
    _action_size: int
    _device: str

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        device: str,
    ):
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._device = device

    @eval_api
    def predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    @abstractmethod
    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @eval_api
    def sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_sample_action(x)

    @abstractmethod
    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @eval_api
    def predict_value(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.inner_predict_value(x, action)

    @abstractmethod
    def inner_predict_value(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        pass

    def to_gpu(self, device: str) -> None:
        self._device = device
        to_cuda(self, self._device)

    def to_cpu(self) -> None:
        self._device = "cpu:0"
        to_cpu(self)

    def save_model(self, fname: str) -> None:
        torch.save(get_state_dict(self), fname)

    def load_model(self, fname: str) -> None:
        chkpt = torch.load(fname, map_location=map_location(self._device))
        set_state_dict(self, chkpt)

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
    def policy_optim(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def copy_policy_optim_from(self, impl: "AlgoImplBase") -> None:
        if not isinstance(impl.policy_optim, type(self.policy_optim)):
            raise ValueError(
                "Invalid policy optimizer type: "
                f"expected={type(self.policy_optim)},"
                f"actual={type(impl.policy_optim)}"
            )
        sync_optimizer_state(self.policy_optim, impl.policy_optim)

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

    @property
    def q_function_optim(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def copy_q_function_optim_from(self, impl: "AlgoImplBase") -> None:
        if not isinstance(impl.q_function_optim, type(self.q_function_optim)):
            raise ValueError(
                "Invalid Q-function optimizer type: "
                f"expected={type(self.q_function_optim)}",
                f"actual={type(impl.q_function_optim)}",
            )
        sync_optimizer_state(self.q_function_optim, impl.q_function_optim)

    def reset_optimizer_states(self) -> None:
        reset_optimizer_states(self)

    @property
    def observation_shape(self) -> Shape:
        return self._observation_shape

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def device(self) -> str:
        return self._device


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

        if is_tuple_shape(self._impl.observation_shape):
            dummy_x = [
                torch.rand(1, *shape, device=self._device)
                for shape in self._impl.observation_shape
            ]
        else:
            dummy_x = torch.rand(
                1, *self._impl.observation_shape, device=self._device
            )

        # workaround until version 1.6
        freeze(self._impl)

        # dummy function to select best actions
        def _func(x: torch.Tensor) -> torch.Tensor:
            assert self._impl

            if self._config.observation_scaler:
                x = self._config.observation_scaler.transform(x)

            action = self._impl.predict_best_action(x)

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
                input_names=["input_0"],
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
        unfreeze(self._impl)

    def predict(self, x: Observation) -> np.ndarray:
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
        assert check_non_1d_array(x), "Input must have batch dimension."

        # TODO: support tuple inputs
        torch_x = cast(
            torch.Tensor, convert_to_torch_recursively(x, self._device)
        )

        with torch.no_grad():
            if self._config.observation_scaler:
                torch_x = self._config.observation_scaler.transform(torch_x)

            action = self._impl.predict_best_action(torch_x)

            if self._config.action_scaler:
                action = self._config.action_scaler.reverse_transform(action)

        return action.cpu().detach().numpy()

    def predict_value(self, x: Observation, action: np.ndarray) -> np.ndarray:
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
            x: observations
            action: actions

        Returns:
            predicted action-values

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert check_non_1d_array(x), "Input must have batch dimension."

        # TODO: support tuple inputs
        torch_x = cast(
            torch.Tensor, convert_to_torch_recursively(x, self._device)
        )

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

        return value.cpu().detach().numpy()

    def sample_action(self, x: Observation) -> np.ndarray:
        """Returns sampled actions.

        The sampled actions are identical to the output of `predict` method if
        the policy is deterministic.

        Args:
            x: observations.

        Returns:
            sampled actions.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        assert check_non_1d_array(x), "Input must have batch dimension."

        # TODO: support tuple inputs
        torch_x = cast(
            torch.Tensor, convert_to_torch_recursively(x, self._device)
        )

        with torch.no_grad():
            if self._config.observation_scaler:
                torch_x = self._config.observation_scaler.transform(torch_x)

            action = self._impl.sample_action(torch_x)

            # transform action back to the original range
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
        eval_episodes: Optional[List[Episode]] = None,
        eval_env: Optional[gym.Env] = None,
        save_interval: int = 1,
        evaluators: Optional[Dict[str, EvaluatorProtocol]] = None,
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
            eval_env: evaluation environment.
            save_interval: interval to save parameters.
            evaluators: list of evaluators used with `eval_episodes`.
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
                eval_env,
                save_interval,
                evaluators,
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
        eval_env: Optional[gym.Env] = None,
        save_interval: int = 1,
        evaluators: Optional[Dict[str, EvaluatorProtocol]] = None,
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
            eval_env: evaluation environment.
            save_interval: interval to save parameters.
            evaluators: list of evaluators used with `eval_episodes`.
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

            if evaluators and eval_episodes:
                for name, evaluator in evaluators.items():
                    test_score = evaluator(
                        algo=self,
                        episodes=eval_episodes,
                        transition_picker=dataset.transition_picker,
                    )
                    logger.add_metric(name, test_score)

            if eval_env:
                eval_score = evaluate_with_environment(self, eval_env)
                logger.add_metric("environment", eval_score)

            # save metrics
            metrics = logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

            yield epoch, metrics

        logger.close()

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
                        action = self.sample_action(
                            np.expand_dims(observation, axis=0)
                        )[0]

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
                if eval_env:
                    eval_score = evaluate_with_environment(
                        self, eval_env, epsilon=eval_epsilon
                    )
                    logger.add_metric("evaluation", eval_score)

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

    def update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: mini-batch data.

        Returns:
            dictionary of metrics.

        """
        torch_batch = TorchMiniBatch.from_batch(
            batch=batch,
            device=self._device,
            observation_scaler=self._config.observation_scaler,
            action_scaler=self._config.action_scaler,
            reward_scaler=self._config.reward_scaler,
        )
        loss = self.inner_update(torch_batch)
        self._grad_step += 1
        return loss

    def inner_update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        """Update parameters with PyTorch mini-batch.

        Args:
            batch: PyTorch mini-batch data.

        Returns:
            dictionary of metrics.

        """
        raise NotImplementedError

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
