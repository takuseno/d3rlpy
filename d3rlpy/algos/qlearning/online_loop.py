from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np
from tqdm.auto import tqdm

from ...dataset import ReplayBufferBase
from ...misc import IncrementalCounter
from ...types import GymEnv, NDArray
from ...logging import D3RLPyLogger
from ...interface import QLearningAlgoProtocol
from .explorers import Explorer


__all__ = ["OnlineLoop", "SimultaneousOnlineLoop"]


def _sample_action(
    env: GymEnv,
    observation: NDArray,
    algo: QLearningAlgoProtocol,
    explorer: Optional[Explorer],
    current_step: int,
    random_steps: int,
    logger: D3RLPyLogger,
) -> NDArray:
    with logger.measure_time("inference"):
        if current_step < random_steps:
            action = env.action_space.sample()
        elif explorer:
            x = observation.reshape((1,) + observation.shape)
            action = explorer.sample(algo, x, current_step)[0]
        else:
            action = algo.sample_action(
                np.expand_dims(observation, axis=0)
            )[0]
    return action


def _step_environment(env: GymEnv, observation: NDArray, action: NDArray, buffer: ReplayBufferBase, logger: D3RLPyLogger) -> NDArray:
    # step environment
    with logger.measure_time("environment_step"):
        (
            next_observation,
            reward,
            terminal,
            truncated,
            _,
        ) = env.step(action)

    # store observation
    buffer.append(observation, action, float(reward))

    # reset if terminated
    clip_episode = terminal or truncated
    if clip_episode:
        buffer.clip_episode(terminal)
        observation, _ = env.reset()
    else:
        observation = next_observation

    return observation


def _update(algo: QLearningAlgoProtocol, buffer: ReplayBufferBase, logger: D3RLPyLogger) -> None:
    # sample mini-batch
    with logger.measure_time("sample_batch"):
        batch = buffer.sample_transition_batch(algo.batch_size)

    # update parameters
    with logger.measure_time("algorithm_update"):
        loss = algo.update(batch)

    # record metrics
    for name, val in loss.items():
        logger.add_metric(name, val)


class OnlineLoop(ABC):
    @abstractmethod
    def rollout_one_epoch(
        self,
        n_steps_per_epoch: int,
        last_observation: NDArray,
        algo: QLearningAlgoProtocol,
        buffer: ReplayBufferBase,
        env: GymEnv,
        explorer: Optional[Explorer],
        total_step: IncrementalCounter,
        max_steps: int,
        update_start_step: int,
        random_steps: int,
        logger: D3RLPyLogger,
        show_progress: bool,
        callback: Optional[Callable[[QLearningAlgoProtocol, int, int], None]],
    ) -> None:
        raise NotImplementedError


class SimultaneousOnlineLoop(OnlineLoop):
    _update_interval: int

    def __init__(self, update_interval: int = 1) -> None:
        self._update_interval = update_interval

    def rollout_one_epoch(
        self,
        n_steps_per_epoch: int,
        last_observation: NDArray,
        algo: QLearningAlgoProtocol,
        buffer: ReplayBufferBase,
        env: GymEnv,
        explorer: Optional[Explorer],
        total_step: IncrementalCounter,
        max_steps: int,
        update_start_step: int,
        random_steps: int,
        logger: D3RLPyLogger,
        show_progress: bool,
        callback: Optional[Callable[[QLearningAlgoProtocol, int, int], None]],
    ) -> NDArray:
        current_epoch = total_step.get_value() // n_steps_per_epoch + 1
        max_epochs = max_steps // n_steps_per_epoch
        range_gen = tqdm(
            range(n_steps_per_epoch),
            disable=not show_progress,
            desc=f"Epoch {current_epoch}/{max_epochs}",
        )
        observation = last_observation
        for _ in range_gen:
            if total_step.get_value() >= max_steps:
                break

            current_step = total_step.increment()

            with logger.measure_time("step"):
                # sample exploration action
                action = _sample_action(
                    env=env,
                    observation=observation,
                    algo=algo,
                    explorer=explorer,
                    current_step=current_step,
                    random_steps=random_steps,
                    logger=logger,
                )

                # step environment
                observation = _step_environment(
                    env=env,
                    observation=observation,
                    action=action,
                    buffer=buffer,
                    logger=logger,
                )

                if buffer.transition_count > 0 and current_step > update_start_step:
                    if current_step % self._update_interval == 0:
                        _update(algo, buffer, logger)

                range_gen.set_postfix({"steps": f"{total_step.get_value()}/{max_steps}"})

            if callback:
                callback(algo, current_epoch, current_step)

        # return latest observation for continuitiy
        return observation


class SequentialOnlineLoop(OnlineLoop):
    def rollout_one_epoch(
        self,
        n_steps_per_epoch: int,
        last_observation: NDArray,
        algo: QLearningAlgoProtocol,
        buffer: ReplayBufferBase,
        env: GymEnv,
        explorer: Optional[Explorer],
        total_step: IncrementalCounter,
        max_steps: int,
        update_start_step: int,
        random_steps: int,
        logger: D3RLPyLogger,
        show_progress: bool,
        callback: Optional[Callable[[QLearningAlgoProtocol, int, int], None]],
    ) -> NDArray:
        current_epoch = total_step.get_value() // n_steps_per_epoch + 1
        max_epochs = max_steps // n_steps_per_epoch

        # rollout episodes
        range_gen = tqdm(
            range(n_steps_per_epoch),
            disable=not show_progress,
            desc=f"Epoch {current_epoch}/{max_epochs}: Collecting data.",
        )
        observation = last_observation
        for i in range_gen:
            current_step = total_step.increment()

            if total_step.get_value() >= max_steps:
                break

            # sample exploration action
            action = _sample_action(
                env=env,
                observation=observation,
                algo=algo,
                explorer=explorer,
                current_step=current_step,
                random_steps=random_steps,
                logger=logger,
            )

            # step environment
            observation = _step_environment(
                env=env,
                observation=observation,
                action=action,
                buffer=buffer,
                logger=logger,
            )

            range_gen.set_postfix({"steps": f"{i}/{n_steps_per_epoch}"})

        current_step = total_step.get_value()

        range_gen = tqdm(
            range(n_steps_per_epoch),
            disable=not show_progress,
            desc=f"Epoch {current_epoch}/{max_epochs}: Updating models.",
        )
        for _ in range_gen:
            if buffer.transition_count > 0 and current_step > update_start_step:
                if current_step % self._update_interval == 0:
                    _update(algo, buffer, logger)

            range_gen.set_postfix({"steps": f"{total_step.get_value()}/{max_steps}"})

            if callback:
                callback(algo, current_epoch, current_step)

        # return latest observation for continuitiy
        return observation
