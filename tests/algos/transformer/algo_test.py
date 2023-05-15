import os
from typing import Sequence
from unittest.mock import Mock

import numpy as np

from d3rlpy.algos import (
    TransformerAlgoBase,
    TransformerAlgoImplBase,
    TransformerConfig,
    TransformerInput,
)
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import (
    EpisodeGenerator,
    PartialTrajectory,
    TrajectoryMiniBatch,
    create_infinite_replay_buffer,
)
from tests.base_test import from_json_tester, load_learnable_tester


def algo_tester(
    algo: TransformerAlgoBase[TransformerAlgoImplBase, TransformerConfig],
    observation_shape: Sequence[int],
    action_size: int = 2,
) -> None:
    fit_tester(algo, observation_shape, action_size)
    from_json_tester(algo, observation_shape, action_size)
    load_learnable_tester(algo, observation_shape, action_size)
    predict_tester(algo, observation_shape, action_size)
    save_and_load_tester(algo, observation_shape, action_size)
    update_tester(algo, observation_shape, action_size)
    stateful_wrapper_tester(algo, observation_shape, action_size)


def fit_tester(
    algo: TransformerAlgoBase[TransformerAlgoImplBase, TransformerConfig],
    observation_shape: Sequence[int],
    action_size: int,
) -> None:
    update_backup = algo.update
    algo.update = Mock(return_value={"loss": np.random.random()})  # type: ignore

    n_episodes = 4
    episode_length = 25
    n_batch = algo.config.batch_size
    n_steps = 10
    n_steps_per_epoch = 5
    data_size = n_episodes * episode_length
    shape = (data_size, *observation_shape)

    if len(observation_shape) == 3:
        observations = np.random.randint(256, size=shape, dtype=np.uint8)
    else:
        observations = np.random.random(shape).astype("f4")
    if algo.get_action_type() == ActionSpace.CONTINUOUS:
        actions = np.random.random((data_size, action_size))
    else:
        actions = np.random.randint(action_size, size=(data_size, 1))
    rewards = np.random.random(data_size)
    terminals = np.zeros(data_size)
    for i in range(n_episodes):
        terminals[(i + 1) * episode_length - 1] = 1.0
    dataset = create_infinite_replay_buffer(
        EpisodeGenerator(observations, actions, rewards, terminals)()
    )

    # check fit
    algo.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        logdir="test_data",
        verbose=False,
        show_progress=False,
    )

    # check if the correct number of iterations are performed
    assert len(algo.update.call_args_list) == n_steps

    # check arguments at each iteration
    for i, call in enumerate(algo.update.call_args_list):
        assert isinstance(call[0][0], TrajectoryMiniBatch)
        assert len(call[0][0]) == n_batch

    # set backed up methods
    algo.update = update_backup  # type: ignore


def predict_tester(
    algo: TransformerAlgoBase[TransformerAlgoImplBase, TransformerConfig],
    observation_shape: Sequence[int],
    action_size: int,
) -> None:
    algo.create_impl(observation_shape, action_size)
    context_size = algo.config.context_size
    if algo.get_action_type() == ActionSpace.DISCRETE:
        actions = np.random.randint(action_size, size=(context_size,))
    else:
        actions = np.random.random((context_size, action_size))
    inpt = TransformerInput(
        observations=np.random.random((context_size, *observation_shape)),
        actions=actions,
        rewards=np.random.random((context_size, 1)),
        returns_to_go=np.random.random((context_size, 1)),
        timesteps=np.arange(context_size),
    )
    y = algo.predict(inpt)
    if algo.get_action_type() == ActionSpace.DISCRETE:
        assert y.ndim == 0
    else:
        assert y.shape == (action_size,)


def save_and_load_tester(
    algo: TransformerAlgoBase[TransformerAlgoImplBase, TransformerConfig],
    observation_shape: Sequence[int],
    action_size: int,
) -> None:
    algo.create_impl(observation_shape, action_size)
    algo.save_model(os.path.join("test_data", "model.pt"))
    algo.load_model(os.path.join("test_data", "model.pt"))


def update_tester(
    algo: TransformerAlgoBase[TransformerAlgoImplBase, TransformerConfig],
    observation_shape: Sequence[int],
    action_size: int,
) -> None:
    context_size = algo.config.context_size
    # make mini-batch
    trajectories = []
    for _ in range(algo.config.batch_size):
        if len(observation_shape) == 3:
            observations = np.random.randint(
                256, size=(context_size, *observation_shape), dtype=np.uint8
            )
        else:
            observations = np.random.random(
                (context_size, *observation_shape)
            ).astype("f4")
        rewards = np.random.random((context_size, 1))
        terminals = np.random.random((context_size, 1))
        if algo.get_action_type() == ActionSpace.DISCRETE:
            actions = np.random.randint(action_size, size=(context_size, 1))
        else:
            actions = np.random.random((context_size, action_size)).astype("f4")

        trajectory = PartialTrajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            returns_to_go=rewards,
            timesteps=np.arange(context_size),
            masks=np.zeros(context_size),
            length=context_size,
        )
        trajectories.append(trajectory)

    batch = TrajectoryMiniBatch.from_partial_trajectories(trajectories)

    # build models
    algo.create_impl(observation_shape, action_size)

    # check if update runs without errors
    grad_step = algo.grad_step
    loss = algo.update(batch)
    assert algo.grad_step == grad_step + 1

    algo.set_grad_step(0)
    assert algo.grad_step == 0

    assert len(loss.items()) > 0


def stateful_wrapper_tester(
    algo: TransformerAlgoBase[TransformerAlgoImplBase, TransformerConfig],
    observation_shape: Sequence[int],
    action_size: int,
) -> None:
    algo.create_impl(observation_shape, action_size)
    wrapper = algo.as_stateful_wrapper(100.0)

    # check predict
    for _ in range(10):
        observation, reward = np.random.random(observation_shape), 0.0
        action = wrapper.predict(observation, reward)
        if algo.get_action_type() == ActionSpace.DISCRETE:
            assert isinstance(action, int)
        else:
            assert isinstance(action, np.ndarray)
            assert action.shape == (action_size,)
    wrapper.reset()

    # check reset
    observation1, reward1 = np.random.random(observation_shape), 0.0
    action1 = wrapper.predict(observation1, reward1)
    observation, reward = np.random.random(observation_shape), 0.0
    action2 = wrapper.predict(observation, reward)
    assert np.all(action1 != action2)
    wrapper.reset()
    action3 = wrapper.predict(observation1, reward1)
    assert np.all(action1 == action3)
