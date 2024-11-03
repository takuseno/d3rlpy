import os
from typing import Any
from unittest.mock import Mock

import numpy as np
import onnxruntime as ort
import torch

from d3rlpy.algos import (
    GreedyTransformerActionSampler,
    IdentityTransformerActionSampler,
    TransformerActionSampler,
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
    is_tuple_shape,
)
from d3rlpy.logging import NoopAdapterFactory
from d3rlpy.torch_utility import convert_to_numpy_recursively
from d3rlpy.types import Float32NDArray, NDArray, Shape

from ...base_test import from_json_tester, load_learnable_tester
from ...testing_utils import (
    create_observation,
    create_observations,
    create_torch_observations,
)


def algo_tester(
    algo: TransformerAlgoBase[TransformerAlgoImplBase, TransformerConfig],
    observation_shape: Shape,
    action_size: int = 2,
) -> None:
    fit_tester(algo, observation_shape, action_size)
    from_json_tester(algo, observation_shape, action_size)
    load_learnable_tester(algo, observation_shape, action_size)
    predict_tester(algo, observation_shape, action_size)
    save_and_load_tester(algo, observation_shape, action_size)
    update_tester(algo, observation_shape, action_size)
    stateful_wrapper_tester(algo, observation_shape, action_size)
    save_policy_tester(algo, observation_shape, action_size)


def fit_tester(
    algo: TransformerAlgoBase[TransformerAlgoImplBase, TransformerConfig],
    observation_shape: Shape,
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

    observations = create_observations(observation_shape, data_size)

    actions: NDArray
    if algo.get_action_type() == ActionSpace.CONTINUOUS:
        actions = np.random.random((data_size, action_size))
    else:
        actions = np.random.randint(action_size, size=(data_size, 1))

    rewards: Float32NDArray = np.random.random(data_size).astype(np.float32)
    terminals: Float32NDArray = np.zeros(data_size, dtype=np.float32)
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
        logger_adapter=NoopAdapterFactory(),
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
    observation_shape: Shape,
    action_size: int,
) -> None:
    algo.create_impl(observation_shape, action_size)
    context_size = algo.config.context_size

    actions: NDArray
    if algo.get_action_type() == ActionSpace.DISCRETE:
        actions = np.random.randint(action_size, size=(context_size,))
    else:
        actions = np.random.random((context_size, action_size))

    inpt = TransformerInput(
        observations=create_observations(observation_shape, context_size),
        actions=actions,
        rewards=np.random.random((context_size, 1)).astype(np.float32),
        returns_to_go=np.random.random((context_size, 1)).astype(np.float32),
        timesteps=np.arange(context_size),
    )
    y = algo.predict(inpt)
    if algo.get_action_type() == ActionSpace.DISCRETE:
        assert y.shape == (action_size,)
    else:
        assert y.shape == (action_size,)


def save_and_load_tester(
    algo: TransformerAlgoBase[TransformerAlgoImplBase, TransformerConfig],
    observation_shape: Shape,
    action_size: int,
) -> None:
    algo.create_impl(observation_shape, action_size)
    algo.save_model(os.path.join("test_data", "model.pt"))

    algo2 = algo.config.create()
    algo2.create_impl(observation_shape, action_size)
    algo2.load_model(os.path.join("test_data", "model.pt"))
    assert isinstance(algo2, TransformerAlgoBase)

    action_sampler: TransformerActionSampler
    if algo.get_action_type() == ActionSpace.DISCRETE:
        action_sampler = GreedyTransformerActionSampler()
    else:
        action_sampler = IdentityTransformerActionSampler()
    actor1 = algo.as_stateful_wrapper(0, action_sampler)
    actor2 = algo2.as_stateful_wrapper(0, action_sampler)

    observation = create_observation(observation_shape)
    action1 = actor1.predict(observation, 0)
    action2 = actor2.predict(observation, 0)
    assert np.all(action1 == action2)


def update_tester(
    algo: TransformerAlgoBase[TransformerAlgoImplBase, TransformerConfig],
    observation_shape: Shape,
    action_size: int,
) -> None:
    context_size = algo.config.context_size
    # make mini-batch
    trajectories = []
    for _ in range(algo.config.batch_size):
        observations = create_observations(observation_shape, context_size)
        rewards: Float32NDArray = np.random.random((context_size, 1)).astype(
            np.float32
        )
        if algo.get_action_type() == ActionSpace.DISCRETE:
            actions = np.random.randint(action_size, size=(context_size, 1))
        else:
            actions = np.random.random((context_size, action_size)).astype("f4")

        trajectory = PartialTrajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=np.random.random((context_size, 1)).astype(np.float32),
            returns_to_go=rewards,
            timesteps=np.arange(context_size),
            masks=np.zeros(context_size, dtype=np.float32),
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
    observation_shape: Shape,
    action_size: int,
) -> None:
    algo.create_impl(observation_shape, action_size)

    action_sampler: TransformerActionSampler
    if algo.get_action_type() == ActionSpace.DISCRETE:
        action_sampler = GreedyTransformerActionSampler()
    else:
        action_sampler = IdentityTransformerActionSampler()
    wrapper = algo.as_stateful_wrapper(100.0, action_sampler)

    # check predict
    for _ in range(10):
        observation, reward = create_observation(observation_shape), 0.0
        action = wrapper.predict(observation, reward)
        if algo.get_action_type() == ActionSpace.DISCRETE:
            assert isinstance(action, int)
        else:
            assert isinstance(action, np.ndarray)
            assert action.shape == (action_size,)
    wrapper.reset()

    # check reset
    observation1, reward1 = create_observation(observation_shape), 0.0
    action1 = wrapper.predict(observation1, reward1)
    observation, reward = create_observation(observation_shape), 0.0
    action2 = wrapper.predict(observation, reward)
    # in discrete case, there is high chance that action is the same.
    if algo.get_action_type() == ActionSpace.CONTINUOUS:
        assert np.all(action1 != action2)
    wrapper.reset()
    action3 = wrapper.predict(observation1, reward1)
    assert np.all(action1 == action3)


def save_policy_tester(
    algo: TransformerAlgoBase[TransformerAlgoImplBase, TransformerConfig],
    observation_shape: Shape,
    action_size: int,
) -> None:
    algo.create_impl(observation_shape, action_size)

    # check save_policy as TorchScript
    algo.save_policy(os.path.join("test_data", "model.pt"))
    policy = torch.jit.load(os.path.join("test_data", "model.pt"))

    inputs: list[Any] = []
    torch_observations = create_torch_observations(
        observation_shape, algo.config.context_size
    )
    if is_tuple_shape(observation_shape):
        inputs.extend(torch_observations)
        num_observations = len(torch_observations)
    else:
        inputs.append(torch_observations)
        num_observations = 1
    # action
    if algo.get_action_type() == ActionSpace.CONTINUOUS:
        inputs.append(torch.rand(algo.config.context_size, action_size))
    else:
        inputs.append(torch.rand(algo.config.context_size, 1))
    # return-to-go
    inputs.append(torch.rand(algo.config.context_size, 1))
    # timestep
    inputs.append(torch.arange(algo.config.context_size))

    # inference
    action = policy(*inputs)

    if algo.get_action_type() == ActionSpace.DISCRETE:
        assert action.shape == tuple()
    else:
        assert action.shape == (action_size,)

    action = action.detach().numpy()
    if num_observations > 1:
        observations = convert_to_numpy_recursively(inputs[:num_observations])
    else:
        observations = inputs[0].numpy()
    inpt = TransformerInput(
        observations=observations,
        actions=inputs[num_observations].numpy(),
        rewards=inputs[num_observations + 1].numpy(),
        returns_to_go=inputs[num_observations + 1].numpy(),
        timesteps=inputs[num_observations + 2].numpy(),
    )
    if algo.get_action_type() == ActionSpace.DISCRETE:
        assert action == algo.predict(inpt).argmax()
    else:
        assert np.allclose(action, algo.predict(inpt), atol=1e-3)

    # check save_policy as ONNX
    algo.save_policy(os.path.join("test_data", "model.onnx"))
    ort_session = ort.InferenceSession(
        os.path.join("test_data", "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    if num_observations > 1:
        input_dict = {f"observation_{i}": x for i, x in enumerate(observations)}
    else:
        input_dict = {"observation_0": observations}
    input_dict["action"] = inpt.actions
    input_dict["return_to_go"] = inpt.returns_to_go
    input_dict["timestep"] = inpt.timesteps
    action = ort_session.run(None, input_dict)[0]
    if algo.get_action_type() == ActionSpace.DISCRETE:
        assert action.shape == tuple()
    else:
        assert action.shape == (action_size,)

    if algo.get_action_type() == ActionSpace.DISCRETE:
        assert action == algo.predict(inpt).argmax()
    else:
        assert np.allclose(action, algo.predict(inpt), atol=1e-3)
