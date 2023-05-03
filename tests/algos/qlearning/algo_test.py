import os
from unittest.mock import Mock

import numpy as np
import onnxruntime as ort
import torch

from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import (
    EpisodeGenerator,
    Transition,
    TransitionMiniBatch,
    create_infinite_replay_buffer,
)
from tests.base_test import from_json_tester, load_learnable_tester


def algo_tester(
    algo,
    observation_shape,
    action_size=2,
    deterministic_best_action=True,
    test_predict_value=True,
    test_policy_copy=True,
    test_q_function_copy=True,
    test_policy_optim_copy=True,
    test_q_function_optim_copy=True,
    test_from_json=True,
):
    fit_tester(algo, observation_shape, action_size)
    if test_from_json:
        from_json_tester(algo, observation_shape, action_size)
        load_learnable_tester(algo, observation_shape, action_size)
    predict_tester(algo, observation_shape, action_size)
    sample_action_tester(algo, observation_shape, action_size)
    save_and_load_tester(algo, observation_shape, action_size)
    update_tester(
        algo,
        observation_shape,
        action_size,
        test_policy_optim_copy=test_policy_optim_copy,
        test_q_function_optim_copy=test_q_function_optim_copy,
    )
    save_policy_tester(
        algo, deterministic_best_action, observation_shape, action_size
    )
    if test_predict_value:
        predict_value_tester(algo, observation_shape, action_size)
    if test_policy_copy:
        policy_copy_tester(algo, observation_shape, action_size)
    if test_q_function_copy:
        q_function_copy_tester(algo, observation_shape, action_size)


def fit_tester(algo, observation_shape, action_size):
    update_backup = algo.update
    algo.update = Mock(return_value={"loss": np.random.random()})

    n_episodes = 4
    episode_length = 25
    n_batch = algo.config.batch_size
    n_steps = 10
    n_steps_per_epoch = 5
    n_epochs = n_steps // n_steps_per_epoch
    data_size = n_episodes * episode_length
    shape = (data_size,) + observation_shape

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
    results = algo.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        logdir="test_data",
        verbose=False,
        show_progress=False,
    )

    assert isinstance(results, list)
    assert len(results) == n_epochs

    # check if the correct number of iterations are performed
    assert len(algo.update.call_args_list) == n_steps

    # check arguments at each iteration
    epoch = 0
    for i, call in enumerate(algo.update.call_args_list):
        epoch = i // n_steps_per_epoch
        assert isinstance(call[0][0], TransitionMiniBatch)
        assert len(call[0][0]) == n_batch

    # check fitter
    fitter = algo.fitter(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        logdir="test_data",
        verbose=False,
        show_progress=False,
    )

    for epoch, metrics in fitter:
        assert isinstance(epoch, int)
        assert isinstance(metrics, dict)

    assert epoch == n_epochs

    # set backed up methods
    algo.update = update_backup


def predict_tester(algo, observation_shape, action_size):
    algo.create_impl(observation_shape, action_size)
    x = np.random.random((100, *observation_shape))
    y = algo.predict(x)
    if algo.get_action_type() == ActionSpace.DISCRETE:
        assert y.shape == (100,)
    else:
        assert y.shape == (100, action_size)


def sample_action_tester(algo, observation_shape, action_size):
    algo.create_impl(observation_shape, action_size)
    x = np.random.random((100, *observation_shape))
    y = algo.sample_action(x)
    if algo.get_action_type() == ActionSpace.DISCRETE:
        assert y.shape == (100,)
    else:
        assert y.shape == (100, action_size)


def policy_copy_tester(algo, observation_shape, action_size):
    algo.create_impl(observation_shape, action_size)
    algo2 = algo.config.create()
    algo2.create_impl(observation_shape, action_size)
    x = np.random.random((100, *observation_shape))

    action1 = algo.predict(x)
    action2 = algo2.predict(x)
    assert not np.all(action1 == action2)

    algo2.copy_policy_from(algo)
    action1 = algo.predict(x)
    action2 = algo2.predict(x)
    assert np.all(action1 == action2)


def q_function_copy_tester(algo, observation_shape, action_size):
    algo.create_impl(observation_shape, action_size)
    algo2 = algo.config.create()
    algo2.create_impl(observation_shape, action_size)
    x = np.random.random((100, *observation_shape))
    if algo.get_action_type() == ActionSpace.DISCRETE:
        action = np.random.randint(action_size, size=(100,))
    else:
        action = np.random.random((100, action_size))

    value1 = algo.predict_value(x, action)
    value2 = algo2.predict_value(x, action)
    assert not np.all(value1 == value2)

    algo2.copy_q_function_from(algo)
    value1 = algo.predict_value(x, action)
    value2 = algo2.predict_value(x, action)
    assert np.all(value1 == value2)


def predict_value_tester(algo, observation_shape, action_size):
    algo.create_impl(observation_shape, action_size)
    x = np.random.random((100, *observation_shape))
    if algo.get_action_type() == ActionSpace.DISCRETE:
        action = np.random.randint(action_size, size=(100,))
    else:
        action = np.random.random((100, action_size))
    value = algo.predict_value(x, action)
    assert value.shape == (100,)


def save_and_load_tester(algo, observation_shape, action_size):
    algo.create_impl(observation_shape, action_size)
    algo.save_model(os.path.join("test_data", "model.pt"))
    algo.load_model(os.path.join("test_data", "model.pt"))


def update_tester(
    algo,
    observation_shape,
    action_size,
    test_policy_optim_copy=True,
    test_q_function_optim_copy=True,
):
    # make mini-batch
    transitions = []
    for i in range(algo.config.batch_size):
        if len(observation_shape) == 3:
            observation = np.random.randint(
                256, size=observation_shape, dtype=np.uint8
            )
            next_observation = np.random.randint(
                256, size=observation_shape, dtype=np.uint8
            )
        else:
            observation = np.random.random(observation_shape).astype("f4")
            next_observation = np.random.random(observation_shape).astype("f4")
        reward = np.random.random((1,))
        terminal = np.random.randint(2)
        if algo.get_action_type() == ActionSpace.DISCRETE:
            action = np.random.randint(action_size, size=(1,))
        else:
            action = np.random.random(action_size).astype("f4")

        transition = Transition(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            interval=1,
        )
        transitions.append(transition)

    batch = TransitionMiniBatch.from_transitions(transitions)

    # build models
    algo.create_impl(observation_shape, action_size)

    # check if update runs without errors
    grad_step = algo.grad_step
    loss = algo.update(batch)
    assert algo.grad_step == grad_step + 1

    algo.set_grad_step(0)
    assert algo.grad_step == 0

    assert len(loss.items()) > 0

    if test_q_function_optim_copy:
        algo2 = algo.config.create()
        algo2.create_impl(observation_shape, action_size)
        assert not algo2.impl.q_function_optim.state
        algo2.copy_q_function_optim_from(algo)
        assert algo2.impl.q_function_optim.state

    if test_policy_optim_copy:
        algo2 = algo.config.create()
        algo2.create_impl(observation_shape, action_size)
        assert not algo2.impl.policy_optim.state
        algo2.copy_policy_optim_from(algo)
        assert algo2.impl.policy_optim.state


def save_policy_tester(
    algo, deterministic_best_action, observation_shape, action_size
):
    algo.create_impl(observation_shape, action_size)

    # check save_policy as TorchScript
    algo.save_policy(os.path.join("test_data", "model.pt"))
    policy = torch.jit.load(os.path.join("test_data", "model.pt"))
    observations = torch.rand(100, *observation_shape)
    action = policy(observations)
    if algo.get_action_type() == ActionSpace.DISCRETE:
        assert action.shape == (100,)
    else:
        assert action.shape == (100, action_size)

    # check output consistency between the full model and TorchScript
    # TODO: check probablistic policy
    # https://github.com/pytorch/pytorch/pull/25753
    if deterministic_best_action:
        action = action.detach().numpy()
        assert np.allclose(
            action, algo.predict(observations.numpy()), atol=1e-6
        )

    # check save_policy as ONNX
    algo.save_policy(os.path.join("test_data", "model.onnx"))
    ort_session = ort.InferenceSession(os.path.join("test_data", "model.onnx"))
    observations = np.random.rand(1, *observation_shape).astype("f4")
    action = ort_session.run(None, {"input_0": observations})[0]
    if algo.get_action_type() == ActionSpace.DISCRETE:
        assert action.shape == (1,)
    else:
        assert action.shape == (1, action_size)

    # check output consistency between the full model and ONNX
    # TODO: check probablistic policy
    # https://github.com/pytorch/pytorch/pull/25753
    if deterministic_best_action:
        assert np.allclose(action, algo.predict(observations), atol=1e-6)
