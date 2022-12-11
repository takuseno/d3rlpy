import os
from unittest.mock import Mock

import numpy as np

from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import (
    DatasetInfo,
    EpisodeGenerator,
    Transition,
    TransitionMiniBatch,
    create_infinite_replay_buffer,
)
from d3rlpy.logger import D3RLPyLogger


def base_tester(
    model, impl, observation_shape, action_size=2, skip_from_json=False
):
    # dummy impl object
    model._impl = impl

    # check save  model
    impl.save_model = Mock()
    model.save_model("model.pt")
    impl.save_model.assert_called_with("model.pt")

    # check load model
    impl.load_model = Mock()
    model.load_model("mock.pt")
    impl.load_model.assert_called_with("mock.pt")

    # check fit and fitter
    update_backup = model.update
    model.update = Mock(return_value={"loss": np.random.random()})
    n_episodes = 4
    episode_length = 25
    n_batch = model.config.batch_size
    n_steps = 10
    n_steps_per_epoch = 5
    n_epochs = n_steps // n_steps_per_epoch
    data_size = n_episodes * episode_length
    shape = (data_size,) + observation_shape
    if len(observation_shape) == 3:
        observations = np.random.randint(256, size=shape, dtype=np.uint8)
    else:
        observations = np.random.random(shape).astype("f4")
    if model.get_action_type() == ActionSpace.CONTINUOUS:
        actions = np.random.random((data_size, action_size))
    else:
        actions = np.random.randint(action_size, size=(data_size, 1))
    rewards = np.random.random(data_size)
    terminals = np.zeros(data_size)
    for i in range(n_episodes):
        terminals[(i + 1) * episode_length - 1] = 1.0
    dataset = create_infinite_replay_buffer(
        EpisodeGenerator(observations, actions, rewards, terminals)
    )
    dataset_info = DatasetInfo.from_episodes(dataset.episodes)

    # check fit
    results = model.fit(
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
    assert len(model.update.call_args_list) == n_steps

    # check arguments at each iteration
    for i, call in enumerate(model.update.call_args_list):
        epoch = i // n_steps_per_epoch
        assert isinstance(call[0][0], TransitionMiniBatch)
        assert len(call[0][0]) == n_batch

    # check fitter
    fitter = model.fitter(
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

    # save params.json
    logger = D3RLPyLogger("test", root_dir="test_data", verbose=False)
    # save parameters to test_data/test/params.json
    model.save_params(logger)
    # load params.json
    if not skip_from_json:
        json_path = os.path.join(logger.logdir, "params.json")
        new_model = model.__class__.from_json(json_path)
        assert new_model.impl is not None
        assert type(new_model) == type(model)
        assert tuple(new_model.impl.observation_shape) == observation_shape
        assert new_model.impl.action_size == action_size
        assert type(model.observation_scaler) == type(
            new_model.observation_scaler
        )

    # check builds
    model._impl = None
    model.build_with_dataset(dataset)
    assert model.impl.observation_shape == dataset_info.observation_shape
    assert model.impl.action_size == dataset_info.action_size

    # set backed up methods
    model._impl = None
    model.update = update_backup

    return dataset


def base_update_tester(model, observation_shape, action_size, discrete=False):
    # make mini-batch
    transitions = []
    for i in range(model.batch_size):
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
        if discrete:
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
    model.create_impl(observation_shape, action_size)

    # check if update runs without errors
    grad_step = model.grad_step
    loss = model.update(batch)
    assert model.grad_step == grad_step + 1

    model.set_grad_step(0)
    assert model.grad_step == 0

    assert len(loss.items()) > 0

    return transitions
