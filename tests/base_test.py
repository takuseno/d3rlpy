import numpy as np
import os

from unittest.mock import Mock
from d3rlpy.dataset import MDPDataset, Transition, TransitionMiniBatch
from d3rlpy.logger import D3RLPyLogger


def base_tester(model, impl, observation_shape, action_size=2):
    # dummy impl object
    model.impl = impl

    # check save  model
    impl.save_model = Mock()
    model.save_model('model.pt')
    impl.save_model.assert_called_with('model.pt')

    # check load model
    impl.load_model = Mock()
    model.load_model('mock.pt')
    impl.load_model.assert_called_with('mock.pt')

    # check get_params
    params = model.get_params(deep=False)
    clone = model.__class__(**params)
    for key, val in clone.get_params(deep=False).items():
        assert params[key] is val

    # check deep flag
    deep_params = model.get_params(deep=True)
    assert deep_params['impl'] is not impl

    # check set_params
    clone = model.__class__()
    for key, val in params.items():
        if np.isscalar(val) and not isinstance(val, str):
            params[key] = val + np.random.random()
    # set_params returns itself
    assert clone.set_params(**params) is clone
    for key, val in clone.get_params(deep=False).items():
        assert params[key] is val

    # check fit
    update_backup = model.update
    model.update = Mock(return_value=range(len(model._get_loss_labels())))
    n_episodes = 4
    episode_length = 25
    n_batch = 32
    n_epochs = 3
    data_size = n_episodes * episode_length
    model.batch_size = n_batch
    model.n_epochs = n_epochs
    observations = np.random.random((data_size, ) + observation_shape)
    actions = np.random.random((data_size, action_size))
    rewards = np.random.random(data_size)
    terminals = np.zeros(data_size)
    for i in range(n_episodes):
        terminals[(i + 1) * episode_length - 1] = 1.0
    dataset = MDPDataset(observations, actions, rewards, terminals)

    model.fit(dataset.episodes,
              logdir='test_data',
              verbose=False,
              show_progress=False,
              tensorboard=False)

    # check if the correct number of iterations are performed
    assert len(model.update.call_args_list) == data_size // n_batch * n_epochs

    # check arguments at each iteration
    for i, call in enumerate(model.update.call_args_list):
        epoch = i // (data_size // n_batch)
        total_step = i
        assert call[0][0] == epoch
        assert call[0][1] == total_step
        assert isinstance(call[0][2], TransitionMiniBatch)
        assert len(call[0][2]) == n_batch

    # save params.json
    logger = D3RLPyLogger('test',
                          root_dir='test_data',
                          verbose=False,
                          tensorboard=False)
    # save parameters to test_data/test/params.json
    model._save_params(logger)
    # load params.json
    json_path = os.path.join(logger.logdir, 'params.json')
    new_model = model.__class__.from_json(json_path)
    assert new_model.impl is not None
    assert new_model.impl.observation_shape == observation_shape
    assert new_model.impl.action_size == action_size
    assert type(model.scaler) == type(new_model.scaler)

    # set backed up methods
    model.impl = None
    model.update = update_backup

    return dataset


def base_update_tester(model, observation_shape, action_size, discrete=False):
    # make mini-batch
    transitions = []
    for _ in range(model.batch_size):
        observation = np.random.random(observation_shape)
        next_observation = np.random.random(observation_shape)
        reward = np.random.random()
        next_reward = np.random.random()
        terminal = np.random.randint(2)
        returns = np.random.random(100)
        consequent_observations = np.random.random((100, *observation_shape))
        if discrete:
            action = np.random.randint(action_size)
            next_action = np.random.randint(action_size)
        else:
            action = np.random.random(action_size)
            next_action = np.random.random(action_size)
        transition = Transition(observation_shape, action_size, observation,
                                action, reward, next_observation, next_action,
                                next_reward, terminal, returns,
                                consequent_observations)
        transitions.append(transition)

    batch = TransitionMiniBatch(transitions)

    # check if update runs without errors
    model.create_impl(observation_shape, action_size)
    loss = model.update(0, 0, batch)

    assert len(loss) == len(model._get_loss_labels())

    return transitions
