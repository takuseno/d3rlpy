import numpy as np
import os
import torch
import pickle
import gym

from unittest.mock import Mock
from skbrl.algos.base import ImplBase
from skbrl.dataset import MDPDataset, TransitionMiniBatch


def algo_tester(algo):
    # dummy impl object
    impl = ImplBase()

    algo.impl = impl

    # check save  model
    impl.save_model = Mock()
    algo.save_model('model.pt')
    impl.save_model.assert_called_with('model.pt')

    # check load model
    impl.load_model = Mock()
    algo.load_model('mock.pt')
    impl.load_model.assert_called_with('mock.pt')

    # check save policy
    impl.save_policy = Mock()
    algo.save_policy('policy.pt')
    impl.save_policy.assert_called_with('policy.pt')

    # check get_params
    params = algo.get_params(deep=False)
    clone = algo.__class__(**params)
    for key, val in clone.get_params(deep=False).items():
        assert params[key] is val

    deep_params = algo.get_params(deep=True)
    assert deep_params['impl'] is not impl

    # check predict
    x = np.random.random((2, 3)).tolist()
    ref_y = np.random.random((2, 3)).tolist()
    impl.predict_best_action = Mock(return_value=ref_y)
    y = algo.predict(x)
    assert y == ref_y
    impl.predict_best_action.assert_called_with(x)

    # check predict_value
    action = np.random.random((2, 3)).tolist()
    ref_value = np.random.random((2, 3)).tolist()
    impl.predict_value = Mock(return_value=ref_value)
    value = algo.predict_value(x, action)
    assert value == ref_value
    impl.predict_value.assert_called_with(x, action)

    # check fit
    n_episodes = 4
    episode_length = 25
    n_batch = 32
    n_epochs = 3
    data_size = n_episodes * episode_length
    algo.update = Mock()
    algo.batch_size = n_batch
    algo.n_epochs = n_epochs
    observations = np.random.random((data_size, 3))
    actions = np.random.random((data_size, 3))
    rewards = np.random.random(data_size)
    terminals = np.zeros(data_size)
    for i in range(n_episodes):
        terminals[(i + 1) * episode_length - 1] = 1.0
    dataset = MDPDataset(observations, actions, rewards, terminals)
    algo.fit(dataset.episodes)

    # check if the correct number of iterations are performed
    assert len(algo.update.call_args_list) == data_size // n_batch * n_epochs

    # check arguments at each iteration
    for i, call in enumerate(algo.update.call_args_list):
        epoch = i // (data_size // n_batch)
        itr = i % (data_size // n_batch)
        assert call[0][0] == epoch
        assert call[0][1] == itr
        assert isinstance(call[0][2], TransitionMiniBatch)
        assert len(call[0][2]) == n_batch


def algo_cartpole_tester(algo, n_evaluations=100, n_episodes=100, n_trials=3):
    # load dataset
    with open('skbrl_data/cartpole.pkl', 'rb') as f:
        observations, actions, rewards, terminals = pickle.load(f)

    dataset = MDPDataset(observations, actions, rewards, terminals, True)

    # try multiple trials to reduce failures due to random seeds
    trial_count = 0
    for _ in range(n_trials):
        # reset parameters
        algo.impl = None

        # train
        algo.fit(dataset.episodes[:n_episodes])

        # environment
        env = gym.make('CartPole-v0')

        # evaluation loop
        success_count = 0
        evaluation_count = 0
        while evaluation_count < n_evaluations:
            observation = env.reset()
            episode_rew = 0.0
            while True:
                action = algo.predict([observation])[0]
                observation, reward, done, _ = env.step(action)
                episode_rew += reward
                if done:
                    break
            evaluation_count += 1
            if episode_rew >= 160:
                success_count += 1

        if success_count >= n_evaluations * 0.8:
            break

        trial_count += 1
        if trial_count == n_trials:
            assert False, 'performance is not good enough.'


def algo_pendulum_tester(algo, n_evaluations=100, n_episodes=100, n_trials=3):
    # load dataset
    with open('skbrl_data/pendulum.pkl', 'rb') as f:
        observations, actions, rewards, terminals = pickle.load(f)

    dataset = MDPDataset(observations, actions, rewards, terminals)

    # try multiple trials to reduce failures due to random seeds
    trial_count = 0
    for _ in range(n_trials):
        # reset parameters
        algo.impl = None

        # train
        algo.fit(dataset.episodes[:n_episodes])

        # environment
        env = gym.make('Pendulum-v0')
        upper_bound = env.action_space.high

        # evaluation loop
        success_count = 0
        evaluation_count = 0
        while evaluation_count < n_evaluations:
            observation = env.reset()
            episode_rew = 0.0
            while True:
                action = algo.predict([observation])[0]
                observation, reward, done, _ = env.step(upper_bound * action)
                episode_rew += reward
                if done:
                    break
            evaluation_count += 1
            if episode_rew >= -400:
                success_count += 1

        if success_count >= n_evaluations * 0.8:
            break

        trial_count += 1
        if trial_count == n_trials:
            assert False, 'performance is not good enough.'


def impl_tester(impl, discrete):
    observations = np.random.random((100, ) + impl.observation_shape)
    if discrete:
        actions = np.random.randint(impl.action_size, size=100)
    else:
        actions = np.random.random((100, impl.action_size))

    # check predict_best_action
    y = impl.predict_best_action(observations)
    if discrete:
        assert y.shape == (100, )
    else:
        assert y.shape == (100, impl.action_size)

    # check predict_values
    value = impl.predict_value(observations, actions)
    assert value.shape == (100, )


def torch_impl_tester(impl, discrete):
    impl_tester(impl, discrete)

    # check save_model and load_model
    impl.save_model(os.path.join('test_data', 'model.pt'))
    impl.load_model(os.path.join('test_data', 'model.pt'))

    # check save_policy
    impl.save_policy(os.path.join('test_data', 'model.pt'))
    policy = torch.jit.load(os.path.join('test_data', 'model.pt'))
    observations = torch.rand(100, *impl.observation_shape)
    action = policy(observations)
    if discrete:
        assert action.shape == (100, )
    else:
        assert action.shape == (100, impl.action_size)
