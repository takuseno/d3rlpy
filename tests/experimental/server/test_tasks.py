import pytest
import os

from d3rlpy.algos import DQN
from d3rlpy.datasets import get_cartpole
from d3rlpy.experimental.server.tasks import _evaluate, _compare
from d3rlpy.experimental.server.tasks import train


def test_evaluate():
    dataset, _ = get_cartpole()
    train_episodes = dataset.episodes[:10]
    test_episodes = dataset.episodes[-10:]

    algo = DQN(n_epochs=1)
    algo.fit(train_episodes, logdir='test_data')

    scores = _evaluate(algo, test_episodes, True)

    eval_keys = [
        'td_error', 'advantage', 'average_value', 'value_std', 'action_match'
    ]

    for key in eval_keys:
        assert key in scores


def test_compare():
    dataset, _ = get_cartpole()
    train_episodes = dataset.episodes[:10]
    test_episodes = dataset.episodes[-10:]

    algo = DQN(n_epochs=1)
    algo.fit(train_episodes, logdir='test_data')

    base_algo = DQN(n_epochs=1)
    base_algo.fit(train_episodes, logdir='test_data')

    score = _compare(algo, base_algo, test_episodes, True)


def test_train():
    dataset, _ = get_cartpole()
    dataset_path = os.path.join('test_data', 'worker_dataset.h5')
    dataset.dump(dataset_path)
    model_save_path = os.path.join('test_data', 'worker_model.h5')
    scores = train('dqn', {'n_epochs': 1},
                   dataset_path,
                   model_save_path,
                   experiment_name='task_test',
                   logdir='test_data')
    assert os.path.exists(model_save_path)
