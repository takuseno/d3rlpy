import os
import json

from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import create_algo
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import value_estimation_std_scorer
from d3rlpy.metrics.scorer import continuous_action_diff_scorer
from d3rlpy.metrics.scorer import discrete_action_match_scorer
from d3rlpy.metrics.scorer import NEGATED_SCORER
from d3rlpy.metrics.comparer import compare_continuous_action_diff
from d3rlpy.metrics.comparer import compare_discrete_action_match
from sklearn.model_selection import train_test_split


def _get_scorers(discrete_action):
    scorers = {}
    scorers['td_error'] = td_error_scorer
    scorers['advantage'] = discounted_sum_of_advantage_scorer
    scorers['average_value'] = average_value_estimation_scorer
    scorers['value_std'] = value_estimation_std_scorer
    if discrete_action:
        scorers['action_match'] = discrete_action_match_scorer
    else:
        scorers['action_diff'] = continuous_action_diff_scorer
    return scorers


def _evaluate(algo, episodes, discrete_action):
    # evaluate
    scores = {}
    for k, scorer in _get_scorers(discrete_action).items():
        score = scorer(algo, episodes)
        if scorer in NEGATED_SCORER:
            score *= -1
        scores[k] = score
    return scores


def _compare(algo, base_algo, episodes, discrete_action):
    if discrete_action:
        compare = compare_discrete_action_match(base_algo)
    else:
        compare = compare_continuous_action_diff(base_algo)
    return compare(algo, episodes)


def train(algo_name,
          params,
          dataset_path,
          model_save_path,
          experiment_name=None,
          with_timestamp=True,
          logdir='d3rlpy_logs',
          prev_model_path=None,
          test_size=0.2):
    dataset = MDPDataset.load(dataset_path)
    train_data, test_data = train_test_split(dataset, test_size=test_size)

    # train
    algo = create_algo(algo_name, dataset.is_action_discrete(), **params)
    algo.fit(train_data,
             experiment_name=experiment_name,
             with_timestamp=with_timestamp,
             logdir=logdir,
             save_interval=1000000)  # never save models for now

    # save final model
    algo.save_model(model_save_path)

    # evaluate
    scores = _evaluate(algo, test_data, dataset.is_action_discrete())

    # compare previous model
    if prev_model_path:
        base_algo = create_algo(algo_name, **params)
        base_algo.load_model(prev_model_path)
        score = _comapre(algo, base_algo, test_data,
                         dataset.is_action_discrete())
        scores['algo_action_diff'] = score

    return scores
