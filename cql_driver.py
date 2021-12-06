from d3rlpy.datasets import get_pybullet
from d3rlpy.algos import DiscreteCQL
from d3rlpy.ope import DiscreteFQE
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer
from d3rlpy.metrics.scorer import true_q_scorer
from sklearn.model_selection import train_test_split
from d3rlpy.ope import FQE
import d3rlpy
import argparse

def main(inputs):
    # prepare dataset
    dataset, env = d3rlpy.datasets.get_pybullet('hopper-bullet-mixed-v0')

    # prepare algorithm
    cql = d3rlpy.algos.CQL(use_gpu=True)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    # start training
    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=inputs.epochs_cql,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'td_error': d3rlpy.metrics.td_error_scorer,
                'trueQ': true_q_scorer},
            experiment_name=f'CQL_{args.log_name}')

    fqe = FQE(algo = cql,
            use_gpu = True)

    fqe.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs = inputs.epochs_fqe,
            scorers={
                'estimated_q_values': initial_state_value_estimation_scorer,
                'soft_opc': soft_opc_scorer(500),
                'trueQ': true_q_scorer},
            with_timestamp=False,
            experiment_name = f'FQE_run{inputs.log_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs_cql', type=int, default=10)
    parser.add_argument('--epochs_fqe', type=int, default=10)
    parser.add_argument('--log_name', type=str, default='placeholder')
    inputs = parser.parse_args()
    main(inputs)
