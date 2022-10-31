import d3rlpy
from sklearn.model_selection import train_test_split
from pdb import set_trace

# prepare dataset
dataset, env = d3rlpy.datasets.get_atari('breakout-expert-v0')


# split dataset
train_episodes, test_episodes = train_test_split(dataset, test_size=0.1)

# TODO: Undo the following
train_episodes = train_episodes[0:50]
test_episodes = test_episodes[0:5]

# prepare algorithm
cql = d3rlpy.algos.DiscreteCQL(
    n_frames=4,
    q_func_factory='qr',
    scaler='pixel',
    use_gpu=False,
)

# start training
cql.fit(
    train_episodes,
    eval_episodes=test_episodes,
    n_epochs=100,
    scorers={
        'environment': d3rlpy.metrics.evaluate_on_environment(env),
        'td_error': d3rlpy.metrics.td_error_scorer,
    },
)
