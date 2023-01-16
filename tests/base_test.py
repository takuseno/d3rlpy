import os

from d3rlpy.base import save_config
from d3rlpy.logger import D3RLPyLogger


def from_json_tester(algo, observation_shape, action_size):
    algo.create_impl(observation_shape, action_size)
    # save params.json
    logger = D3RLPyLogger("test", root_dir="test_data", verbose=False)
    # save parameters to test_data/test/params.json
    save_config(algo, logger)
    # load params.json
    json_path = os.path.join(logger.logdir, "params.json")
    new_algo = algo.__class__.from_json(json_path)
    assert new_algo.impl is not None
    assert type(new_algo) == type(algo)
    assert tuple(new_algo.impl.observation_shape) == observation_shape
    assert new_algo.impl.action_size == action_size
    assert type(algo.observation_scaler) == type(new_algo.observation_scaler)
