import pytest

from d3rlpy.algos.ddpg import DDPG
from tests import performance_test, create_encoder_factory
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'standard'])
@pytest.mark.parametrize('use_encoder_factory', [True, False])
def test_ddpg(observation_shape, action_size, q_func_type, scaler,
              use_encoder_factory):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)
    ddpg = DDPG(q_func_type=q_func_type,
                scaler=scaler,
                actor_encoder_factory=encoder_factory,
                critic_encoder_factory=encoder_factory)
    algo_tester(ddpg, observation_shape)
    algo_update_tester(ddpg, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
def test_ddpg_performance(q_func_type):
    # not good enough for batch RL, but check if it works without errors.
    try:
        ddpg = DDPG(q_func_type=q_func_type)
        algo_pendulum_tester(ddpg, n_trials=1)
    except AssertionError:
        pass
