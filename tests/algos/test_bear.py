import pytest

from d3rlpy.algos.bear import BEAR
from tests import performance_test, create_encoder_factory
from .algo_test import algo_tester, algo_update_tester
from .algo_test import algo_pendulum_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'standard'])
@pytest.mark.parametrize('use_encoder_factory', [True, False])
def test_bear(observation_shape, action_size, q_func_type, scaler,
              use_encoder_factory):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)
    bear = BEAR(q_func_type=q_func_type,
                scaler=scaler,
                actor_encoder_factory=encoder_factory,
                critic_encoder_factory=encoder_factory,
                imitator_encoder_factory=encoder_factory)
    algo_tester(bear, observation_shape)
    algo_update_tester(bear, observation_shape, action_size)


@pytest.mark.skip(reason='BEAR is computationally expensive.')
def test_bear_performance():
    bear = BEAR()
    algo_pendulum_tester(bear, n_trials=3)
