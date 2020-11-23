import pytest

from d3rlpy.algos.bcq import BCQ, DiscreteBCQ
from tests import performance_test, create_encoder_factory
from .algo_test import algo_tester, algo_update_tester
from .algo_test import algo_pendulum_tester, algo_cartpole_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'standard'])
@pytest.mark.parametrize('use_encoder_factory', [True, False])
def test_bcq(observation_shape, action_size, q_func_type, scaler,
             use_encoder_factory):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)
    bcq = BCQ(q_func_type=q_func_type,
              scaler=scaler,
              actor_encoder_factory=encoder_factory,
              critic_encoder_factory=encoder_factory,
              imitator_encoder_factory=encoder_factory)
    algo_tester(bcq, observation_shape)
    algo_update_tester(bcq, observation_shape, action_size)


@pytest.mark.skip(reason='BCQ is computationally expensive.')
def test_bcq_performance():
    bcq = BCQ(use_batch_norm=False)
    algo_pendulum_tester(bcq, n_trials=5)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'standard'])
@pytest.mark.parametrize('use_encoder_factory', [True, False])
def test_discrete_bcq(observation_shape, action_size, q_func_type, scaler,
                      use_encoder_factory):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)
    bcq = DiscreteBCQ(q_func_type=q_func_type,
                      scaler=scaler,
                      encoder_factory=encoder_factory)
    algo_tester(bcq, observation_shape)
    algo_update_tester(bcq, observation_shape, action_size, discrete=True)


@performance_test
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
def test_discrete_bcq_performance(q_func_type):
    bcq = DiscreteBCQ(q_func_type=q_func_type)
    algo_cartpole_tester(bcq)
