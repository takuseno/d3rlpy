import pytest

from d3rlpy.algos.sac import SAC, DiscreteSAC
from tests import performance_test, create_encoder_factory
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester
from .algo_test import algo_pendulum_tester, algo_cartpole_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'standard'])
@pytest.mark.parametrize('use_encoder_factory', [True, False])
def test_sac(observation_shape, action_size, q_func_type, scaler,
             use_encoder_factory):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)
    sac = SAC(q_func_type=q_func_type,
              scaler=scaler,
              actor_encoder_factory=encoder_factory,
              critic_encoder_factory=encoder_factory)
    algo_tester(sac, observation_shape)
    algo_update_tester(sac, observation_shape, action_size)


@performance_test
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
def test_sac_performance(q_func_type):
    if q_func_type == 'iqn' or q_func_type == 'fqf':
        pytest.skip('IQN is computationally expensive')

    sac = SAC(q_func_type=q_func_type)
    algo_pendulum_tester(sac, n_trials=3)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'standard'])
@pytest.mark.parametrize('use_encoder_factory', [True, False])
def test_discrete_sac(observation_shape, action_size, q_func_type, scaler,
                      use_encoder_factory):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)
    sac = DiscreteSAC(q_func_type=q_func_type,
                      scaler=scaler,
                      actor_encoder_factory=encoder_factory,
                      critic_encoder_factory=encoder_factory)
    algo_tester(sac, observation_shape)
    algo_update_tester(sac, observation_shape, action_size, discrete=True)


@performance_test
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
def test_discrete_sac_performance(q_func_type):
    if q_func_type == 'iqn' or q_func_type == 'fqf':
        pytest.skip('IQN is computationally expensive')

    sac = DiscreteSAC(q_func_type=q_func_type)
    algo_cartpole_tester(sac, n_trials=3)
