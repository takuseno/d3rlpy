import pytest

from d3rlpy.algos.dqn import DQN, DoubleDQN
from tests import performance_test, create_encoder_factory
from .algo_test import algo_tester, algo_update_tester, algo_cartpole_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'standard'])
@pytest.mark.parametrize('use_encoder_factory', [True, False])
def test_dqn(observation_shape, action_size, q_func_type, scaler,
             use_encoder_factory):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)
    dqn = DQN(q_func_type=q_func_type,
              scaler=scaler,
              encoder_factory=encoder_factory)
    algo_tester(dqn, observation_shape)
    algo_update_tester(dqn, observation_shape, action_size, discrete=True)


@performance_test
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
def test_dqn_performance(q_func_type):
    dqn = DQN(q_func_type=q_func_type)
    algo_cartpole_tester(dqn)


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'standard'])
@pytest.mark.parametrize('use_encoder_factory', [True, False])
def test_double_dqn(observation_shape, action_size, q_func_type, scaler,
                    use_encoder_factory):
    encoder_factory = create_encoder_factory(use_encoder_factory,
                                             observation_shape)
    double_dqn = DoubleDQN(q_func_type=q_func_type,
                           scaler=scaler,
                           encoder_factory=encoder_factory)
    algo_tester(double_dqn, observation_shape)
    algo_update_tester(double_dqn, observation_shape, action_size, True)


@performance_test
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
def test_double_dqn_performance(q_func_type):
    double_dqn = DoubleDQN(q_func_type=q_func_type)
    algo_cartpole_tester(double_dqn)
