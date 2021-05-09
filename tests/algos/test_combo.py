import pytest

from d3rlpy.algos.combo import COMBO
from d3rlpy.dynamics import ProbabilisticEnsembleDynamics

from .algo_test import algo_pendulum_tester, algo_tester, algo_update_tester


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("q_func_factory", ["mean", "qr", "iqn", "fqf"])
@pytest.mark.parametrize("scaler", [None, "min_max"])
@pytest.mark.parametrize("action_scaler", [None, "min_max"])
@pytest.mark.parametrize("target_reduction_type", ["min", "none"])
@pytest.mark.parametrize("rollout_interval", [1])
@pytest.mark.parametrize("rollout_horizon", [2])
@pytest.mark.parametrize("rollout_batch_size", [4])
def test_combo(
    observation_shape,
    action_size,
    q_func_factory,
    scaler,
    action_scaler,
    target_reduction_type,
    rollout_interval,
    rollout_horizon,
    rollout_batch_size,
):
    dynamics = ProbabilisticEnsembleDynamics()
    dynamics.create_impl(observation_shape, action_size)

    combo = COMBO(
        dynamics=dynamics,
        q_func_factory=q_func_factory,
        scaler=scaler,
        action_scaler=action_scaler,
        target_reduction_type=target_reduction_type,
        rollout_interval=rollout_interval,
        rollout_horizon=rollout_horizon,
        rollout_batch_size=rollout_batch_size,
    )
    algo_tester(combo, observation_shape)
    algo_update_tester(combo, observation_shape, action_size)


@pytest.mark.skip(reason="COMBO is computationally expensive.")
def test_combo_performance():
    combo = COMBO()
    algo_pendulum_tester(combo, n_trials=3)
