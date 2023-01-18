import pytest

from d3rlpy.algos.qlearning.crr import CRRConfig
from d3rlpy.models import MeanQFunctionFactory, QRQFunctionFactory

from ...testing_utils import create_scaler_tuple
from .algo_test import algo_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
@pytest.mark.parametrize("advantage_type", ["mean", "max"])
@pytest.mark.parametrize("weight_type", ["exp", "binary"])
@pytest.mark.parametrize("target_update_type", ["hard", "soft"])
def test_crr(
    observation_shape,
    q_func_factory,
    scalers,
    advantage_type,
    weight_type,
    target_update_type,
):
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers
    )
    config = CRRConfig(
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        advantage_type=advantage_type,
        weight_type=weight_type,
        target_update_type=target_update_type,
    )
    crr = config.create()
    algo_tester(
        crr,
        observation_shape,
        deterministic_best_action=False,
        test_policy_copy=False,
    )
