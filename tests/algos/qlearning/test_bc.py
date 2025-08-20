from typing import Optional

import pytest

from d3rlpy.algos.qlearning.bc import BCConfig, DiscreteBCConfig
from d3rlpy.types import Shape

from ...models.torch.model_test import DummyEncoderFactory
from ...testing_utils import create_scaler_tuple
from .algo_test import algo_tester


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 32, 32), ((100,), (200,))]
)
@pytest.mark.parametrize("policy_type", ["deterministic", "stochastic"])
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_bc(
    observation_shape: Shape, policy_type: str, scalers: Optional[str]
) -> None:
    observation_scaler, action_scaler, _ = create_scaler_tuple(
        scalers, observation_shape
    )
    config = BCConfig(
        encoder_factory=DummyEncoderFactory(),
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        policy_type=policy_type,
    )
    bc = config.create()
    algo_tester(
        bc,  # type: ignore
        observation_shape,
        test_predict_value=False,
        test_policy_copy=False,
        test_policy_optim_copy=False,
        test_q_function_optim_copy=False,
        test_q_function_copy=False,
    )


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 32, 32), ((100,), (200,))]
)
@pytest.mark.parametrize("scaler", [None, "min_max"])
def test_discrete_bc(observation_shape: Shape, scaler: Optional[str]) -> None:
    observation_scaler, _, _ = create_scaler_tuple(scaler, observation_shape)
    config = DiscreteBCConfig(
        encoder_factory=DummyEncoderFactory(),
        observation_scaler=observation_scaler,
    )
    bc = config.create()
    algo_tester(
        bc,  # type: ignore
        observation_shape,
        test_policy_copy=False,
        test_predict_value=False,
        test_policy_optim_copy=False,
        test_q_function_optim_copy=False,
        test_q_function_copy=False,
    )
