from typing import Optional

import pytest

from d3rlpy.algos.qlearning.cql import CQLConfig, DiscreteCQLConfig
from d3rlpy.models import (
    MeanQFunctionFactory,
    QFunctionFactory,
    QRQFunctionFactory,
)
from d3rlpy.types import Shape

from ...models.torch.model_test import DummyEncoderFactory
from ...testing_utils import create_scaler_tuple
from .algo_test import algo_tester


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 32, 32), ((100,), (200,))]
)
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_cql(
    observation_shape: Shape,
    q_func_factory: QFunctionFactory,
    scalers: Optional[str],
) -> None:
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    config = CQLConfig(
        actor_encoder_factory=DummyEncoderFactory(),
        critic_encoder_factory=DummyEncoderFactory(),
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    cql = config.create()
    algo_tester(cql, observation_shape)  # type: ignore


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 32, 32), ((100,), (200,))]
)
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, None, "min_max"])
def test_discrete_cql(
    observation_shape: Shape,
    n_critics: int,
    q_func_factory: QFunctionFactory,
    scalers: Optional[str],
) -> None:
    observation_scaler, _, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    config = DiscreteCQLConfig(
        encoder_factory=DummyEncoderFactory(),
        n_critics=n_critics,
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    cql = config.create()
    algo_tester(
        cql,  # type: ignore
        observation_shape,
        test_policy_copy=False,
        test_policy_optim_copy=False,
    )
