from typing import Optional

import pytest

from d3rlpy.algos.qlearning.dqn import DoubleDQNConfig, DQNConfig
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
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_dqn(
    observation_shape: Shape,
    n_critics: int,
    q_func_factory: QFunctionFactory,
    scalers: Optional[str],
) -> None:
    observation_scaler, _, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    config = DQNConfig(
        encoder_factory=DummyEncoderFactory(),
        n_critics=n_critics,
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    dqn = config.create()
    algo_tester(
        dqn,  # type: ignore
        observation_shape,
        test_policy_copy=False,
        test_policy_optim_copy=False,
    )


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 32, 32), ((100,), (200,))]
)
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_double_dqn(
    observation_shape: Shape,
    n_critics: int,
    q_func_factory: QFunctionFactory,
    scalers: Optional[str],
) -> None:
    observation_scaler, _, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    config = DoubleDQNConfig(
        encoder_factory=DummyEncoderFactory(),
        n_critics=n_critics,
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    double_dqn = config.create()
    algo_tester(
        double_dqn,  # type: ignore
        observation_shape,
        test_policy_copy=False,
        test_policy_optim_copy=False,
    )
