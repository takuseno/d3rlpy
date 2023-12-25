from typing import Optional

import pytest

from d3rlpy.algos.qlearning import DDPGConfig, DQNConfig
from d3rlpy.models import (
    MeanQFunctionFactory,
    QFunctionFactory,
    QRQFunctionFactory,
)
from d3rlpy.ope.fqe import FQE, DiscreteFQE, FQEConfig
from d3rlpy.types import Shape
from tests.algos.qlearning.algo_test import algo_tester

from ..models.torch.model_test import DummyEncoderFactory
from ..testing_utils import create_scaler_tuple


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 84, 84), ((100,), (200,))]
)
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_fqe(
    observation_shape: Shape,
    q_func_factory: QFunctionFactory,
    scalers: Optional[str],
) -> None:
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    algo = DDPGConfig(
        actor_encoder_factory=DummyEncoderFactory(),
        critic_encoder_factory=DummyEncoderFactory(),
    ).create()
    algo.create_impl(observation_shape, 2)
    config = FQEConfig(
        encoder_factory=DummyEncoderFactory(),
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        q_func_factory=q_func_factory,
    )
    fqe = FQE(algo=algo, config=config)  # type: ignore
    algo_tester(
        fqe,  # type: ignore
        observation_shape,
        test_policy_copy=False,
        test_policy_optim_copy=False,
        test_from_json=False,
        test_q_function_copy=False,
        test_q_function_optim_copy=False,
    )


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 84, 84), ((100,), (200,))]
)
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_discrete_fqe(
    observation_shape: Shape,
    q_func_factory: QFunctionFactory,
    scalers: Optional[str],
) -> None:
    observation_scaler, _, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    algo = DQNConfig(encoder_factory=DummyEncoderFactory()).create()
    algo.create_impl(observation_shape, 2)
    config = FQEConfig(
        encoder_factory=DummyEncoderFactory(),
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
        q_func_factory=q_func_factory,
    )
    fqe = DiscreteFQE(algo=algo, config=config)  # type: ignore
    algo_tester(
        fqe,  # type: ignore
        observation_shape,
        test_policy_copy=False,
        test_policy_optim_copy=False,
        test_from_json=False,
        test_q_function_copy=False,
        test_q_function_optim_copy=False,
    )
