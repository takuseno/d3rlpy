from typing import Optional

import pytest

from d3rlpy.algos.qlearning.plas import PLASConfig, PLASWithPerturbationConfig
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
    "observation_shape", [(100,), (4, 84, 84), ((100,), (200,))]
)
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
@pytest.mark.parametrize("clip_gradient_norm", [None, 1.0])
def test_plas(
    observation_shape: Shape,
    q_func_factory: QFunctionFactory,
    scalers: Optional[str],
    clip_gradient_norm: Optional[float],
) -> None:
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    config = PLASConfig(
        actor_encoder_factory=DummyEncoderFactory(),
        critic_encoder_factory=DummyEncoderFactory(),
        imitator_encoder_factory=DummyEncoderFactory(),
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        warmup_steps=0,
        clip_gradient_norm=clip_gradient_norm,
    )
    plas = config.create()
    algo_tester(plas, observation_shape, test_policy_copy=False)  # type: ignore


@pytest.mark.parametrize(
    "observation_shape", [(100,), (4, 84, 84), ((100,), (200,))]
)
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
@pytest.mark.parametrize("clip_gradient_norm", [None, 1.0])
def test_plas_with_perturbation(
    observation_shape: Shape,
    q_func_factory: QFunctionFactory,
    scalers: Optional[str],
    clip_gradient_norm: Optional[float],
) -> None:
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers, observation_shape
    )
    config = PLASWithPerturbationConfig(
        actor_encoder_factory=DummyEncoderFactory(),
        critic_encoder_factory=DummyEncoderFactory(),
        imitator_encoder_factory=DummyEncoderFactory(),
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
        warmup_steps=0,
        clip_gradient_norm=clip_gradient_norm,
    )
    plas = config.create()
    algo_tester(plas, observation_shape, test_policy_copy=False)  # type: ignore
