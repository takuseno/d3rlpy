from typing import Any

import numpy as np
import pytest

from d3rlpy.algos import (
    DecisionTransformerConfig,
    DiscreteDecisionTransformerConfig,
    DQNConfig,
    QLearningAlgoBase,
    SACConfig,
    TransformerAlgoBase,
)
from d3rlpy.dataset import (
    BasicTransitionPicker,
    Episode,
    FrameStackTransitionPicker,
    InfiniteBuffer,
    ReplayBuffer,
)
from d3rlpy.types import ObservationSequence, Shape

from .models.torch.model_test import DummyEncoderFactory


def _dataset(
    observation_kind: str, discrete: bool, n_frames: int
) -> tuple[ReplayBuffer, Shape]:
    rng = np.random.default_rng(42)
    shapes = {
        "vector": [(3,)],
        "image": [(1, 4, 4)],
        "list": [(1, 4, 4), (3,)],
        "tuple": [(2,), (3,), (4,)],
    }[observation_kind]
    components = [
        rng.normal(size=(8, *shape)).astype(np.float32) for shape in shapes
    ]
    observations: ObservationSequence
    if observation_kind == "list":
        observations = components
    elif observation_kind == "tuple":
        observations = tuple(components)
    else:
        observations = components[0]
    actions = (
        np.arange(8).reshape(-1, 1) % 2
        if discrete
        else rng.uniform(-1, 1, (8, 2)).astype(np.float32)
    )
    episode = Episode(
        observations=observations,
        actions=actions,
        rewards=np.ones((8, 1), dtype=np.float32),
        terminated=True,
    )
    dataset = ReplayBuffer(
        InfiniteBuffer(),
        episodes=[episode],
        transition_picker=(
            BasicTransitionPicker()
            if n_frames == 1
            else FrameStackTransitionPicker(n_frames)
        ),
    )
    expected_shapes = [(shape[0] * n_frames, *shape[1:]) for shape in shapes]
    expected_shape: Shape = (
        expected_shapes[0] if len(expected_shapes) == 1 else expected_shapes
    )
    return dataset, expected_shape


@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize(
    "observation_kind", ["vector", "image", "list", "tuple"]
)
@pytest.mark.parametrize("n_frames", [1, 3])
def test_build_qlearning_with_dataset(
    discrete: bool, observation_kind: str, n_frames: int
) -> None:
    dataset, expected_shape = _dataset(observation_kind, discrete, n_frames)
    algo: QLearningAlgoBase[Any, Any]
    if discrete:
        algo = DQNConfig(encoder_factory=DummyEncoderFactory()).create()
    else:
        algo = SACConfig(
            actor_encoder_factory=DummyEncoderFactory(),
            critic_encoder_factory=DummyEncoderFactory(),
        ).create()

    algo.build_with_dataset(dataset)

    assert algo.observation_shape == expected_shape
    assert algo.action_size == 2
    batch = dataset.sample_transition_batch(4)
    losses = algo.update(batch)
    assert losses and np.isfinite(list(losses.values())).all()
    predictions = algo.predict(batch.observations)
    assert predictions.shape == ((4,) if discrete else (4, 2))
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize(
    "observation_kind", ["vector", "image", "list", "tuple"]
)
def test_build_transformer_with_dataset(
    discrete: bool, observation_kind: str
) -> None:
    dataset, expected_shape = _dataset(observation_kind, discrete, n_frames=1)
    algo: TransformerAlgoBase[Any, Any]
    if discrete:
        algo = DiscreteDecisionTransformerConfig(
            encoder_factory=DummyEncoderFactory(), num_heads=1, context_size=3
        ).create()
    else:
        algo = DecisionTransformerConfig(
            encoder_factory=DummyEncoderFactory(), num_heads=1, context_size=3
        ).create()

    algo.build_with_dataset(dataset)

    assert algo.observation_shape == expected_shape
    assert algo.action_size == 2
    losses = algo.update(dataset.sample_trajectory_batch(4, length=3))
    assert losses and np.isfinite(list(losses.values())).all()
