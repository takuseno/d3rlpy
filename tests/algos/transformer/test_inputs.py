from typing import Sequence

import numpy as np
import pytest
import torch

from d3rlpy.algos.transformer.inputs import (
    TorchTransformerInput,
    TransformerInput,
)
from d3rlpy.dataset import batch_pad_array, batch_pad_observations
from d3rlpy.preprocessing import ActionScaler, ObservationScaler, RewardScaler

from ...testing_utils import create_episode


class DummyObservationScaler(ObservationScaler):
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1


class DummyActionScaler(ActionScaler):
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2


class DummyRewardScaler(RewardScaler):
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2


@pytest.mark.parametrize("length", [10])
@pytest.mark.parametrize("context_size", [5, 15])
@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("use_observation_scaler", [False, True])
@pytest.mark.parametrize("use_action_scaler", [False, True])
@pytest.mark.parametrize("use_reward_scaler", [False, True])
def test_torch_transformer_input(
    length: int,
    context_size: int,
    observation_shape: Sequence[int],
    action_size: int,
    use_observation_scaler: bool,
    use_action_scaler: bool,
    use_reward_scaler: bool,
) -> None:
    episode = create_episode(observation_shape, action_size, length)

    inpt = TransformerInput(
        observations=episode.observations,
        actions=episode.actions,
        rewards=episode.rewards,
        returns_to_go=episode.rewards,
        timesteps=np.arange(length),
    )

    if length < context_size:
        pad_size = context_size - length
        ref_observations = batch_pad_observations(
            episode.observations, pad_size
        )
        ref_actions = batch_pad_array(episode.actions, pad_size)
        ref_rewards = batch_pad_array(episode.rewards, pad_size)
    else:
        ref_observations = episode.observations[-context_size:]
        ref_actions = episode.actions[-context_size:]
        ref_rewards = episode.rewards[-context_size:]

    if use_observation_scaler:
        observation_scaler = DummyObservationScaler()
    else:
        observation_scaler = None

    if use_action_scaler:
        action_scaler = DummyActionScaler()
    else:
        action_scaler = None

    if use_reward_scaler:
        reward_scaler = DummyRewardScaler()
    else:
        reward_scaler = None

    torch_inpt = TorchTransformerInput.from_numpy(
        inpt=inpt,
        context_size=context_size,
        device="cpu:0",
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )

    assert torch_inpt.observations.shape == (
        1,
        context_size,
        *observation_shape,
    )
    assert torch_inpt.actions.shape == (1, context_size, action_size)
    assert torch_inpt.rewards.shape == (1, context_size, 1)
    assert torch_inpt.returns_to_go.shape == (1, context_size, 1)
    assert torch_inpt.timesteps.shape == (1, context_size)
    assert torch_inpt.masks.shape == (1, context_size)
    assert torch_inpt.length == context_size

    if observation_scaler:
        assert isinstance(ref_observations, np.ndarray)
        assert np.allclose(
            torch_inpt.observations.numpy()[0], ref_observations + 0.1
        )
    else:
        assert np.allclose(torch_inpt.observations.numpy()[0], ref_observations)

    if action_scaler:
        assert np.allclose(torch_inpt.actions.numpy()[0], ref_actions + 0.2)
    else:
        assert np.allclose(torch_inpt.actions.numpy()[0], ref_actions)

    if reward_scaler:
        assert np.allclose(torch_inpt.rewards.numpy()[0], ref_rewards + 0.2)
        assert np.allclose(
            torch_inpt.returns_to_go.numpy()[0], ref_rewards + 0.2
        )
    else:
        assert np.allclose(torch_inpt.rewards.numpy()[0], ref_rewards)
        assert np.allclose(torch_inpt.returns_to_go.numpy()[0], ref_rewards)
