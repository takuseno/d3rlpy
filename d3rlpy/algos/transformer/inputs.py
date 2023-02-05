import dataclasses
from typing import Optional

import numpy as np
import torch

from ...dataset import (
    ObservationSequence,
    batch_pad_array,
    batch_pad_observations,
    get_axis_size,
    slice_observations,
)
from ...preprocessing import ActionScaler, ObservationScaler, RewardScaler
from ...torch_utility import convert_to_torch, convert_to_torch_recursively

__all__ = ["TransformerInput", "TorchTransformerInput"]


@dataclasses.dataclass(frozen=True)
class TransformerInput:
    observations: ObservationSequence  # (L, ...)
    actions: np.ndarray  # (L, ...)
    rewards: np.ndarray  # (L, 1)
    returns_to_go: np.ndarray  # (L, 1)
    timesteps: np.ndarray  # (L,)

    def __post_init__(self) -> None:
        # check sequence size
        length = get_axis_size(self.observations, axis=0)
        assert get_axis_size(self.actions, axis=0) == length
        assert get_axis_size(self.rewards, axis=0) == length
        assert get_axis_size(self.returns_to_go, axis=0) == length
        assert get_axis_size(self.timesteps, axis=0) == length

    @property
    def length(self) -> int:
        return get_axis_size(self.actions, axis=0)


@dataclasses.dataclass(frozen=True)
class TorchTransformerInput:
    observations: torch.Tensor  # (1, L, ...)
    actions: torch.Tensor  # (1, L, ...)
    rewards: torch.Tensor  # (1, L, 1)
    returns_to_go: torch.Tensor  # (1, L, 1)
    timesteps: torch.Tensor  # (1, L)
    masks: torch.Tensor  # (1, L)
    length: int

    @classmethod
    def from_numpy(
        cls,
        inpt: TransformerInput,
        context_size: int,
        device: str,
        observation_scaler: Optional[ObservationScaler] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scaler: Optional[RewardScaler] = None,
    ) -> "TorchTransformerInput":
        if context_size < inpt.length:
            observations = slice_observations(
                inpt.observations, inpt.length - context_size, inpt.length
            )
            actions = inpt.actions[-context_size:]
            rewards = inpt.rewards[-context_size:]
            returns_to_go = inpt.returns_to_go[-context_size:]
            timesteps = inpt.timesteps[-context_size:]
            masks = np.ones(context_size, dtype=np.float32)
        else:
            pad_size = context_size - inpt.length
            observations = batch_pad_observations(inpt.observations, pad_size)
            actions = batch_pad_array(inpt.actions, pad_size)
            rewards = batch_pad_array(inpt.rewards, pad_size)
            returns_to_go = batch_pad_array(inpt.returns_to_go, pad_size)
            timesteps = batch_pad_array(inpt.timesteps, pad_size)
            masks = batch_pad_array(
                np.ones(inpt.length, dtype=np.float32), pad_size
            )

        # convert numpy array to torch tensor
        observations = convert_to_torch_recursively(observations, device)
        actions = convert_to_torch(actions, device)
        rewards = convert_to_torch(rewards, device)
        returns_to_go = convert_to_torch(returns_to_go, device)
        timesteps = convert_to_torch(timesteps, device).long()
        masks = convert_to_torch(masks, device)

        # TODO: support tuple observation
        assert isinstance(observations, torch.Tensor)

        # apply scaler
        if observation_scaler:
            observations = observation_scaler.transform(observations)
        if action_scaler:
            actions = action_scaler.transform(actions)
        if reward_scaler:
            rewards = reward_scaler.transform(rewards)
            returns_to_go = reward_scaler.transform(returns_to_go)

        return TorchTransformerInput(
            observations=observations.unsqueeze(0),
            actions=actions.unsqueeze(0),
            rewards=rewards.unsqueeze(0),
            returns_to_go=returns_to_go.unsqueeze(0),
            timesteps=timesteps.unsqueeze(0),
            masks=masks.unsqueeze(0),
            length=context_size,
        )
