import dataclasses
from typing import Optional

import numpy as np
import torch

from ...dataset import (
    batch_pad_array,
    batch_pad_observations,
    get_axis_size,
    slice_observations,
)
from ...preprocessing import ActionScaler, ObservationScaler, RewardScaler
from ...torch_utility import convert_to_torch, convert_to_torch_recursively
from ...types import (
    Float32NDArray,
    Int32NDArray,
    NDArray,
    ObservationSequence,
    TorchObservation,
)

__all__ = ["TransformerInput", "TorchTransformerInput"]


@dataclasses.dataclass(frozen=True)
class TransformerInput:
    observations: ObservationSequence  # (L, ...)
    actions: NDArray  # (L, ...)
    rewards: Float32NDArray  # (L, 1)
    returns_to_go: Float32NDArray  # (L, 1)
    timesteps: Int32NDArray  # (L,)
    embeddings: Optional[Float32NDArray]

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
    observations: TorchObservation  # (1, L, ...)
    actions: torch.Tensor  # (1, L, ...)
    rewards: torch.Tensor  # (1, L, 1)
    returns_to_go: torch.Tensor  # (1, L, 1)
    timesteps: torch.Tensor  # (1, L)
    masks: torch.Tensor  # (1, L)
    length: int
    embeddings: Optional[torch.Tensor] = None

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
        masks: Float32NDArray
        if context_size < inpt.length:
            observations = slice_observations(
                inpt.observations, inpt.length - context_size, inpt.length
            )
            actions = inpt.actions[-context_size:]
            rewards = inpt.rewards[-context_size:]
            returns_to_go = inpt.returns_to_go[-context_size:]
            timesteps = inpt.timesteps[-context_size:]
            masks = np.ones(context_size, dtype=np.float32)
            embeddings = inpt.embeddings[-context_size:] if inpt.embeddings is not None else None
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
            embeddings = batch_pad_array(inpt.embeddings, pad_size) if inpt.embeddings is not None else None

        # convert numpy array to torch tensor
        observations_pt = convert_to_torch_recursively(observations, device)
        actions_pt = convert_to_torch(actions, device)
        rewards_pt = convert_to_torch(rewards, device)
        returns_to_go_pt = convert_to_torch(returns_to_go, device)
        timesteps_pt = convert_to_torch(timesteps, device).long()
        masks_pt = convert_to_torch(masks, device)
        embeddings_pt = None if embeddings is None else convert_to_torch(embeddings, device)

        # apply scaler
        if observation_scaler:
            observations_pt = observation_scaler.transform(observations_pt)
        if action_scaler:
            actions_pt = action_scaler.transform(actions_pt)
        if reward_scaler:
            rewards_pt = reward_scaler.transform(rewards_pt)
            returns_to_go_pt = reward_scaler.transform(returns_to_go_pt)

        if isinstance(observations_pt, torch.Tensor):
            unsqueezed_observation = observations_pt.unsqueeze(0)
        else:
            unsqueezed_observation = [o.unsqueeze(0) for o in observations_pt]

        return TorchTransformerInput(
            observations=unsqueezed_observation,
            actions=actions_pt.unsqueeze(0),
            rewards=rewards_pt.unsqueeze(0),
            returns_to_go=returns_to_go_pt.unsqueeze(0),
            timesteps=timesteps_pt.unsqueeze(0),
            masks=masks_pt.unsqueeze(0),
            length=context_size,
            embeddings=embeddings_pt.unsqueeze(0),
        )
