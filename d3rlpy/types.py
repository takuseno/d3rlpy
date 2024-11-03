from typing import Any, Mapping, Sequence, Union

import gym
import gymnasium
import numpy as np
import numpy.typing as npt
import torch
from torch.optim import Optimizer
from typing_extensions import Protocol, runtime_checkable

__all__ = [
    "NDArray",
    "Float32NDArray",
    "Int32NDArray",
    "UInt8NDArray",
    "DType",
    "Observation",
    "ObservationSequence",
    "Shape",
    "TorchObservation",
    "GymEnv",
    "OptimizerWrapperProto",
]


NDArray = npt.NDArray[Any]
Float32NDArray = npt.NDArray[np.float32]
Int32NDArray = npt.NDArray[np.int32]
UInt8NDArray = npt.NDArray[np.uint8]
DType = npt.DTypeLike

Observation = Union[NDArray, Sequence[NDArray]]
ObservationSequence = Union[NDArray, Sequence[NDArray]]
Shape = Union[Sequence[int], Sequence[Sequence[int]]]
TorchObservation = Union[torch.Tensor, Sequence[torch.Tensor]]

GymEnv = Union[gym.Env[Any, Any], gymnasium.Env[Any, Any]]


@runtime_checkable
class OptimizerWrapperProto(Protocol):
    @property
    def optim(self) -> Optimizer:
        raise NotImplementedError

    def state_dict(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        raise NotImplementedError
