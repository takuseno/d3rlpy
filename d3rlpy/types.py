from typing import Any, Sequence, Union

import gym
import gymnasium
import numpy as np
import numpy.typing as npt

__all__ = [
    "NDArray",
    "Float32NDArray",
    "Int32NDArray",
    "UInt8NDArray",
    "DType",
    "Observation",
    "ObservationSequence",
    "Shape",
    "GymEnv",
]


NDArray = npt.NDArray[Any]
Float32NDArray = npt.NDArray[np.float32]
Int32NDArray = npt.NDArray[np.int32]
UInt8NDArray = npt.NDArray[np.uint8]
DType = npt.DTypeLike

Observation = Union[NDArray, Sequence[NDArray]]
ObservationSequence = Union[NDArray, Sequence[NDArray]]
Shape = Union[Sequence[int], Sequence[Sequence[int]]]

GymEnv = Union[gym.Env[Any, Any], gymnasium.Env[Any, Any]]
