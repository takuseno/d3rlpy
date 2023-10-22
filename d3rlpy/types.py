from typing import Any, Sequence, Union

import numpy as np
import numpy.typing as npt

__all__ = [
    "NDArray",
    "FloatNDArray",
    "IntNDArray",
    "UInt8NDArray",
    "DType",
    "Observation",
    "ObservationSequence",
    "Shape",
]


NDArray = npt.NDArray[Any]
FloatNDArray = npt.NDArray[np.float32]
IntNDArray = npt.NDArray[np.int32]
UInt8NDArray = npt.NDArray[np.uint8]
DType = npt.DTypeLike

Observation = Union[NDArray, Sequence[NDArray]]
ObservationSequence = Union[NDArray, Sequence[NDArray]]
Shape = Union[Sequence[int], Sequence[Sequence[int]]]
