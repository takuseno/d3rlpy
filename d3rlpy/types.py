from typing import Any, Sequence, Union

import numpy.typing as npt

__all__ = ["NDArray", "DType", "Observation", "ObservationSequence", "Shape"]


NDArray = npt.NDArray[Any]
DType = npt.DTypeLike

Observation = Union[NDArray, Sequence[NDArray]]
ObservationSequence = Union[NDArray, Sequence[NDArray]]
Shape = Union[Sequence[int], Sequence[Sequence[int]]]
