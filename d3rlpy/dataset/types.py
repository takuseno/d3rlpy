from typing import Sequence, Union

from ..types import NDArray

__all__ = ["Observation", "ObservationSequence", "Shape"]


Observation = Union[NDArray, Sequence[NDArray]]
ObservationSequence = Union[NDArray, Sequence[NDArray]]
Shape = Union[Sequence[int], Sequence[Sequence[int]]]
