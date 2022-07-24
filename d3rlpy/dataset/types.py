from typing import Sequence, Union

import numpy as np

__all__ = ["Observation", "ObservationSequence", "Shape"]


Observation = Union[np.ndarray, Sequence[np.ndarray]]
ObservationSequence = Union[np.ndarray, Sequence[np.ndarray]]
Shape = Union[Sequence[int], Sequence[Sequence[int]]]
