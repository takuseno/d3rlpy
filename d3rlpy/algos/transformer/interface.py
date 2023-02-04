from typing import Union

import numpy as np
from typing_extensions import Protocol

from ...dataset import Observation

__all__ = ["StatefulTransformerAlgoProtocol"]


class StatefulTransformerAlgoProtocol(Protocol):
    def predict(self, x: Observation, reward: float) -> Union[np.ndarray, int]:
        ...

    def reset(self) -> None:
        ...
