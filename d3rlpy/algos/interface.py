from typing import Any, List, Optional, Tuple, Union

import numpy as np
from typing_extensions import Protocol

from ..preprocessing import ActionScaler, ObservationScaler, RewardScaler

__all__ = ["AlgoProtocol"]


class AlgoProtocol(Protocol):
    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        ...

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    @property
    def gamma(self) -> float:
        ...

    @property
    def observation_scaler(self) -> Optional[ObservationScaler]:
        ...

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        ...
