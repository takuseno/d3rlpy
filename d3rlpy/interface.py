from typing import Optional, Protocol, Union

from .preprocessing import ActionScaler, ObservationScaler, RewardScaler
from .types import NDArray, Observation

__all__ = ["QLearningAlgoProtocol", "StatefulTransformerAlgoProtocol"]


class QLearningAlgoProtocol(Protocol):
    def predict(self, x: Observation) -> NDArray: ...

    def predict_value(self, x: Observation, action: NDArray) -> NDArray: ...

    def sample_action(self, x: Observation) -> NDArray: ...

    @property
    def gamma(self) -> float: ...

    @property
    def observation_scaler(self) -> Optional[ObservationScaler]: ...

    @property
    def action_scaler(self) -> Optional[ActionScaler]: ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]: ...

    @property
    def action_size(self) -> Optional[int]: ...


class StatefulTransformerAlgoProtocol(Protocol):
    def predict(self, x: Observation, reward: float) -> Union[NDArray, int]: ...

    def reset(self) -> None: ...
