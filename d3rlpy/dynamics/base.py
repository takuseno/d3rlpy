from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..argument_utility import ActionScalerArg, ScalerArg
from ..base import ImplBase, LearnableBase
from ..constants import IMPL_NOT_INITIALIZED_ERROR


class DynamicsImplBase(ImplBase):
    @abstractmethod
    def predict(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


class DynamicsBase(LearnableBase):

    _impl: Optional[DynamicsImplBase]

    def __init__(
        self,
        batch_size: int,
        n_frames: int,
        scaler: ScalerArg,
        action_scaler: ActionScalerArg,
        kwargs: Dict[str, Any],
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=1,
            gamma=1.0,
            scaler=scaler,
            action_scaler=action_scaler,
            kwargs=kwargs,
        )
        self._impl = None

    def predict(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_variance: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Returns predicted observation and reward.

        Args:
            x: observation
            action: action
            with_variance: flag to return prediction variance.

        Returns:
            tuple of predicted observation and reward.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        observations, rewards, variances = self._impl.predict(x, action)
        if with_variance:
            return observations, rewards, variances
        return observations, rewards
