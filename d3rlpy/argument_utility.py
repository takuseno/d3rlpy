# pylint: disable=unidiomatic-typecheck

from typing import Optional, Union

from .gpu import Device
from .models.encoders import EncoderFactory, create_encoder_factory
from .models.q_functions import QFunctionFactory, create_q_func_factory
from .preprocessing.action_scalers import ActionScaler, create_action_scaler
from .preprocessing.observation_scalers import (
    ObservationScaler,
    create_observation_scaler,
)
from .preprocessing.reward_scalers import RewardScaler, create_reward_scaler

__all__ = [
    "EncoderArg",
    "QFuncArg",
    "ObservationScalerArg",
    "ActionScalerArg",
    "RewardScalerArg",
    "UseGPUArg",
    "check_encoder",
    "check_q_func",
    "check_observation_scaler",
    "check_action_scaler",
    "check_reward_scaler",
    "check_use_gpu",
]

EncoderArg = Union[EncoderFactory, str]
QFuncArg = Union[QFunctionFactory, str]
ObservationScalerArg = Optional[Union[ObservationScaler, str]]
ActionScalerArg = Optional[Union[ActionScaler, str]]
RewardScalerArg = Optional[Union[RewardScaler, str]]
UseGPUArg = Optional[Union[bool, int, Device]]


def check_encoder(value: EncoderArg) -> EncoderFactory:
    """Checks value and returns EncoderFactory object.

    Returns:
        d3rlpy.encoders.EncoderFactory: encoder factory object.

    """
    if isinstance(value, EncoderFactory):
        return value
    if isinstance(value, str):
        return create_encoder_factory(value)
    raise ValueError("This argument must be str or EncoderFactory object.")


def check_q_func(value: QFuncArg) -> QFunctionFactory:
    """Checks value and returns QFunctionFactory object.

    Returns:
        d3rlpy.q_functions.QFunctionFactory: Q function factory object.

    """
    if isinstance(value, QFunctionFactory):
        return value
    if isinstance(value, str):
        return create_q_func_factory(value)
    raise ValueError("This argument must be str or QFunctionFactory object.")


def check_observation_scaler(
    value: ObservationScalerArg,
) -> Optional[ObservationScaler]:
    """Checks value and returns Scaler object.

    Returns:
        observation scaler object.

    """
    if isinstance(value, ObservationScaler):
        return value
    if isinstance(value, str):
        return create_observation_scaler(value)
    if value is None:
        return None
    raise ValueError("This argument must be str or ObservationScaler object.")


def check_action_scaler(value: ActionScalerArg) -> Optional[ActionScaler]:
    """Checks value and returns Scaler object.

    Returns:
        action scaler object.

    """
    if isinstance(value, ActionScaler):
        return value
    if isinstance(value, str):
        return create_action_scaler(value)
    if value is None:
        return None
    raise ValueError("This argument must be str or ActionScaler object.")


def check_reward_scaler(value: RewardScalerArg) -> Optional[RewardScaler]:
    """Checks value and returns Scaler object.

    Returns:
        reward scaler object.

    """
    if isinstance(value, RewardScaler):
        return value
    if isinstance(value, str):
        return create_reward_scaler(value)
    if value is None:
        return None
    raise ValueError("This argument must be str or RewardScaler object.")


def check_use_gpu(value: UseGPUArg) -> Optional[Device]:
    """Checks value and returns Device object.

    Returns:
        d3rlpy.gpu.Device: device object.

    """
    # isinstance cannot tell difference between bool and int
    if type(value) == bool:
        if value:
            return Device(0)
        return None
    if type(value) == int:
        return Device(value)
    if isinstance(value, Device):
        return value
    if value is None:
        return None
    raise ValueError("This argument must be bool, int or Device.")
