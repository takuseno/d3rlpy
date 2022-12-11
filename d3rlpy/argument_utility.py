# pylint: disable=unidiomatic-typecheck

from typing import Optional, Union

from .gpu import Device

__all__ = ["UseGPUArg", "check_use_gpu"]

UseGPUArg = Optional[Union[bool, int, Device]]


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
