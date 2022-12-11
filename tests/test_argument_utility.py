import pytest

from d3rlpy.argument_utility import check_use_gpu
from d3rlpy.gpu import Device


@pytest.mark.parametrize("value", [False, True, 0, Device(0)])
def test_check_use_gpu(value):
    device = check_use_gpu(value)
    if type(value) == bool and value:
        assert device.get_id() == 0
    elif type(value) == bool and not value:
        assert device is None
    elif type(value) == int:
        assert device.get_id() == 0
    elif isinstance(value, Device):
        assert device.get_id() == 0
