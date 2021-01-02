import copy

from unittest.mock import patch
from d3rlpy.gpu import Device
from d3rlpy.context import parallel


@patch("d3rlpy.gpu.get_gpu_count", return_value=2)
def test_device(mock):
    device = Device()

    copy_device = copy.deepcopy(device)
    assert device.get_id() == 0
    assert copy_device.get_id() == 0

    with parallel():
        inc_device = copy.deepcopy(device)
        assert device.get_id() == 1
        assert inc_device.get_id() == 1

        # check circulation
        inc2_device = copy.deepcopy(device)
        assert device.get_id() == 0
        assert inc2_device.get_id() == 0
