from d3rlpy.gpu import Device


def test_device():
    device = Device()
    assert device.get_id() == 0
