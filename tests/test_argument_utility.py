import pytest

from d3rlpy.argument_utility import (
    check_action_scaler,
    check_encoder,
    check_q_func,
    check_reward_scaler,
    check_scaler,
    check_use_gpu,
)
from d3rlpy.gpu import Device
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.preprocessing.action_scalers import MinMaxActionScaler
from d3rlpy.preprocessing.reward_scalers import MinMaxRewardScaler
from d3rlpy.preprocessing.scalers import MinMaxScaler


@pytest.mark.parametrize("value", ["default", DefaultEncoderFactory()])
def test_check_encoder(value):
    assert isinstance(check_encoder(value), DefaultEncoderFactory)


@pytest.mark.parametrize("value", ["mean", MeanQFunctionFactory()])
def test_check_q_func(value):
    assert isinstance(check_q_func(value), MeanQFunctionFactory)


@pytest.mark.parametrize("value", ["min_max", MinMaxScaler(), None])
def test_check_scaler(value):
    scaler = check_scaler(value)
    if value is None:
        assert scaler is None
    else:
        assert isinstance(scaler, MinMaxScaler)


@pytest.mark.parametrize("value", ["min_max", MinMaxActionScaler(), None])
def test_check_action_scaler(value):
    scaler = check_action_scaler(value)
    if value is None:
        assert scaler is None
    else:
        assert isinstance(scaler, MinMaxActionScaler)


@pytest.mark.parametrize("value", ["min_max", MinMaxRewardScaler(), None])
def test_check_reward_scaler(value):
    scaler = check_reward_scaler(value)
    if value is None:
        assert scaler is None
    else:
        assert isinstance(scaler, MinMaxRewardScaler)


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
