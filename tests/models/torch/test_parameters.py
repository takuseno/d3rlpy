from typing import Sequence

import pytest
import torch

from d3rlpy.models.torch.parameters import Parameter, get_parameter


@pytest.mark.parametrize("shape", [(100,)])
def test_parameter(shape: Sequence[int]) -> None:
    data = torch.rand(shape)
    parameter = Parameter(data)

    assert get_parameter(parameter).data.shape == shape
    assert torch.all(get_parameter(parameter).data == data)
