import pytest
from typing import Any, Sequence, List, Union
from unittest.mock import MagicMock, Mock
from d3rlpy.dataset import Shape

from d3rlpy.algos.qlearning.torch.callbacks import ParameterReset
from d3rlpy.algos import QLearningAlgoBase, QLearningAlgoImplBase
from d3rlpy.torch_utility import Modules
import torch

from ...test_torch_utility import DummyModules


class LayerHasResetMock: 
       
    def reset_parameters(self):
        return True
    
class LayerNoResetMock: 
       pass

fc = torch.nn.Linear(100, 100)
optim = torch.optim.Adam(fc.parameters())
modules = DummyModules(fc=fc, optim=optim)

class ImplMock(MagicMock):
    
    def __init__(
        self, q_funcs:List[Union[LayerHasResetMock, LayerNoResetMock]]
        ) -> None:
        super().__init__(spec=QLearningAlgoImplBase)
        self.q_function = q_funcs


class QLearningAlgoBaseMock(MagicMock):
    
    def __init__(self, spec, layer_setup:Sequence[bool]) -> None:
        super().__init__(spec=spec)
        q_funcs = []
        for i in layer_setup:
            if i:
                q_funcs.append(LayerHasResetMock())
            else:
                q_funcs.append(LayerNoResetMock())
        self._impl = ImplMock(q_funcs=q_funcs)
    


def test_check_layer_resets():
    algo = QLearningAlgoBaseMock(spec=QLearningAlgoBase, 
                                 layer_setup=[True, True, False])
    replay_ratio = 2
    layer_reset_valid = [True, True, False]
    pr = ParameterReset(
        replay_ratio=replay_ratio, 
        layer_reset=layer_reset_valid,
        algo=algo
        )
    assert pr._check is True
    
    layer_reset_invalid = [True, True, True]
    try:
        pr = ParameterReset(
            replay_ratio=replay_ratio, 
            layer_reset=layer_reset_invalid,
            algo=algo
        )
        raise Exception
    except ValueError as e:
        assert True
    
    layer_reset_long = [True, True, True, False]
    try:
        pr = ParameterReset(
            replay_ratio=replay_ratio, 
            layer_reset=layer_reset_long,
            algo=algo
        )
        raise Exception
    except ValueError as e:
        assert True
    
    layer_reset_shrt = [True, True]
    try:
        pr = ParameterReset(
            replay_ratio=replay_ratio, 
            layer_reset=layer_reset_shrt,
            algo=algo
        )
        raise Exception
    except ValueError as e:
        assert True
    

def test_call():
    algo = QLearningAlgoBaseMock(spec=QLearningAlgoBase, 
                                 layer_setup=[True, True, False])
    replay_ratio = 2
    layer_reset_valid = [True, True, False]
    pr = ParameterReset(
        replay_ratio=replay_ratio, 
        layer_reset=layer_reset_valid,
        algo=algo
        )
    pr(algo=algo, epoch=1, total_step=100)
    pr(algo=algo, epoch=2, total_step=100)
    
    pr = ParameterReset(
        replay_ratio=replay_ratio, 
        layer_reset=layer_reset_valid,
        )
    pr(algo=algo, epoch=1, total_step=100)
    pr(algo=algo, epoch=2, total_step=100)