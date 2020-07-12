import torch

from d3rlpy.algos.base import ImplBase
from .utility import freeze, unfreeze
from .utility import to_cuda, to_cpu
from .utility import torch_api, eval_api
from .utility import map_location
from .utility import get_state_dict, set_state_dict


class TorchImplBase(ImplBase):
    @eval_api
    @torch_api
    def predict_best_action(self, x):
        if self.scaler:
            x = self.scaler.transform(x)

        with torch.no_grad():
            return self._predict_best_action(x).cpu().detach().numpy()

    def _predict_best_action(self, x):
        raise NotImplementedError

    @eval_api
    def save_policy(self, fname):
        dummy_x = torch.rand(1, *self.observation_shape, device=self.device)

        # workaround until version 1.6
        freeze(self)

        # dummy function to select best actions
        def _func(x):
            if self.scaler:
                x = self.scaler.transform(x)
            return self._predict_best_action(x)

        traced_script = torch.jit.trace(_func, dummy_x)
        traced_script.save(fname)

        # workaround until version 1.6
        unfreeze(self)

    def to_gpu(self):
        to_cuda(self)
        self.device = 'cuda:0'

    def to_cpu(self):
        to_cpu(self)
        self.device = 'cpu:0'

    def save_model(self, fname):
        torch.save(get_state_dict(self), fname)

    def load_model(self, fname):
        chkpt = torch.load(fname, map_location=map_location(self.device))
        set_state_dict(self, chkpt)
