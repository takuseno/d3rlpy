import torch

from d3rlpy.dynamics.base import DynamicsImplBase
from d3rlpy.gpu import Device
from d3rlpy.algos.torch.utility import to_cuda, to_cpu
from d3rlpy.algos.torch.utility import torch_api, eval_api
from d3rlpy.algos.torch.utility import map_location
from d3rlpy.algos.torch.utility import get_state_dict, set_state_dict


class TorchImplBase(DynamicsImplBase):
    @eval_api
    @torch_api
    def predict(self, x, action):
        if self.scaler:
            x = self.scaler.transform(x)

        with torch.no_grad():
            observation, reward = self._predict(x, action)
            observation = observation.cpu().detach().numpy()
            reward = reward.cpu().detach().numpy()
            return observation, reward

    def _predict(self, x, action):
        raise NotImplementedError

    @eval_api
    @torch_api
    def generate(self, x, action):
        if self.scaler:
            x = self.scaler.transform(x)

        with torch.no_grad():
            observation, reward = self._generate(x, action)
            observation = observation.cpu().detach().numpy()
            reward = reward.cpu().detach().numpy()
            return observation, reward

    def _generate(self, x, action):
        raise NotImplementedError

    def to_gpu(self, device=Device()):
        self.device = 'cuda:%d' % device.get_id()
        to_cuda(self, self.device)

    def to_cpu(self):
        self.device = 'cpu:0'
        to_cpu(self)

    def save_model(self, fname):
        torch.save(get_state_dict(self), fname)

    def load_model(self, fname):
        chkpt = torch.load(fname, map_location=map_location(self.device))
        set_state_dict(self, chkpt)
