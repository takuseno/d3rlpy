import torch
import copy

from torch.optim import Adam
from skbrl.models.torch.imitators import create_deterministic_regressor
from skbrl.algos.base import ImplBase
from skbrl.algos.torch.utility import torch_api, train_api, eval_api
from skbrl.algos.torch.utility import to_cuda, to_cpu
from skbrl.algos.torch.utility import freeze, unfreeze
from skbrl.algos.torch.utility import map_location


class BCImpl(ImplBase):
    def __init__(self, observation_shape, action_size, learning_rate, eps,
                 use_batch_norm, use_gpu):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.eps = eps
        self.use_batch_norm = use_batch_norm

        self._build_network()

        self.device = 'cpu:0'
        if use_gpu:
            self.to_gpu()

        self._build_optim()

    def _build_network(self):
        self.imitator = create_deterministic_regressor(self.observation_shape,
                                                       self.action_size)

    def _build_optim(self):
        self.optim = Adam(self.imitator.parameters(),
                          lr=self.learning_rate,
                          eps=self.eps)

    @train_api
    @torch_api
    def update_imitator(self, obs_t, act_t):
        loss = self.imitator.compute_l2_loss(obs_t, act_t)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.cpu().detach().numpy()

    def _predict_best_action(self, x):
        return self.imitator(x)

    @eval_api
    @torch_api
    def predict_best_action(self, x):
        with torch.no_grad():
            return self._predict_best_action(x).cpu().detach().numpy()

    def save_model(self, fname):
        torch.save(
            {
                'imitator': self.imitator.state_dict(),
                'optim': self.optim.state_dict()
            }, fname)

    def load_model(self, fname):
        chkpt = torch.load(fname, map_location=map_location(self.device))
        self.imitator.load_state_dict(chkpt['imitator'])
        self.optim.load_state_dict(chkpt['optim'])

    @eval_api
    def save_policy(self, fname):
        dummy_x = torch.rand(1, *self.observation_shape, device=self.device)

        # workaround until version 1.6
        freeze(self)

        # dummy function to select best actions
        def _func(x):
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
