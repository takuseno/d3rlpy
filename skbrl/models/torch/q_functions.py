import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import create_head


def create_discrete_q_function(observation_shape,
                               action_size,
                               n_ensembles=1,
                               n_quantiles=200,
                               use_batch_norm=False,
                               use_quantile_regression=False):
    q_funcs = []
    for _ in range(n_ensembles):
        head = create_head(observation_shape, use_batch_norm=use_batch_norm)
        if use_quantile_regression:
            q_func = QRQFunction(head, action_size, n_quantiles)
        else:
            q_func = DiscreteQFunction(head, action_size)
        q_funcs.append(q_func)

    if n_ensembles == 1:
        return q_funcs[0]

    return EnsembleDiscreteQFunction(q_funcs)


def create_continuous_q_function(observation_shape,
                                 action_size,
                                 n_ensembles=1,
                                 n_quantiles=200,
                                 use_batch_norm=False,
                                 use_quantile_regression=False):
    q_funcs = []
    for _ in range(n_ensembles):
        head = create_head(observation_shape,
                           action_size,
                           use_batch_norm=use_batch_norm)
        if use_quantile_regression:
            q_func = ContinuousQRQFunction(head, n_quantiles)
        else:
            q_func = ContinuousQFunction(head)
        q_funcs.append(q_func)

    if n_ensembles == 1:
        return q_funcs[0]

    return EnsembleContinuousQFunction(q_funcs)


def quantile_huber_loss(y, target, taus):
    # compute huber loss
    huber_loss = F.smooth_l1_loss(y, target)
    delta = ((target - y) < 0.0).float()
    return ((taus - delta).abs() * huber_loss).sum(dim=1).mean()


def make_mid_taus(n_quantiles, device):
    steps = torch.arange(n_quantiles, dtype=torch.float32, device=device)
    taus = ((steps + 1) / n_quantiles).view(1, -1)
    taus_dot = (steps / n_quantiles).view(1, -1)
    return (taus + taus_dot) / 2.0


def reduce_ensemble(y, reduction='min'):
    if reduction == 'min':
        return y.min(dim=0).values
    elif reduction == 'max':
        return y.max(dim=0).values
    elif reduction == 'mean':
        return y.mean(dim=0)
    elif reduction == 'none':
        return y
    else:
        raise ValueError


def reduce_quantile_ensemble(y, reduction='min'):
    # reduction beased on expectation
    mean = y.mean(dim=2)
    if reduction == 'min':
        indices = mean.min(dim=0).indices
    elif reduction == 'max':
        indices = mean.max(dim=0).indices
    elif reduction == 'none':
        return y
    else:
        raise ValueError
    return y.transpose(0, 1)[torch.arange(y.shape[1]), indices]


class QRQFunction(nn.Module):
    def __init__(self, head, action_size, n_quantiles):
        super().__init__()
        self.head = head
        self.action_size = action_size
        self.n_quantiles = n_quantiles
        self.fc = nn.Linear(head.feature_size, action_size * n_quantiles)

    def forward(self, x, as_quantiles=False):
        h = self.head(x)
        quantiles = self.fc(h).view(-1, self.action_size, self.n_quantiles)

        if as_quantiles:
            return quantiles

        return quantiles.mean(dim=2)

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        # extraect quantiles corresponding to act_t
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        quantiles = self.forward(obs_t, as_quantiles=True)
        masked_quantiles = quantiles * one_hot.view(-1, self.action_size, 1)
        quantiles_t = masked_quantiles.sum(dim=1)

        # exception will be raised when q_tp1 has an invald shape
        quantiles_tp1 = q_tp1.view(obs_t.shape[0], self.n_quantiles)

        # prepare taus for probabilities
        taus = make_mid_taus(self.n_quantiles, obs_t.device)

        # compute quantile huber loss
        y = rew_tp1 + gamma * quantiles_tp1
        return quantile_huber_loss(quantiles_t, y, taus)

    def compute_target(self, x, action):
        one_hot = F.one_hot(action.view(-1), num_classes=self.action_size)
        quantiles = self.forward(x, as_quantiles=True)
        masked_quantiles = quantiles * one_hot.view(-1, self.action_size, 1)
        return masked_quantiles.sum(dim=1)


class ContinuousQRQFunction(nn.Module):
    def __init__(self, head, n_quantiles):
        super().__init__()
        self.head = head
        self.action_size = head.action_size
        self.n_quantiles = n_quantiles
        self.fc = nn.Linear(head.feature_size, n_quantiles)

    def forward(self, x, action, as_quantiles=False):
        h = self.head(x, action)
        quantiles = self.fc(h)

        if as_quantiles:
            return quantiles

        return quantiles.mean(dim=1, keepdims=True)

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        assert q_tp1.shape == (obs_t.shape[0], self.n_quantiles)

        quantiles_t = self.forward(obs_t, act_t, as_quantiles=True)
        quantiles_tp1 = q_tp1

        # prepare taus for probabilities
        taus = make_mid_taus(self.n_quantiles, obs_t.device)

        # compute quantile huber loss
        y = rew_tp1 + gamma * quantiles_tp1
        return quantile_huber_loss(quantiles_t, y, taus)

    def compute_target(self, x, action):
        return self.forward(x, action, as_quantiles=True)


class DiscreteQFunction(nn.Module):
    def __init__(self, head, action_size):
        super().__init__()
        self.action_size = action_size
        self.head = head
        self.fc = nn.Linear(head.feature_size, action_size)

    def forward(self, x):
        h = self.head(x)
        return self.fc(h)

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        q_t = (self.forward(obs_t) * one_hot).sum(dim=1, keepdims=True)
        y = rew_tp1 + gamma * q_tp1
        return F.smooth_l1_loss(q_t, y)

    def compute_target(self, x, action):
        one_hot = F.one_hot(action.view(-1), num_classes=self.action_size)
        q_t = self.forward(x)
        return (q_t * one_hot).sum(dim=1, keepdims=True)


class ContinuousQFunction(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.action_size = head.action_size
        self.fc = nn.Linear(head.feature_size, 1)

    def forward(self, x, action):
        h = self.head(x, action)
        return self.fc(h)

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        q_t = self.forward(obs_t, act_t)
        y = rew_tp1 + gamma * q_tp1
        return F.mse_loss(q_t, y)

    def compute_target(self, x, action):
        return self.forward(x, action)


class EnsembleQFunction(nn.Module):
    def __init__(self, q_funcs):
        super().__init__()
        self.action_size = q_funcs[0].action_size
        self.q_funcs = nn.ModuleList(q_funcs)

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        td_sum = 0.0
        for q_func in self.q_funcs:
            td_sum += q_func.compute_td(obs_t, act_t, rew_tp1, q_tp1, gamma)
        return td_sum

    def compute_target(self, x, action, reduction='min'):
        values = []
        for q_func in self.q_funcs:
            target = q_func.compute_target(x, action)
            values.append(target.view(1, x.shape[0], -1))

        values = torch.cat(values, dim=0)

        if values.shape[2] == 1:
            return reduce_ensemble(values, reduction)

        return reduce_quantile_ensemble(values, reduction)


class EnsembleDiscreteQFunction(EnsembleQFunction):
    def forward(self, x, reduction='min'):
        values = []
        for q_func in self.q_funcs:
            values.append(q_func(x).view(1, x.shape[0], self.action_size))
        return reduce_ensemble(torch.cat(values, dim=0), reduction)


class EnsembleContinuousQFunction(EnsembleQFunction):
    def forward(self, x, action, reduction='min'):
        values = []
        for q_func in self.q_funcs:
            values.append(q_func(x, action).view(1, x.shape[0], 1))
        return reduce_ensemble(torch.cat(values, dim=0), reduction)
