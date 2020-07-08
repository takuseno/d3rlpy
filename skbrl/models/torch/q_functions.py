import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .heads import create_head


def create_discrete_q_function(observation_shape,
                               action_size,
                               n_ensembles=1,
                               n_quantiles=32,
                               embed_size=64,
                               use_batch_norm=False,
                               distribution_type=None):
    q_funcs = []
    for _ in range(n_ensembles):
        head = create_head(observation_shape, use_batch_norm=use_batch_norm)
        if distribution_type:
            if distribution_type == 'qr':
                q_func = DiscreteQRQFunction(head, action_size, n_quantiles)
            elif distribution_type == 'iqn':
                q_func = DiscreteIQNQFunction(head, action_size, n_quantiles,
                                              embed_size)
            else:
                raise ValueError('invalid quantile regression type')
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
                                 embed_size=64,
                                 use_batch_norm=False,
                                 distribution_type=None):
    q_funcs = []
    for _ in range(n_ensembles):
        head = create_head(observation_shape,
                           action_size,
                           use_batch_norm=use_batch_norm)
        if distribution_type:
            if distribution_type == 'qr':
                q_func = ContinuousQRQFunction(head, n_quantiles)
            elif distribution_type == 'iqn':
                q_func = ContinuousIQNQFunction(head, n_quantiles, embed_size)
            else:
                raise ValueError('invalid quantile regression type')
        else:
            q_func = ContinuousQFunction(head)
        q_funcs.append(q_func)

    if n_ensembles == 1:
        return q_funcs[0]

    return EnsembleContinuousQFunction(q_funcs)


def _pick_value_by_action(values, action, keepdims=False):
    action_size = values.shape[1]
    one_hot = F.one_hot(action.view(-1), num_classes=action_size)
    # take care of 3 dimensional vectors
    if values.ndim == 3:
        one_hot = one_hot.view(-1, action_size, 1)
    masked_values = values * one_hot
    return masked_values.sum(dim=1, keepdims=keepdims)


def _huber_loss(y, target, beta=1.0):
    diff = target - y
    cond = diff.detach().abs() < beta
    return torch.where(cond, 0.5 * diff**2, beta * (diff.abs() - 0.5 * beta))


def _quantile_huber_loss(y, target, taus):
    # compute huber loss
    huber_loss = _huber_loss(y, target)
    delta = ((target - y).detach() < 0.0).float()
    element_wise_loss = ((taus - delta).abs() * huber_loss)
    if element_wise_loss.ndim == 3:  # for implicit quantile network
        element_wise_loss = element_wise_loss.sum(dim=2)
    return element_wise_loss.mean()


def _make_taus_prime(n_quantiles, device):
    steps = torch.arange(n_quantiles, dtype=torch.float32, device=device)
    taus = ((steps + 1) / n_quantiles).view(1, -1)
    taus_dot = (steps / n_quantiles).view(1, -1)
    return (taus + taus_dot) / 2.0


class DiscreteQRQFunction(nn.Module):
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
        assert q_tp1.shape == (obs_t.shape[0], self.n_quantiles)

        # extraect quantiles corresponding to act_t
        quantiles_t = _pick_value_by_action(self.forward(obs_t, True), act_t)

        # prepare taus for probabilities
        taus = _make_taus_prime(self.n_quantiles, obs_t.device)

        # compute quantile huber loss
        y = rew_tp1 + gamma * q_tp1
        return _quantile_huber_loss(quantiles_t, y, taus)

    def compute_target(self, x, action):
        return _pick_value_by_action(self.forward(x, True), action)


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

        # prepare taus for probabilities
        taus = _make_taus_prime(self.n_quantiles, obs_t.device)

        # compute quantile huber loss
        y = rew_tp1 + gamma * q_tp1
        return _quantile_huber_loss(quantiles_t, y, taus)

    def compute_target(self, x, action):
        return self.forward(x, action, as_quantiles=True)


class DiscreteIQNQFunction(nn.Module):
    def __init__(self, head, action_size, n_quantiles, embed_size):
        super().__init__()
        self.head = head
        self.action_size = action_size
        self.embed_size = embed_size
        self.n_quantiles = n_quantiles
        self.embed = nn.Linear(embed_size, self.head.feature_size)
        self.fc = nn.Linear(head.feature_size, action_size)

    def forward(self, x, as_quantiles=False, with_taus=False):
        taus = torch.rand(x.shape[0], self.n_quantiles, 1, device=x.device)
        steps = torch.arange(self.embed_size, device=x.device) + 1
        # (batch, quantile, embedding)
        prior = torch.cos(math.pi * steps.view(1, 1, -1) * taus)
        # (batch, quantile, embedding) -> (batch, quantile, feature)
        phi = torch.relu(self.embed(prior))

        h = self.head(x)
        # (batch, 1, feature) -> (batch, quantile, feature)
        prod = h.view(x.shape[0], 1, -1) * phi

        # (batch, quantile, feature) -> (batch, action, quantile)
        quantiles = self.fc(prod).transpose(1, 2)

        rets = []

        if as_quantiles:
            rets.append(quantiles)
        else:
            rets.append(quantiles.mean(dim=2))

        if with_taus:
            rets.append(taus)

        if len(rets) == 1:
            return rets[0]
        return rets

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        assert q_tp1.shape == (obs_t.shape[0], self.n_quantiles)

        # extraect quantiles corresponding to act_t
        values, taus = self.forward(obs_t, as_quantiles=True, with_taus=True)
        quantiles_t = _pick_value_by_action(values, act_t)

        # compute errors with all combination
        y = (rew_tp1 + gamma * q_tp1).view(obs_t.shape[0], -1, 1)
        quantiles_t = quantiles_t.view(obs_t.shape[0], 1, -1)

        return _quantile_huber_loss(quantiles_t, y, taus)

    def compute_target(self, x, action):
        quantiles = self.forward(x, as_quantiles=True)
        return _pick_value_by_action(quantiles, action)


class ContinuousIQNQFunction(nn.Module):
    def __init__(self, head, n_quantiles, embed_size):
        super().__init__()
        self.head = head
        self.action_size = head.action_size
        self.embed_size = embed_size
        self.n_quantiles = n_quantiles
        self.embed = nn.Linear(embed_size, self.head.feature_size)
        self.fc = nn.Linear(head.feature_size, 1)

    def forward(self, x, action, as_quantiles=False, with_taus=False):
        taus = torch.rand(x.shape[0], self.n_quantiles, 1, device=x.device)
        steps = torch.arange(self.embed_size, device=x.device) + 1
        # (batch, quantile, embedding)
        prior = torch.cos(math.pi * steps.view(1, 1, -1) * taus)
        # (batch, quantile, embedding) -> (batch, quantile, feature)
        phi = torch.relu(self.embed(prior))

        h = self.head(x, action)
        # (batch, 1, feature) -> (batch, quantile, feature)
        prod = h.view(x.shape[0], 1, -1) * phi

        # (batch, quantile, feature) -> (batch, quantile)
        quantiles = self.fc(prod).view(x.shape[0], self.n_quantiles)

        rets = []

        if as_quantiles:
            rets.append(quantiles)
        else:
            rets.append(quantiles.mean(dim=1, keepdims=True))

        if with_taus:
            rets.append(taus)

        if len(rets) == 1:
            return rets[0]
        return rets

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        assert q_tp1.shape == (obs_t.shape[0], self.n_quantiles)

        quantiles_t, taus = self.forward(obs_t,
                                         act_t,
                                         as_quantiles=True,
                                         with_taus=True)

        # compute errors with all combination
        y = (rew_tp1 + gamma * q_tp1).view(obs_t.shape[0], -1, 1)
        quantiles_t = quantiles_t.view(obs_t.shape[0], 1, -1)

        return _quantile_huber_loss(quantiles_t, y, taus)

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
        return _huber_loss(q_t, y).mean()

    def compute_target(self, x, action):
        return _pick_value_by_action(self.forward(x), action, keepdims=True)


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


def _reduce_ensemble(y, reduction='min'):
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


def _reduce_quantile_ensemble(y, reduction='min'):
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
            return _reduce_ensemble(values, reduction)

        return _reduce_quantile_ensemble(values, reduction)


class EnsembleDiscreteQFunction(EnsembleQFunction):
    def forward(self, x, reduction='min'):
        values = []
        for q_func in self.q_funcs:
            values.append(q_func(x).view(1, x.shape[0], self.action_size))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)


class EnsembleContinuousQFunction(EnsembleQFunction):
    def forward(self, x, action, reduction='min'):
        values = []
        for q_func in self.q_funcs:
            values.append(q_func(x, action).view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)
