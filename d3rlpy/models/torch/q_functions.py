import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .heads import create_head


def create_discrete_q_function(observation_shape,
                               action_size,
                               n_ensembles=1,
                               n_quantiles=32,
                               embed_size=64,
                               use_batch_norm=False,
                               q_func_type='mean',
                               bootstrap=False):
    q_funcs = []
    for _ in range(n_ensembles):
        head = create_head(observation_shape, use_batch_norm=use_batch_norm)
        if q_func_type == 'mean':
            q_func = DiscreteQFunction(head, action_size)
        elif q_func_type == 'qr':
            q_func = DiscreteQRQFunction(head, action_size, n_quantiles)
        elif q_func_type == 'iqn':
            q_func = DiscreteIQNQFunction(head, action_size, n_quantiles,
                                          embed_size)
        elif q_func_type == 'fqf':
            q_func = DiscreteFQFQFunction(head, action_size, n_quantiles,
                                          embed_size)
        else:
            raise ValueError('invalid quantile regression type')
        q_funcs.append(q_func)
    return EnsembleDiscreteQFunction(q_funcs, bootstrap)


def create_continuous_q_function(observation_shape,
                                 action_size,
                                 n_ensembles=1,
                                 n_quantiles=32,
                                 embed_size=64,
                                 use_batch_norm=False,
                                 q_func_type='mean',
                                 bootstrap=False):
    q_funcs = []
    for _ in range(n_ensembles):
        head = create_head(observation_shape,
                           action_size,
                           use_batch_norm=use_batch_norm)
        if q_func_type == 'mean':
            q_func = ContinuousQFunction(head)
        elif q_func_type == 'qr':
            q_func = ContinuousQRQFunction(head, n_quantiles)
        elif q_func_type == 'iqn':
            q_func = ContinuousIQNQFunction(head, n_quantiles, embed_size)
        elif q_func_type == 'fqf':
            q_func = ContinuousFQFQFunction(head, n_quantiles, embed_size)
        else:
            raise ValueError('invalid quantile regression type')
        q_funcs.append(q_func)
    return EnsembleContinuousQFunction(q_funcs, bootstrap)


def _pick_value_by_action(values, action, keepdims=False):
    action_size = values.shape[1]
    one_hot = F.one_hot(action.view(-1), num_classes=action_size)
    # take care of 3 dimensional vectors
    if values.ndim == 3:
        one_hot = one_hot.view(-1, action_size, 1)
    masked_values = values * one_hot.float()
    return masked_values.sum(dim=1, keepdims=keepdims)


def _huber_loss(y, target, beta=1.0):
    diff = target - y
    cond = diff.detach().abs() < beta
    return torch.where(cond, 0.5 * diff**2, beta * (diff.abs() - 0.5 * beta))


def _quantile_huber_loss(y, target, taus):
    assert y.ndim == 3
    assert target.ndim == 3
    assert taus.ndim == 3
    # compute huber loss
    huber_loss = _huber_loss(y, target)
    delta = ((target - y).detach() < 0.0).float()
    element_wise_loss = ((taus - delta).abs() * huber_loss)
    return element_wise_loss.sum(dim=2).mean(dim=1)


def _make_taus_prime(n_quantiles, device):
    steps = torch.arange(n_quantiles, dtype=torch.float32, device=device)
    taus = ((steps + 1).float() / n_quantiles).view(1, -1)
    taus_dot = (steps.float() / n_quantiles).view(1, -1)
    return (taus + taus_dot) / 2.0


def _reduce(value, reduction_type):
    if reduction_type == 'mean':
        return value.mean()
    elif reduction_type == 'sum':
        return value.sum()
    elif reduction_type == 'none':
        return value.view(-1, 1)
    raise ValueError('invalid reduction type.')


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

    def compute_error(self,
                      obs_t,
                      act_t,
                      rew_tp1,
                      q_tp1,
                      gamma=0.99,
                      reduction='mean'):
        assert q_tp1.shape == (obs_t.shape[0], self.n_quantiles)

        # extraect quantiles corresponding to act_t
        quantiles_t = _pick_value_by_action(self.forward(obs_t, True), act_t)

        # prepare taus for probabilities
        taus = _make_taus_prime(self.n_quantiles, obs_t.device)

        # compute quantile huber loss
        y = (rew_tp1 + gamma * q_tp1).view(obs_t.shape[0], -1, 1)
        quantiles_t = quantiles_t.view(obs_t.shape[0], 1, -1)
        taus = taus.view(1, 1, -1)
        loss = _quantile_huber_loss(quantiles_t, y, taus)

        return _reduce(loss, reduction)

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

    def compute_error(self,
                      obs_t,
                      act_t,
                      rew_tp1,
                      q_tp1,
                      gamma=0.99,
                      reduction='mean'):
        assert q_tp1.shape == (obs_t.shape[0], self.n_quantiles)

        quantiles_t = self.forward(obs_t, act_t, as_quantiles=True)

        # prepare taus for probabilities
        taus = _make_taus_prime(self.n_quantiles, obs_t.device)

        # compute quantile huber loss
        y = (rew_tp1 + gamma * q_tp1).view(obs_t.shape[0], -1, 1)
        quantiles_t = quantiles_t.view(obs_t.shape[0], 1, -1)
        taus = taus.view(1, 1, -1)
        loss = _quantile_huber_loss(quantiles_t, y, taus)

        return _reduce(loss, reduction)

    def compute_target(self, x, action):
        return self.forward(x, action, as_quantiles=True)


class DiscreteIQNQFunction(nn.Module):
    def __init__(self, head, action_size, n_quantiles, embed_size):
        super().__init__()
        self.head = head
        self.action_size = action_size
        self.embed_size = embed_size
        self.n_quantiles = n_quantiles
        self.embed = nn.Linear(embed_size, head.feature_size)
        self.fc = nn.Linear(head.feature_size, self.action_size)

    def _make_taus(self, h):
        taus = torch.rand(h.shape[0], self.n_quantiles, device=h.device)
        return taus

    def _compute_quantiles(self, h, taus):
        # compute embedding
        steps = torch.arange(self.embed_size, device=h.device).float() + 1
        # (batch, quantile, embedding)
        expanded_taus = taus.view(h.shape[0], self.n_quantiles, 1)
        prior = torch.cos(math.pi * steps.view(1, 1, -1) * expanded_taus)
        # (batch, quantile, embedding) -> (batch, quantile, feature)
        phi = torch.relu(self.embed(prior))

        # (batch, 1, feature) -> (batch,  quantile, feature)
        prod = h.view(h.shape[0], 1, -1) * phi

        # (batch, quantile, feature) -> (batch, action, quantile)
        return self.fc(prod).transpose(1, 2)

    def forward(self, x, as_quantiles=False, with_taus=False):
        h = self.head(x)

        taus = self._make_taus(h)

        quantiles = self._compute_quantiles(h, taus)

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

    def compute_error(self,
                      obs_t,
                      act_t,
                      rew_tp1,
                      q_tp1,
                      gamma=0.99,
                      reduction='mean'):
        assert q_tp1.shape == (obs_t.shape[0], self.n_quantiles)

        # extraect quantiles corresponding to act_t
        values, taus = self.forward(obs_t, as_quantiles=True, with_taus=True)
        quantiles_t = _pick_value_by_action(values, act_t)

        # compute errors with all combination
        y = (rew_tp1 + gamma * q_tp1).view(obs_t.shape[0], -1, 1)
        quantiles_t = quantiles_t.view(obs_t.shape[0], 1, -1)
        taus = taus.view(obs_t.shape[0], 1, -1)
        loss = _quantile_huber_loss(quantiles_t, y, taus)

        return _reduce(loss, reduction)

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
        self.embed = nn.Linear(embed_size, head.feature_size)
        self.fc = nn.Linear(head.feature_size, 1)

    def _make_taus(self, h):
        taus = torch.rand(h.shape[0], self.n_quantiles, device=h.device)
        return taus

    def _compute_quantiles(self, h, taus, detach=False):
        if detach:
            embed_W = self.embed.weight.detach()
            embed_b = self.embed.bias.detach()
            fc_W = self.fc.weight.detach()
            fc_b = self.fc.bias.detach()
        else:
            embed_W = self.embed.weight
            embed_b = self.embed.bias
            fc_W = self.fc.weight
            fc_b = self.fc.bias

        # compute embedding
        steps = torch.arange(self.embed_size, device=h.device).float() + 1
        # (batch, quantile, embedding)
        expanded_taus = taus.view(h.shape[0], -1, 1)
        prior = torch.cos(math.pi * steps.view(1, 1, -1) * expanded_taus)
        # (batch, quantile, embedding) -> (batch, quantile, feature)
        phi = torch.relu(F.linear(prior, embed_W, embed_b))

        # (batch, 1, feature) -> (batch, quantile, feature)
        prod = h.view(h.shape[0], 1, -1) * phi

        # (batch, quantile, feature) -> (batch, quantile)
        quantiles = F.linear(prod, fc_W, fc_b).view(h.shape[0], -1)

        return quantiles

    def forward(self, x, action, as_quantiles=False, with_taus=False):
        h = self.head(x, action)

        taus = self._make_taus(h)

        quantiles = self._compute_quantiles(h, taus)

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

    def compute_error(self,
                      obs_t,
                      act_t,
                      rew_tp1,
                      q_tp1,
                      gamma=0.99,
                      reduction='mean'):
        assert q_tp1.shape == (obs_t.shape[0], self.n_quantiles)

        quantiles_t, taus = self.forward(obs_t,
                                         act_t,
                                         as_quantiles=True,
                                         with_taus=True)

        # compute errors with all combination
        y = (rew_tp1 + gamma * q_tp1).view(obs_t.shape[0], -1, 1)
        quantiles_t = quantiles_t.view(obs_t.shape[0], 1, -1)
        taus = taus.view(obs_t.shape[0], 1, -1)
        loss = _quantile_huber_loss(quantiles_t, y, taus)

        return _reduce(loss, reduction)

    def compute_target(self, x, action):
        return self.forward(x, action, as_quantiles=True)


class DiscreteFQFQFunction(nn.Module):
    def __init__(self, head, action_size, n_quantiles, embed_size):
        super().__init__()
        self.head = head
        self.action_size = action_size
        self.embed_size = embed_size
        self.n_quantiles = n_quantiles
        self.embed = nn.Linear(embed_size, head.feature_size)

        # TODO: refactor this.
        self.fcs = nn.ModuleList()
        for _ in range(self.action_size):
            self.fcs.append(nn.Linear(head.feature_size, 1))

        self.proposal = nn.Linear(head.feature_size, action_size * n_quantiles)

    def _make_taus(self, h):
        # compute taus without making backward path
        flat_proposals = self.proposal(h.detach())
        # (batch, action * quantle) -> (batch, action, quantile)
        proposals = flat_proposals.view(h.shape[0], self.action_size, -1)
        # tau_i+1
        taus = torch.cumsum(torch.softmax(proposals, dim=2), dim=2)

        # tau_i
        pads = torch.zeros(h.shape[0], self.action_size, 1, device=h.device)
        taus_minus = torch.cat([pads, taus[:, :, :-1]], dim=2)

        # tau^
        taus_prime = (taus + taus_minus) / 2

        return taus, taus_minus, taus_prime

    def _compute_quantiles(self, h, taus, detach=False):
        if detach:
            embed_W = self.embed.weight.detach()
            embed_b = self.embed.bias.detach()
            fc_Ws = []
            fc_bs = []
            for fc in self.fcs:
                fc_Ws.append(fc.weight.detach())
                fc_bs.append(fc.bias.detach())
        else:
            embed_W = self.embed.weight
            embed_b = self.embed.bias
            fc_Ws = []
            fc_bs = []
            for fc in self.fcs:
                fc_Ws.append(fc.weight)
                fc_bs.append(fc.bias)

        # compute embedding
        steps = torch.arange(self.embed_size, device=h.device).float() + 1
        # (batch, action, quantile, embedding)
        expanded_taus = taus.view(h.shape[0], self.action_size, -1, 1)
        prior = torch.cos(math.pi * steps.view(1, 1, 1, -1) * expanded_taus)
        # (batch, action, quantile, embedding) -> (batch, action, quantile, feature)
        phi = torch.relu(F.linear(prior, embed_W, embed_b))

        # (batch, 1, 1, feature) -> (batch, action, quantile, feature)
        prod = h.view(h.shape[0], 1, 1, -1) * phi

        # TODO: refactor this
        # (batch, action, quantile, feature) -> (action, batch, quantile, feature)
        prod = prod.transpose(0, 1)
        quantiles = []
        for i in range(self.action_size):
            # (batch, quantile, feature) -> (batch, quantile, 1)
            q = F.linear(prod[i], fc_Ws[i], fc_bs[i])
            # (batch, quantile, 1) -> (1, batch, quantile)
            quantiles.append(q.view(1, h.shape[0], self.n_quantiles))

        # (action, batch, quantile) -> (batch, action, quantile)
        return torch.cat(quantiles, dim=0).transpose(0, 1)

    def forward(self, x, as_quantiles=False, with_taus=False, with_h=False):
        h = self.head(x)

        taus, taus_minus, taus_prime = self._make_taus(h)

        # compute quantiles without making backward path to proposal net
        quantiles = self._compute_quantiles(h, taus_prime.detach())

        rets = []

        if as_quantiles:
            rets.append(quantiles)
        else:
            rets.append(((taus - taus_minus) * quantiles).sum(dim=2))

        if with_taus:
            rets.append([taus, taus_prime])

        if with_h:
            rets.append(h)

        if len(rets) == 1:
            return rets[0]
        return rets

    def compute_error(self,
                      obs_t,
                      act_t,
                      rew_tp1,
                      q_tp1,
                      gamma=0.99,
                      reduction='mean'):
        assert q_tp1.shape == (obs_t.shape[0], self.n_quantiles)

        # extraect quantiles corresponding to act_t
        values, taus, h = self.forward(obs_t,
                                       as_quantiles=True,
                                       with_taus=True,
                                       with_h=True)
        taus, taus_prime = taus

        quantiles_t = _pick_value_by_action(values, act_t)
        taus_prime_t = _pick_value_by_action(taus_prime, act_t)

        # compute errors with all combination
        y = (rew_tp1 + gamma * q_tp1).view(obs_t.shape[0], -1, 1)
        quantiles_t = quantiles_t.view(obs_t.shape[0], 1, -1)
        taus_prime_t = taus_prime_t.view(obs_t.shape[0], 1, -1).detach()
        quantile_loss = _quantile_huber_loss(quantiles_t, y, taus_prime_t)

        # compute proposal network loss
        # original paper explicitly separates the optimization process
        # but, it's combined here
        q_taus = self._compute_quantiles(h.detach(), taus, True)
        q_taus_prime = self._compute_quantiles(h.detach(), taus_prime, True)
        batch_steps = torch.arange(obs_t.shape[0])
        proposal_loss = 2 * q_taus[batch_steps, act_t.view(-1), :-1]
        proposal_loss -= q_taus_prime[batch_steps, act_t.view(-1), :-1]
        proposal_loss -= q_taus_prime[batch_steps, act_t.view(-1), 1:]

        loss = quantile_loss + proposal_loss.mean(dim=1)

        return _reduce(loss, reduction)

    def compute_target(self, x, action):
        quantiles = self.forward(x, as_quantiles=True)
        return _pick_value_by_action(quantiles, action)


class ContinuousFQFQFunction(ContinuousIQNQFunction):
    def __init__(self, head, n_quantiles, embed_size):
        super().__init__(head, n_quantiles, embed_size)
        self.proposal = nn.Linear(head.feature_size, n_quantiles)

    def _make_taus(self, h):
        # compute taus without making backward path
        proposals = self.proposal(h.detach())
        # tau_i+1
        taus = torch.cumsum(torch.softmax(proposals, dim=1), dim=1)

        # tau_i
        pads = torch.zeros(h.shape[0], 1, device=h.device)
        taus_minus = torch.cat([pads, taus[:, :-1]], dim=1)

        # tau^
        taus_prime = (taus + taus_minus) / 2

        return taus, taus_minus, taus_prime

    def forward(self,
                x,
                action,
                as_quantiles=False,
                with_taus=False,
                with_h=False):
        h = self.head(x, action)

        taus, taus_minus, taus_prime = self._make_taus(h)

        # compute quantiles without making backward path to proposal net
        quantiles = self._compute_quantiles(h, taus_prime.detach())

        rets = []

        if as_quantiles:
            rets.append(quantiles)
        else:
            weight = taus - taus_minus
            rets.append((weight * quantiles).sum(dim=1, keepdims=True))

        if with_taus:
            rets.append([taus, taus_prime])

        if with_h:
            rets.append(h)

        if len(rets) == 1:
            return rets[0]
        return rets

    def compute_error(self,
                      obs_t,
                      act_t,
                      rew_tp1,
                      q_tp1,
                      gamma=0.99,
                      reduction='mean'):
        assert q_tp1.shape == (obs_t.shape[0], self.n_quantiles)

        values, taus, h = self.forward(obs_t,
                                       act_t,
                                       as_quantiles=True,
                                       with_taus=True,
                                       with_h=True)
        taus, taus_prime = taus

        # compute errors with all combination
        y = (rew_tp1 + gamma * q_tp1).view(obs_t.shape[0], -1, 1)
        quantiles_t = values.view(obs_t.shape[0], 1, -1)
        taus_prime_t = taus_prime.view(obs_t.shape[0], 1, -1).detach()
        quantile_loss = _quantile_huber_loss(quantiles_t, y, taus_prime_t)

        # compute proposal network loss
        # original paper explicitly separates the optimization process
        # but, it's combined here
        q_taus = self._compute_quantiles(h.detach(), taus, True)
        q_taus_prime = self._compute_quantiles(h.detach(), taus_prime, True)
        proposal_target = q_taus_prime[:, :-1] + q_taus_prime[:, 1:]
        proposal_loss = (2 * q_taus[:, :-1] - proposal_target).mean(dim=1)

        loss = quantile_loss + proposal_loss

        return _reduce(loss, reduction)


class DiscreteQFunction(nn.Module):
    def __init__(self, head, action_size):
        super().__init__()
        self.action_size = action_size
        self.head = head
        self.fc = nn.Linear(head.feature_size, action_size)

    def forward(self, x):
        h = self.head(x)
        return self.fc(h)

    def compute_error(self,
                      obs_t,
                      act_t,
                      rew_tp1,
                      q_tp1,
                      gamma=0.99,
                      reduction='mean'):
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        q_t = (self.forward(obs_t) * one_hot.float()).sum(dim=1, keepdims=True)
        y = rew_tp1 + gamma * q_tp1
        loss = _huber_loss(q_t, y)
        return _reduce(loss, reduction)

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

    def compute_error(self,
                      obs_t,
                      act_t,
                      rew_tp1,
                      q_tp1,
                      gamma=0.99,
                      reduction='mean'):
        q_t = self.forward(obs_t, act_t)
        y = rew_tp1 + gamma * q_tp1
        loss = F.mse_loss(q_t, y, reduction='none')
        return _reduce(loss, reduction)

    def compute_target(self, x, action):
        return self.forward(x, action)


def _reduce_ensemble(y, reduction='min', dim=0, lam=0.75):
    if reduction == 'min':
        return y.min(dim=dim).values
    elif reduction == 'max':
        return y.max(dim=dim).values
    elif reduction == 'mean':
        return y.mean(dim=dim)
    elif reduction == 'none':
        return y
    elif reduction == 'mix':
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    else:
        raise ValueError


def _reduce_quantile_ensemble(y, reduction='min', dim=0, lam=0.75):
    # reduction beased on expectation
    mean = y.mean(dim=2)
    if reduction == 'min':
        indices = mean.min(dim=dim).indices
        return y.transpose(0, 1)[torch.arange(y.shape[1]), indices]
    elif reduction == 'max':
        indices = mean.max(dim=dim).indices
        return y.transpose(0, 1)[torch.arange(y.shape[1]), indices]
    elif reduction == 'none':
        return y
    elif reduction == 'mix':
        min_indices = mean.min(dim=dim).indices
        max_indices = mean.max(dim=dim).indices
        min_values = y.transpose(0, 1)[torch.arange(y.shape[1]), min_indices]
        max_values = y.transpose(0, 1)[torch.arange(y.shape[1]), max_indices]
        return lam * min_values + (1.0 - lam) * max_values
    else:
        raise ValueError


class EnsembleQFunction(nn.Module):
    def __init__(self, q_funcs, bootstrap=False):
        super().__init__()
        self.action_size = q_funcs[0].action_size
        self.q_funcs = nn.ModuleList(q_funcs)
        self.bootstrap = bootstrap and len(q_funcs) > 1

    def compute_error(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        td_sum = 0.0
        for i, q_func in enumerate(self.q_funcs):
            loss = q_func.compute_error(obs_t,
                                        act_t,
                                        rew_tp1,
                                        q_tp1,
                                        gamma,
                                        reduction='none')

            if self.bootstrap:
                mask = torch.randint(0, 2, loss.shape, device=obs_t.device)
                loss *= mask.float()

            td_sum += loss.mean()

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
    def forward(self, x, reduction='mean'):
        values = []
        for q_func in self.q_funcs:
            values.append(q_func(x).view(1, x.shape[0], self.action_size))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)


class EnsembleContinuousQFunction(EnsembleQFunction):
    def forward(self, x, action, reduction='mean'):
        values = []
        for q_func in self.q_funcs:
            values.append(q_func(x, action).view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)


def compute_max_with_n_actions(x,
                               actions,
                               q_func,
                               lam,
                               with_action_indices=False):
    """ Returns weighted target value from sampled actions.

    This calculation is proposed in BCQ paper for the first time.

    `x` should be shaped with `(batch, dim_obs)`.
    `actions` should be shaped with `(batch, N, dim_action)`.

    """
    batch_size = actions.shape[0]
    n_critics = len(q_func.q_funcs)
    n_actions = actions.shape[1]

    # (batch, observation) -> (batch, n, observation)
    expanded_x = x.expand(n_actions, *x.shape).transpose(0, 1)
    # (batch * n, observation)
    flat_x = expanded_x.reshape(-1, *x.shape[1:])
    # (batch, n, action) -> (batch * n, action)
    flat_actions = actions.reshape(batch_size * n_actions, -1)

    # estimate values while taking care of quantiles
    flat_values = q_func.compute_target(flat_x, flat_actions, 'none')
    # reshape to (n_ensembles, batch_size, n, -1)
    transposed_values = flat_values.view(n_critics, batch_size, n_actions, -1)
    # (n_ensembles, batch_size, n, -1) -> (batch_size, n_ensembles, n, -1)
    values = transposed_values.transpose(0, 1)

    # get combination indices
    # (batch_size, n_ensembles, n, -1) -> (batch_size, n_ensembles, n)
    mean_values = values.mean(dim=3)
    #(batch_size, n_ensembles, n) -> (batch_size, n)
    max_values, max_indices = mean_values.max(dim=1)
    min_values, min_indices = mean_values.min(dim=1)
    mix_values = (1.0 - lam) * max_values + lam * min_values
    #(batch_size, n) -> (batch_size,)
    action_indices = mix_values.argmax(dim=1)

    # fuse maximum values and minimum values
    # (batch_size, n_ensembles, n, -1) -> (batch_size, n, n_ensembles, -1)
    values_T = values.transpose(1, 2)
    # (batch, n, n_ensembles, -1) -> (batch * n, n_ensembles, -1)
    flat_values = values_T.reshape(batch_size * n_actions, n_critics, -1)
    # (batch * n, n_ensembles, -1) -> (batch * n, -1)
    bn_indices = torch.arange(batch_size * n_actions)
    max_values = flat_values[bn_indices, max_indices.view(-1)]
    min_values = flat_values[bn_indices, min_indices.view(-1)]
    # (batch * n, -1) -> (batch, n, -1)
    max_values = max_values.view(batch_size, n_actions, -1)
    min_values = min_values.view(batch_size, n_actions, -1)
    mix_values = (1.0 - lam) * max_values + lam * min_values
    # (batch, n, -1) -> (batch, -1)
    result_values = mix_values[torch.arange(x.shape[0]), action_indices]

    if with_action_indices:
        return result_values, action_indices

    return result_values
