import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EnsembleDiscreteQFunction(nn.Module):
    def __init__(self, heads, action_size):
        super().__init__()
        self.action_size = action_size
        self.q_funcs = nn.ModuleList()
        for head in heads:
            self.q_funcs.append(DiscreteQFunction(head, action_size))

    def forward(self, x, reduction='min'):
        values = []
        for q_func in self.q_funcs:
            values.append(q_func(x).view(1, x.shape[0], self.action_size))
        values = torch.cat(values, dim=0)

        if reduction == 'min':
            return values.min(dim=0).values
        elif reduction == 'max':
            return values.max(dim=0).values
        elif reduction == 'mean':
            return values.mean(dim=0)
        elif reduction == 'none':
            return values
        else:
            raise ValueError

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        td_sum = 0.0
        for q_func in self.q_funcs:
            td_sum += q_func.compute_td(obs_t, act_t, rew_tp1, q_tp1, gamma)
        return td_sum


class ContinuousQFunction(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.fc = nn.Linear(head.feature_size, 1)

    def forward(self, x, action):
        h = self.head(x, action)
        return self.fc(h)

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        q_t = self.forward(obs_t, act_t)
        y = rew_tp1 + gamma * q_tp1
        return F.mse_loss(q_t, y)


class EnsembleContinuousQFunction(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.q_funcs = nn.ModuleList()
        for head in heads:
            self.q_funcs.append(ContinuousQFunction(head))

    def forward(self, x, action, reduction='min'):
        values = []
        for q_func in self.q_funcs:
            values.append(q_func(x, action).view(1, x.shape[0], 1))
        values = torch.cat(values, dim=0)

        if reduction == 'min':
            return values.min(dim=0).values
        elif reduction == 'max':
            return values.max(dim=0).values
        elif reduction == 'mean':
            return values.mean(dim=0)
        elif reduction == 'none':
            return values
        else:
            raise ValueError

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        td_sum = 0.0
        for q_func in self.q_funcs:
            td_sum += q_func.compute_td(obs_t, act_t, rew_tp1, q_tp1, gamma)
        return td_sum
