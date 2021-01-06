import math
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import Encoder, EncoderWithAction


def _pick_value_by_action(
    values: torch.Tensor, action: torch.Tensor, keepdim: bool = False
) -> torch.Tensor:
    action_size = values.shape[1]
    one_hot = F.one_hot(action.view(-1), num_classes=action_size)
    # take care of 3 dimensional vectors
    if values.dim() == 3:
        one_hot = one_hot.view(-1, action_size, 1)
    masked_values = values * cast(torch.Tensor, one_hot.float())
    return masked_values.sum(dim=1, keepdim=keepdim)


def _huber_loss(
    y: torch.Tensor, target: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    diff = target - y
    cond = diff.detach().abs() < beta
    return torch.where(cond, 0.5 * diff ** 2, beta * (diff.abs() - 0.5 * beta))


def _quantile_huber_loss(
    y: torch.Tensor, target: torch.Tensor, taus: torch.Tensor
) -> torch.Tensor:
    assert y.dim() == 3 and target.dim() == 3 and taus.dim() == 3
    # compute huber loss
    huber_loss = _huber_loss(y, target)
    delta = cast(torch.Tensor, ((target - y).detach() < 0.0).float())
    element_wise_loss = (taus - delta).abs() * huber_loss
    return element_wise_loss.sum(dim=2).mean(dim=1)


def _reduce(value: torch.Tensor, reduction_type: str) -> torch.Tensor:
    if reduction_type == "mean":
        return value.mean()
    elif reduction_type == "sum":
        return value.sum()
    elif reduction_type == "none":
        return value.view(-1, 1)
    raise ValueError("invalid reduction type.")


class QFunction(metaclass=ABCMeta):
    @abstractmethod
    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        pass

    @property
    def action_size(self) -> int:
        pass


class DiscreteQFunction(QFunction):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor]
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def encoder(self) -> Encoder:
        pass


class ContinuousQFunction(QFunction):
    @abstractmethod
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def encoder(self) -> EncoderWithAction:
        pass


class DiscreteMeanQFunction(nn.Module, DiscreteQFunction):  # type: ignore
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self._fc(self._encoder(x)))

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        q_t = (self.forward(obs_t) * one_hot.float()).sum(dim=1, keepdim=True)
        y = rew_tp1 + gamma * q_tp1
        loss = _huber_loss(q_t, y)
        return _reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if action is None:
            return self.forward(x)
        return _pick_value_by_action(self.forward(x), action, keepdim=True)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ContinuousMeanQFunction(nn.Module, ContinuousQFunction):  # type: ignore
    _encoder: EncoderWithAction
    _action_size: int
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction):
        super().__init__()
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._fc = nn.Linear(encoder.get_feature_size(), 1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self._fc(self._encoder(x, action)))

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        q_t = self.forward(obs_t, act_t)
        y = rew_tp1 + gamma * q_tp1
        loss = F.mse_loss(q_t, y, reduction="none")
        return _reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(x, action)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder


class QRQFunction(nn.Module):  # type: ignore
    _n_quantiles: int

    def __init__(self, n_quantiles: int):
        super().__init__()
        self._n_quantiles = n_quantiles

    def _make_taus(self, h: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(
            self._n_quantiles, dtype=torch.float32, device=h.device
        )
        taus = ((steps + 1).float() / self._n_quantiles).view(1, -1)
        taus_dot = (steps.float() / self._n_quantiles).view(1, -1)
        return (taus + taus_dot) / 2.0

    def _compute_quantile_loss(
        self,
        quantiles_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        taus: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        batch_size = rew_tp1.shape[0]
        y = (rew_tp1 + gamma * q_tp1).view(batch_size, -1, 1)
        quantiles_t = quantiles_t.view(batch_size, 1, -1)
        expanded_taus = taus.view(-1, 1, self._n_quantiles)
        return _quantile_huber_loss(quantiles_t, y, expanded_taus)


class DiscreteQRQFunction(QRQFunction, DiscreteQFunction):
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int, n_quantiles: int):
        super().__init__(n_quantiles)
        self._encoder = encoder
        self._action_size = action_size
        self._fc = nn.Linear(
            encoder.get_feature_size(), action_size * n_quantiles
        )

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        h = cast(torch.Tensor, self._fc(h))
        return h.view(-1, self._action_size, self._n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=2)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert q_tp1.shape == (obs_t.shape[0], self._n_quantiles)

        # extraect quantiles corresponding to act_t
        h = self._encoder(obs_t)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        quantiles_t = _pick_value_by_action(quantiles, act_t)

        loss = self._compute_quantile_loss(
            quantiles_t=quantiles_t,
            rew_tp1=rew_tp1,
            q_tp1=q_tp1,
            taus=taus,
            gamma=gamma,
        )

        return _reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self._encoder(x)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        if action is None:
            return quantiles
        return _pick_value_by_action(quantiles, action)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ContinuousQRQFunction(QRQFunction, ContinuousQFunction):
    _action_size: int
    _encoder: EncoderWithAction
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction, n_quantiles: int):
        super().__init__(n_quantiles)
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._fc = nn.Linear(encoder.get_feature_size(), n_quantiles)

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        return cast(torch.Tensor, self._fc(h))

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=1, keepdim=True)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert q_tp1.shape == (obs_t.shape[0], self._n_quantiles)

        h = self._encoder(obs_t, act_t)
        taus = self._make_taus(h)
        quantiles_t = self._compute_quantiles(h, taus)

        loss = self._compute_quantile_loss(
            quantiles_t=quantiles_t,
            rew_tp1=rew_tp1,
            q_tp1=q_tp1,
            taus=taus,
            gamma=gamma,
        )

        return _reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        h = self._encoder(x, action)
        taus = self._make_taus(h)
        return self._compute_quantiles(h, taus)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder


class IQNQFunction(QRQFunction):
    _n_greedy_quantiles: int
    _embed_size: int
    _embed: nn.Linear

    def __init__(
        self,
        n_quantiles: int,
        n_greedy_quantiles: int,
        embed_size: int,
        feature_size: int,
    ):
        super().__init__(n_quantiles)
        self._n_greedy_quantiles = n_greedy_quantiles
        self._embed_size = embed_size
        self._embed = nn.Linear(embed_size, feature_size)

    def _make_taus(self, h: torch.Tensor) -> torch.Tensor:
        if self.training:
            taus = torch.rand(h.shape[0], self._n_quantiles, device=h.device)
        else:
            taus = torch.linspace(
                start=0,
                end=1,
                steps=self._n_greedy_quantiles,
                device=h.device,
                dtype=torch.float32,
            )
            taus = taus.view(1, -1).repeat(h.shape[0], 1)
        return taus

    def _compute_last_feature(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        # compute embedding
        steps = torch.arange(self._embed_size, device=h.device).float() + 1
        # (batch, quantile, embedding)
        expanded_taus = taus.view(h.shape[0], -1, 1)
        prior = torch.cos(math.pi * steps.view(1, 1, -1) * expanded_taus)
        # (batch, quantile, embedding) -> (batch, quantile, feature)
        phi = torch.relu(self._embed(prior))

        # (batch, 1, feature) -> (batch,  quantile, feature)
        return h.view(h.shape[0], 1, -1) * phi


class DiscreteIQNQFunction(IQNQFunction, DiscreteQFunction):
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        n_quantiles: int,
        n_greedy_quantiles: int,
        embed_size: int,
    ):
        super().__init__(
            n_quantiles,
            n_greedy_quantiles,
            embed_size,
            encoder.get_feature_size(),
        )
        self._encoder = encoder
        self._action_size = action_size
        self._fc = nn.Linear(encoder.get_feature_size(), self._action_size)

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        # element-wise product on feature and phi (batch, quantile, feature)
        prod = self._compute_last_feature(h, taus)
        # (batch, quantile, feature) -> (batch, action, quantile)
        return cast(torch.Tensor, self._fc(prod)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=2)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert q_tp1.shape == (obs_t.shape[0], self._n_quantiles)

        # extraect quantiles corresponding to act_t
        h = self._encoder(obs_t)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        quantiles_t = _pick_value_by_action(quantiles, act_t)

        loss = self._compute_quantile_loss(
            quantiles_t=quantiles_t,
            rew_tp1=rew_tp1,
            q_tp1=q_tp1,
            taus=taus,
            gamma=gamma,
        )

        return _reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self._encoder(x)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        if action is None:
            return quantiles
        return _pick_value_by_action(quantiles, action)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ContinuousIQNQFunction(IQNQFunction, ContinuousQFunction):
    _action_size: int
    _encoder: EncoderWithAction
    _fc: nn.Linear

    def __init__(
        self,
        encoder: EncoderWithAction,
        n_quantiles: int,
        n_greedy_quantiles: int,
        embed_size: int,
    ):
        super().__init__(
            n_quantiles,
            n_greedy_quantiles,
            embed_size,
            encoder.get_feature_size(),
        )
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._fc = nn.Linear(encoder.get_feature_size(), 1)

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        # element-wise product on feature and phi (batch, quantile, feature)
        prod = self._compute_last_feature(h, taus)
        # (batch, quantile, feature) -> (batch, quantile)
        return cast(torch.Tensor, self._fc(prod)).view(h.shape[0], -1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=1, keepdim=True)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert q_tp1.shape == (obs_t.shape[0], self._n_quantiles)

        h = self._encoder(obs_t, act_t)
        taus = self._make_taus(h)
        quantiles_t = self._compute_quantiles(h, taus)

        loss = self._compute_quantile_loss(
            quantiles_t=quantiles_t,
            rew_tp1=rew_tp1,
            q_tp1=q_tp1,
            taus=taus,
            gamma=gamma,
        )

        return _reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        h = self._encoder(x, action)
        taus = self._make_taus(h)
        return self._compute_quantiles(h, taus)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder


class FQFQFunction(IQNQFunction):
    _proposal: nn.Linear

    def __init__(self, n_quantiles: int, embed_size: int, feature_size: int):
        super().__init__(n_quantiles, n_quantiles, embed_size, feature_size)
        self._proposal = nn.Linear(feature_size, n_quantiles)

    def _make_fqf_taus(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        proposals = self._proposal(h.detach())

        # tau_i+1
        log_probs = torch.log_softmax(proposals, dim=1)
        probs = log_probs.exp()
        taus = torch.cumsum(probs, dim=1)

        # tau_i
        pads = torch.zeros(h.shape[0], 1, device=h.device)
        taus_minus = torch.cat([pads, taus[:, :-1]], dim=1)

        # tau^
        taus_prime = (taus + taus_minus) / 2

        # entropy for penalty
        entropies = -(log_probs * probs).sum(dim=1)

        return taus, taus_minus, taus_prime, entropies


class DiscreteFQFQFunction(FQFQFunction, DiscreteQFunction):
    _action_size: int
    _entropy_coeff: float
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        n_quantiles: int,
        embed_size: int,
        entropy_coeff: float = 0.0,
    ):
        super().__init__(n_quantiles, embed_size, encoder.get_feature_size())
        self._encoder = encoder
        self._action_size = action_size
        self._fc = nn.Linear(encoder.get_feature_size(), self._action_size)
        self._entropy_coeff = entropy_coeff

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        # element-wise product on feature and phi (batch, quantile, feature)
        prod = self._compute_last_feature(h, taus)
        # (batch, quantile, feature) -> (batch, action, quantile)
        return cast(torch.Tensor, self._fc(prod)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        taus, taus_minus, taus_prime, _ = self._make_fqf_taus(h)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        weight = (taus - taus_minus).view(-1, 1, self._n_quantiles).detach()
        return (weight * quantiles).sum(dim=2)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert q_tp1.shape == (obs_t.shape[0], self._n_quantiles)

        # compute quantiles
        h = self._encoder(obs_t)
        taus, _, taus_prime, entropies = self._make_fqf_taus(h)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        quantiles_t = _pick_value_by_action(quantiles, act_t)

        quantile_loss = self._compute_quantile_loss(
            quantiles_t=quantiles_t,
            rew_tp1=rew_tp1,
            q_tp1=q_tp1,
            taus=taus_prime.detach(),
            gamma=gamma,
        )

        # compute proposal network loss
        # original paper explicitly separates the optimization process
        # but, it's combined here
        proposal_loss = self._compute_proposal_loss(h, act_t, taus, taus_prime)
        proposal_params = list(self._proposal.parameters())
        proposal_grads = torch.autograd.grad(
            outputs=proposal_loss.mean(),
            inputs=proposal_params,
            retain_graph=True,
        )
        # directly apply gradients
        for param, grad in zip(list(proposal_params), proposal_grads):
            param.grad = 1e-4 * grad

        loss = quantile_loss - self._entropy_coeff * entropies

        return _reduce(loss, reduction)

    def _compute_proposal_loss(
        self,
        h: torch.Tensor,
        action: torch.Tensor,
        taus: torch.Tensor,
        taus_prime: torch.Tensor,
    ) -> torch.Tensor:
        q_taus = self._compute_quantiles(h.detach(), taus)
        q_taus_prime = self._compute_quantiles(h.detach(), taus_prime)
        batch_steps = torch.arange(h.shape[0])
        # (batch, n_quantiles - 1)
        q_taus = q_taus[batch_steps, action.view(-1)][:, :-1]
        # (batch, n_quantiles)
        q_taus_prime = q_taus_prime[batch_steps, action.view(-1)]

        # compute gradients
        proposal_grad = 2 * q_taus - q_taus_prime[:, :-1] - q_taus_prime[:, 1:]

        return proposal_grad.sum(dim=1)

    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self._encoder(x)
        _, _, taus_prime, _ = self._make_fqf_taus(h)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        if action is None:
            return quantiles
        return _pick_value_by_action(quantiles, action)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ContinuousFQFQFunction(FQFQFunction, ContinuousQFunction):
    _action_size: int
    _entropy_coeff: float
    _encoder: EncoderWithAction
    _fc: nn.Linear

    def __init__(
        self,
        encoder: EncoderWithAction,
        n_quantiles: int,
        embed_size: int,
        entropy_coeff: float = 0.0,
    ):
        super().__init__(n_quantiles, embed_size, encoder.get_feature_size())
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._fc = nn.Linear(encoder.get_feature_size(), 1)
        self._entropy_coeff = entropy_coeff

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        # element-wise product on feature and phi (batch, quantile, feature)
        prod = self._compute_last_feature(h, taus)
        # (batch, quantile, feature) -> (batch, quantile)
        return cast(torch.Tensor, self._fc(prod)).view(h.shape[0], -1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action)
        taus, taus_minus, taus_prime, _ = self._make_fqf_taus(h)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        weight = (taus - taus_minus).detach()
        return (weight * quantiles).sum(dim=1, keepdim=True)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert q_tp1.shape == (obs_t.shape[0], self._n_quantiles)

        h = self._encoder(obs_t, act_t)
        taus, _, taus_prime, entropies = self._make_fqf_taus(h)
        quantiles_t = self._compute_quantiles(h, taus_prime.detach())

        quantile_loss = self._compute_quantile_loss(
            quantiles_t=quantiles_t,
            rew_tp1=rew_tp1,
            q_tp1=q_tp1,
            taus=taus_prime.detach(),
            gamma=gamma,
        )

        # compute proposal network loss
        # original paper explicitly separates the optimization process
        # but, it's combined here
        proposal_loss = self._compute_proposal_loss(h, taus, taus_prime)
        proposal_params = list(self._proposal.parameters())
        proposal_grads = torch.autograd.grad(
            outputs=proposal_loss.mean(),
            inputs=proposal_params,
            retain_graph=True,
        )
        # directly apply gradients
        for param, grad in zip(list(proposal_params), proposal_grads):
            param.grad = 1e-4 * grad

        loss = quantile_loss - self._entropy_coeff * entropies

        return _reduce(loss, reduction)

    def _compute_proposal_loss(
        self, h: torch.Tensor, taus: torch.Tensor, taus_prime: torch.Tensor
    ) -> torch.Tensor:
        # (batch, n_quantiles - 1)
        q_taus = self._compute_quantiles(h.detach(), taus)[:, :-1]
        # (batch, n_quantiles)
        q_taus_prime = self._compute_quantiles(h.detach(), taus_prime)

        # compute gradients
        proposal_grad = 2 * q_taus - q_taus_prime[:, :-1] - q_taus_prime[:, 1:]
        return proposal_grad.sum(dim=1)

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        h = self._encoder(x, action)
        _, _, taus_prime, _ = self._make_fqf_taus(h)
        return self._compute_quantiles(h, taus_prime.detach())

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder


def _reduce_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    if reduction == "min":
        return y.min(dim=dim).values
    elif reduction == "max":
        return y.max(dim=dim).values
    elif reduction == "mean":
        return y.mean(dim=dim)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


def _gather_quantiles_by_indices(
    y: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # TODO: implement this in general case
    if y.dim() == 3:
        # (N, batch, n_quantiles) -> (batch, n_quantiles)
        return y.transpose(0, 1)[torch.arange(y.shape[1]), indices]
    elif y.dim() == 4:
        # (N, batch, action, n_quantiles) -> (batch, action, N, n_quantiles)
        transposed_y = y.transpose(0, 1).transpose(1, 2)
        # (batch, action, N, n_quantiles) -> (batch * action, N, n_quantiles)
        flat_y = transposed_y.reshape(-1, y.shape[0], y.shape[3])
        head_indices = torch.arange(y.shape[1] * y.shape[2])
        # (batch * action, N, n_quantiles) -> (batch * action, n_quantiles)
        gathered_y = flat_y[head_indices, indices.view(-1)]
        # (batch * action, n_quantiles) -> (batch, action, n_quantiles)
        return gathered_y.view(y.shape[1], y.shape[2], -1)
    raise ValueError


def _reduce_quantile_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    # reduction beased on expectation
    mean = y.mean(dim=-1)
    if reduction == "min":
        indices = mean.min(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "max":
        indices = mean.max(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        min_indices = mean.min(dim=dim).indices
        max_indices = mean.max(dim=dim).indices
        min_values = _gather_quantiles_by_indices(y, min_indices)
        max_values = _gather_quantiles_by_indices(y, max_indices)
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


class EnsembleQFunction(nn.Module):  # type: ignore
    _action_size: int
    _q_funcs: nn.ModuleList
    _bootstrap: bool

    def __init__(
        self,
        q_funcs: List[Union[DiscreteQFunction, ContinuousQFunction]],
        bootstrap: bool = False,
    ):
        super().__init__()
        self._action_size = q_funcs[0].action_size
        self._q_funcs = nn.ModuleList(q_funcs)
        self._bootstrap = bootstrap and len(q_funcs) > 1

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        td_sum = torch.tensor(0.0, dtype=torch.float32, device=obs_t.device)
        for q_func in self._q_funcs:
            loss = q_func.compute_error(
                obs_t, act_t, rew_tp1, q_tp1, gamma, reduction="none"
            )

            if self._bootstrap:
                mask = torch.randint(0, 2, loss.shape, device=obs_t.device)
                loss *= mask.float()

            td_sum += loss.mean()

        return td_sum

    def _compute_target(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        values_list: List[torch.Tensor] = []
        for q_func in self._q_funcs:
            target = q_func.compute_target(x, action)
            values_list.append(target.reshape(1, x.shape[0], -1))

        values = torch.cat(values_list, dim=0)

        if action is None:
            # mean Q function
            if values.shape[2] == self._action_size:
                return _reduce_ensemble(values, reduction)
            # distributional Q function
            n_q_funcs = values.shape[0]
            values = values.view(n_q_funcs, x.shape[0], self._action_size, -1)
            return _reduce_quantile_ensemble(values, reduction)

        if values.shape[2] == 1:
            return _reduce_ensemble(values, reduction, lam=lam)

        return _reduce_quantile_ensemble(values, reduction, lam=lam)

    @property
    def q_funcs(self) -> nn.ModuleList:
        return self._q_funcs


class EnsembleDiscreteQFunction(EnsembleQFunction):
    def forward(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        values = []
        for q_func in self._q_funcs:
            values.append(q_func(x).view(1, x.shape[0], self._action_size))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, reduction))

    def compute_target(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return self._compute_target(x, action, reduction, lam)


class EnsembleContinuousQFunction(EnsembleQFunction):
    def forward(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        values = []
        for q_func in self._q_funcs:
            values.append(q_func(x, action).view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action, reduction))

    def compute_target(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return self._compute_target(x, action, reduction, lam)


def compute_max_with_n_actions_and_indices(
    x: torch.Tensor,
    actions: torch.Tensor,
    q_func: EnsembleContinuousQFunction,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns weighted target value from sampled actions.

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
    flat_values = q_func.compute_target(flat_x, flat_actions, "none")
    # reshape to (n_ensembles, batch_size, n, -1)
    transposed_values = flat_values.view(n_critics, batch_size, n_actions, -1)
    # (n_ensembles, batch_size, n, -1) -> (batch_size, n_ensembles, n, -1)
    values = transposed_values.transpose(0, 1)

    # get combination indices
    # (batch_size, n_ensembles, n, -1) -> (batch_size, n_ensembles, n)
    mean_values = values.mean(dim=3)
    # (batch_size, n_ensembles, n) -> (batch_size, n)
    max_values, max_indices = mean_values.max(dim=1)
    min_values, min_indices = mean_values.min(dim=1)
    mix_values = (1.0 - lam) * max_values + lam * min_values
    # (batch_size, n) -> (batch_size,)
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

    return result_values, action_indices


def compute_max_with_n_actions(
    x: torch.Tensor,
    actions: torch.Tensor,
    q_func: EnsembleContinuousQFunction,
    lam: float,
) -> torch.Tensor:
    return compute_max_with_n_actions_and_indices(x, actions, q_func, lam)[0]
