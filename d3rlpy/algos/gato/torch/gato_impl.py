import dataclasses
import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from ....models.torch import (
    GatoTransformer,
    SeparatorTokenEmbedding,
    TokenEmbedding,
)
from ....torch_utility import Modules
from ..base import GatoAlgoImplBase
from ..dataset import GatoEmbeddingMiniBatch, GatoInputEmbedding

__all__ = ["GatoModules", "GatoImpl"]


@dataclasses.dataclass(frozen=True)
class GatoModules(Modules):
    transformer: GatoTransformer
    embedding_modules: nn.ModuleDict
    separator_token_embedding: SeparatorTokenEmbedding
    optim: Optimizer


class GatoImpl(GatoAlgoImplBase):
    _modules: GatoModules
    _token_embeddings: Dict[str, TokenEmbedding]
    _clip_grad_norm: float
    _initial_learning_rate: float
    _maximum_learning_rate: float
    _warmup_steps: int
    _final_steps: int

    def __init__(
        self,
        modules: GatoModules,
        token_embeddings: Dict[str, TokenEmbedding],
        clip_grad_norm: float,
        initial_learning_rate: float,
        maximum_learning_rate: float,
        warmup_steps: int,
        final_steps: int,
        device: str,
    ):
        super().__init__(
            observation_shape=(0,),
            action_size=0,
            modules=modules,
            device=device,
        )
        self._token_embeddings = token_embeddings
        self._clip_grad_norm = clip_grad_norm
        self._initial_learning_rate = initial_learning_rate
        self._maximum_learning_rate = maximum_learning_rate
        self._warmup_steps = warmup_steps
        self._final_steps = final_steps

    def inner_predict(self, inpt: GatoInputEmbedding) -> int:
        _, logits = self._modules.transformer(
            torch.unsqueeze(inpt.embeddings, dim=0),
            torch.unsqueeze(inpt.observation_masks, dim=0),
            torch.unsqueeze(inpt.observation_positions, dim=0),
            torch.unsqueeze(inpt.action_masks, dim=0),
        )
        return int(np.argmax(logits[0][-1].cpu().detach().numpy()))

    def inner_update(
        self, batch: GatoEmbeddingMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        self._modules.optim.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self._modules.transformer.parameters())
            + list(self._modules.embedding_modules.parameters())
            + list(self._modules.separator_token_embedding.parameters()),
            self._clip_grad_norm,
        )

        self._modules.optim.step()

        # schedule learning rate
        # linear warmup
        offset = self._maximum_learning_rate - self._initial_learning_rate
        learning_rate = self._initial_learning_rate + offset * min(
            1.0, grad_step / self._warmup_steps
        )
        if grad_step > self._warmup_steps:
            # cosine learning rate decay
            progress = (grad_step - self._warmup_steps) / max(
                1, self._final_steps - self._warmup_steps
            )
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            learning_rate = lr_mult * learning_rate
        for param_group in self._modules.optim.param_groups:
            param_group["lr"] = learning_rate

        return {
            "loss": float(loss.cpu().detach().numpy()),
            "learning_rate": learning_rate,
        }

    def compute_loss(self, batch: GatoEmbeddingMiniBatch) -> torch.Tensor:
        _, logits = self._modules.transformer(
            batch.embeddings,
            batch.observation_masks,
            batch.observation_positions,
            batch.action_masks,
        )
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[2]),
            batch.action_tokens[:, 1:].reshape(-1).long(),
            reduction="none",
        )
        return (loss * batch.action_masks[:, 1:, :].reshape(-1)).mean()

    @property
    def token_embeddings(self) -> Dict[str, TokenEmbedding]:
        return self._token_embeddings

    @property
    def separator_token_embedding(self) -> SeparatorTokenEmbedding:
        return self._modules.separator_token_embedding
