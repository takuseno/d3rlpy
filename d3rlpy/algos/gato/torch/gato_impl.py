import dataclasses
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from ....models.torch import GatoTransformer, TokenEmbedding
from ....torch_utility import Modules
from ..base import GatoAlgoImplBase
from ..dataset import GatoEmbeddingMiniBatch

__all__ = ["GatoModules", "GatoImpl"]


@dataclasses.dataclass(frozen=True)
class GatoModules(Modules):
    transformer: GatoTransformer
    embedding_modules: nn.ModuleDict
    optim: Optimizer


class GatoImpl(GatoAlgoImplBase):
    _modules: GatoModules
    _token_embeddings: Dict[str, TokenEmbedding]

    def __init__(
        self,
        modules: GatoModules,
        token_embeddings: Dict[str, TokenEmbedding],
        device: str,
    ):
        super().__init__(
            observation_shape=(0,),
            action_size=0,
            modules=modules,
            device=device,
        )
        self._token_embeddings = token_embeddings

    def inner_update(
        self, batch: GatoEmbeddingMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        self._modules.optim.zero_grad()

        loss = self.compute_loss(batch)

        loss.backward()
        self._modules.optim.step()

        return {"loss": float(loss.cpu().detach().numpy())}

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
