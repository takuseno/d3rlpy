import dataclasses
from abc import abstractmethod
from collections import defaultdict
from typing import Callable, Dict, Generic, Optional, Sequence, TypeVar

import numpy as np
from tqdm.auto import tqdm
from typing_extensions import Self

from ...base import ImplBase, LearnableBase, LearnableConfig, save_config
from ...constants import IMPL_NOT_INITIALIZED_ERROR
from ...logging import (
    LOG,
    D3RLPyLogger,
    FileAdapterFactory,
    LoggerAdapterFactory,
)
from ...models import EmbeddingModuleFactory, TokenEmbeddingFactory
from ...models.torch import TokenEmbedding
from ...serializable_config import generate_dict_config_field
from ...torch_utility import train_api
from .dataset import (
    GatoEmbeddingMiniBatch,
    GatoReplayBuffer,
    ReplayBufferWithEmbeddingKeys,
)

__all__ = [
    "GatoAlgoImplBase",
    "GatoBaseConfig",
    "GatoAlgoBase",
]


class GatoAlgoImplBase(ImplBase):
    @train_api
    def update(
        self, batch: GatoEmbeddingMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        return self.inner_update(batch, grad_step)

    @abstractmethod
    def inner_update(
        self, batch: GatoEmbeddingMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        pass

    @property
    @abstractmethod
    def token_embeddings(self) -> Dict[str, TokenEmbedding]:
        pass


@dataclasses.dataclass()
class GatoBaseConfig(LearnableConfig):
    context_size: int = 20
    embedding_modules: Dict[
        str, EmbeddingModuleFactory
    ] = generate_dict_config_field(EmbeddingModuleFactory)()
    token_embeddings: Dict[
        str, TokenEmbeddingFactory
    ] = generate_dict_config_field(TokenEmbeddingFactory)()


TGatoImpl = TypeVar("TGatoImpl", bound=GatoAlgoImplBase)
TGatoConfig = TypeVar("TGatoConfig", bound=GatoBaseConfig)


class GatoAlgoBase(
    Generic[TGatoImpl, TGatoConfig],
    LearnableBase[TGatoImpl, TGatoConfig],
):
    def fit(
        self,
        datasets: Sequence[ReplayBufferWithEmbeddingKeys],
        n_steps: int,
        n_steps_per_epoch: int = 10000,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logger_adapter: LoggerAdapterFactory = FileAdapterFactory(),
        show_progress: bool = True,
        save_interval: int = 1,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        enable_ddp: bool = False,
    ) -> None:
        """Trains with given dataset.

        Args:
            datasets: List of offline datasets to train.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            save_interval: Interval to save parameters.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
            enable_ddp: Flag to wrap models with DataDistributedParallel.
        """

        # setup logger
        if experiment_name is None:
            experiment_name = self.__class__.__name__
        logger = D3RLPyLogger(
            adapter_factory=logger_adapter,
            experiment_name=experiment_name,
            with_timestamp=with_timestamp,
        )

        # instantiate implementation
        if self._impl is None:
            LOG.debug("Building models...")
            self.create_impl((0,), 0)
            LOG.debug("Models have been built.")
        else:
            LOG.warning("Skip building models since they're already built.")
        assert self._impl

        # wrap all PyTorch modules with DataDistributedParallel
        if enable_ddp:
            assert self._impl
            self._impl.wrap_models_by_ddp()

        # create GatoReplayBuffer
        replay_buffer = GatoReplayBuffer(datasets, self._impl.token_embeddings)

        # save hyperparameters
        save_config(self, logger)

        # training loop
        n_epochs = n_steps // n_steps_per_epoch
        total_step = 0
        for epoch in range(1, n_epochs + 1):
            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            range_gen = tqdm(
                range(n_steps_per_epoch),
                disable=not show_progress,
                desc=f"Epoch {int(epoch)}/{n_epochs}",
            )

            for itr in range_gen:
                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        batch = replay_buffer.sample_embedding_mini_batch(
                            self._config.batch_size,
                            length=self._config.context_size,
                        )

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(batch)

                    # record metrics
                    for name, val in loss.items():
                        logger.add_metric(name, val)
                        epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                total_step += 1

                # call callback if given
                if callback:
                    callback(self, epoch, total_step)

            # save metrics
            logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

        logger.close()

    def update(self, batch: GatoEmbeddingMiniBatch) -> Dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: Mini-batch data.

        Returns:
            Dictionary of metrics.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch, self._grad_step)
        self._grad_step += 1
        return loss
