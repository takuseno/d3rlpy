import dataclasses
import math
from abc import abstractmethod
from collections import defaultdict, deque
from typing import (
    Callable,
    Deque,
    Dict,
    Generic,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
import torch
from tqdm.auto import tqdm
from typing_extensions import Self

from ...base import ImplBase, LearnableBase, LearnableConfig, save_config
from ...constants import IMPL_NOT_INITIALIZED_ERROR
from ...dataset import EpisodeBase
from ...logging import (
    LOG,
    D3RLPyLogger,
    FileAdapterFactory,
    LoggerAdapterFactory,
)
from ...metrics import evaluate_gato_with_environment
from ...mixed_precision import NoCastPrecisionScaler, PrecisionScaler
from ...models import EmbeddingModuleFactory, TokenEmbeddingFactory
from ...models.torch import (
    SeparatorTokenEmbedding,
    TokenEmbedding,
    get_parameter,
)
from ...serializable_config import generate_dict_config_field
from ...torch_utility import eval_api, train_api
from ...types import GymEnv, NDArray, Observation
from .dataset import (
    GatoEmbeddingMiniBatch,
    GatoInputEmbedding,
    GatoReplayBuffer,
    GatoTokenEpisode,
    GatoTokenSlicer,
    ReplayBufferWithEmbeddingKeys,
)

__all__ = [
    "GatoAlgoImplBase",
    "GatoBaseConfig",
    "StatefulGatoWrapper",
    "GatoAlgoBase",
    "GatoEnvironmentEvaluator",
]


class GatoAlgoImplBase(ImplBase):
    @eval_api
    def predict(self, inpt: GatoInputEmbedding) -> int:
        return self.inner_predict(inpt)

    @abstractmethod
    def inner_predict(self, inpt: GatoInputEmbedding) -> int:
        pass

    @train_api
    def update(
        self,
        batch: GatoEmbeddingMiniBatch,
        grad_step: int,
        precision_scaler: PrecisionScaler,
    ) -> Dict[str, float]:
        return self.inner_update(batch, grad_step, precision_scaler)

    @abstractmethod
    def inner_update(
        self,
        batch: GatoEmbeddingMiniBatch,
        grad_step: int,
        precision_scaler: PrecisionScaler,
    ) -> Dict[str, float]:
        pass

    @property
    @abstractmethod
    def token_embeddings(self) -> Dict[str, TokenEmbedding]:
        pass

    @property
    @abstractmethod
    def separator_token_embedding(self) -> SeparatorTokenEmbedding:
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


class StatefulGatoWrapper(Generic[TGatoImpl, TGatoConfig]):
    r"""A stateful wrapper for inference of Gato-based algorithms."""
    _algo: "GatoAlgoBase[TGatoImpl, TGatoConfig]"
    _embeddings: Deque[torch.Tensor]
    _observation_positions: Deque[int]
    _observation_masks: Deque[int]
    _action_masks: Deque[int]
    _observation_to_embedding_keys: Union[str, Sequence[str]]
    _action_embedding_key: str
    _action_token_size: int
    _return_integer: bool
    _context_size: int
    _demonstration: Optional[GatoTokenEpisode]

    def __init__(
        self,
        algo: "GatoAlgoBase[TGatoImpl, TGatoConfig]",
        observation_to_embedding_keys: Union[str, Sequence[str]],
        action_embedding_key: str,
        action_token_size: int,
        return_integer: bool,
        demonstration: Optional[EpisodeBase] = None,
    ):
        assert algo.impl, "algo must be built."
        self._algo = algo
        self._observation_to_embedding_keys = observation_to_embedding_keys
        self._action_embedding_key = action_embedding_key
        self._action_token_size = action_token_size
        self._return_integer = return_integer
        self._context_size = algo.config.context_size
        self._embeddings = deque([], maxlen=self._context_size)
        self._observation_positions = deque([], maxlen=self._context_size)
        self._observation_masks = deque([], maxlen=self._context_size)
        self._action_masks = deque([], maxlen=self._context_size)
        if demonstration:
            self._demonstration = GatoTokenEpisode.from_episode(
                episode=demonstration,
                observation_to_embedding_keys=observation_to_embedding_keys,
                action_embedding_key=action_embedding_key,
                token_embeddings=algo.impl.token_embeddings,
                task_id="demonstration",
            )
        else:
            self._demonstration = None

    def predict(self, x: Observation) -> Union[NDArray, int]:
        r"""Returns action.

        Args:
            x: Observation.

        Returns:
            Action.
        """
        assert self._algo.impl
        token_embeddings = self._algo.impl.token_embeddings

        if isinstance(x, np.ndarray):
            assert isinstance(self._observation_to_embedding_keys, str)
            token_embedding = token_embeddings[
                self._observation_to_embedding_keys
            ]
            embedding = token_embedding(np.expand_dims(x, axis=0))[0]
            for i in range(embedding.shape[0]):
                self._append_observation_embedding(embedding[i], i)
        else:
            assert isinstance(
                self._observation_to_embedding_keys, (list, tuple)
            )
            position = 0
            for key, obs in zip(self._observation_to_embedding_keys, x):
                token_embedding = token_embeddings[key]
                embedding = token_embedding(np.expand_dims(obs, axis=0))[0]
                for i in range(embedding.shape[0]):
                    self._append_observation_embedding(embedding[i], position)
                    position += 1

        self._append_separator_embedding()

        action_token_embedding = token_embeddings[self._action_embedding_key]
        action_values = []
        for i in range(self._action_token_size):
            inpt = GatoInputEmbedding(
                embeddings=torch.stack(list(self._embeddings), dim=0),
                observation_positions=torch.tensor(
                    list(self._observation_positions),
                    dtype=torch.int32,
                    device=self._algo.impl.device,
                ),
                observation_masks=torch.tensor(
                    list(self._observation_masks),
                    dtype=torch.float32,
                    device=self._algo.impl.device,
                ).unsqueeze(dim=1),
                action_masks=torch.tensor(
                    list(self._action_masks),
                    dtype=torch.float32,
                    device=self._algo.impl.device,
                ).unsqueeze(dim=1),
            )
            action_token = self._algo.impl.predict(inpt)
            action_value = action_token_embedding.decode(
                np.array([[action_token]])
            )
            action_values.append(action_value[0][0])
            action_embedding = action_token_embedding(action_value)[0][0]
            self._append_action_embedding(action_embedding)

        ret: Union[NDArray, int]
        if self._return_integer:
            assert self._action_token_size == 1
            ret = int(action_values[0])
        else:
            ret = np.array(action_values, dtype=np.float32)

        return ret

    def _append_observation_embedding(
        self, embedding: torch.Tensor, position: int
    ) -> None:
        if len(self._embeddings) == 0:
            if self._demonstration:
                self._fill_with_demonstration()
            else:
                self._fill_with_padding(embedding)
        self._embeddings.append(embedding)
        self._observation_positions.append(position)
        self._observation_masks.append(1)
        self._action_masks.append(0)

    def _append_action_embedding(self, embedding: torch.Tensor) -> None:
        self._embeddings.append(embedding)
        self._observation_positions.append(0)
        self._observation_masks.append(0)
        self._action_masks.append(1)

    def _append_separator_embedding(self) -> None:
        assert self._algo.impl
        self._embeddings.append(
            get_parameter(self._algo.impl.separator_token_embedding)
        )
        self._observation_positions.append(0)
        self._observation_masks.append(0)
        self._action_masks.append(0)

    def _fill_with_padding(self, embedding: torch.Tensor) -> None:
        for _ in range(self._context_size):
            self._embeddings.append(torch.zeros_like(embedding))
            self._observation_positions.append(0)
            self._observation_masks.append(0)
            self._action_masks.append(0)

    def _fill_with_demonstration(self) -> None:
        assert self._demonstration
        assert self._algo.impl
        end_step = math.ceil(
            self._context_size / self._demonstration.one_step_block_size
        )
        inpt = GatoTokenSlicer()(
            episode=self._demonstration,
            end_step=end_step,
            token_size=self._context_size,
            token_embeddings=self._algo.impl.token_embeddings,
            separator_token_embedding=self._algo.impl.separator_token_embedding,
        )
        for i in range(self._context_size):
            self._embeddings.append(inpt.embeddings[i])
            self._observation_positions.append(
                int(inpt.observation_positions[0])
            )
            self._observation_masks.append(int(inpt.observation_masks[0][0]))
            self._action_masks.append(int(inpt.action_masks[0][0]))

    def reset(self) -> None:
        """Clears stateful information."""
        self._embeddings.clear()
        self._observation_positions.clear()
        self._observation_masks.clear()
        self._action_masks.clear()


class GatoEnvironmentEvaluator:
    _env: GymEnv
    _return_integer: bool
    _observation_to_embedding_keys: Union[str, Sequence[str]]
    _action_embedding_key: str
    _action_token_size: int
    _n_trials: int
    _demonstration: Optional[EpisodeBase]

    def __init__(
        self,
        env: GymEnv,
        return_integer: bool,
        observation_to_embedding_keys: Union[str, Sequence[str]],
        action_embedding_key: str,
        n_trials: int = 10,
        demonstration: Optional[EpisodeBase] = None,
    ):
        self._env = env
        self._return_integer = return_integer
        self._observation_to_embedding_keys = observation_to_embedding_keys
        self._action_embedding = action_embedding_key
        if return_integer:
            self._action_token_size = 1
        else:
            self._action_token_size = env.action_space.shape[0]  # type: ignore
        self._n_trials = n_trials
        self._demonstration = demonstration

    def __call__(
        self, algo: "GatoAlgoBase[GatoAlgoImplBase, GatoBaseConfig]"
    ) -> float:
        wrapper = StatefulGatoWrapper(
            algo=algo,
            observation_to_embedding_keys=self._observation_to_embedding_keys,
            action_embedding_key=self._action_embedding,
            return_integer=self._return_integer,
            action_token_size=self._action_token_size,
            demonstration=self._demonstration,
        )
        return evaluate_gato_with_environment(
            algo=wrapper,
            env=self._env,
            n_trials=self._n_trials,
        )


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
        evaluators: Optional[Dict[str, GatoEnvironmentEvaluator]] = None,
        callback: Optional[Callable[[Self, int, int], None]] = None,
        enable_ddp: bool = False,
        precision_scaler: PrecisionScaler = NoCastPrecisionScaler(),
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
            precision_scaler: Precision scaler for mixed precision training.
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
        replay_buffer = GatoReplayBuffer(
            replay_buffers=datasets,
            token_embeddings=self._impl.token_embeddings,
            separator_token_embedding=self._impl.separator_token_embedding,
            prompt_probability=0.25,
        )

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
                        loss = self.update(batch, precision_scaler)

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

            if evaluators:
                for name, evaluator in evaluators.items():
                    eval_score = evaluator(self)  # type: ignore
                    logger.add_metric(name, eval_score)

            # save metrics
            logger.commit(epoch, total_step)

            # save model parameters
            if epoch % save_interval == 0:
                logger.save_model(total_step, self)

        logger.close()

    def update(
        self, batch: GatoEmbeddingMiniBatch, precision_scaler: PrecisionScaler
    ) -> Dict[str, float]:
        """Update parameters with mini-batch of data.

        Args:
            batch: Mini-batch data.

        Returns:
            Dictionary of metrics.
        """
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        loss = self._impl.update(batch, self._grad_step, precision_scaler)
        self._grad_step += 1
        return loss
