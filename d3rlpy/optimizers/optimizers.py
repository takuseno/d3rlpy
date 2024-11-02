import dataclasses
from typing import Iterable, Optional, Sequence, Tuple

from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop
from torch.optim.lr_scheduler import LRScheduler

from ..serializable_config import DynamicConfig, generate_config_registration
from .lr_schedulers import LRSchedulerFactory, make_lr_scheduler_field

__all__ = [
    "OptimizerWrapper",
    "OptimizerFactory",
    "SGDFactory",
    "AdamFactory",
    "AdamWFactory",
    "RMSpropFactory",
    "GPTAdamWFactory",
    "register_optimizer_factory",
    "make_optimizer_field",
]


def _get_parameters_from_named_modules(
    named_modules: Iterable[Tuple[str, nn.Module]]
) -> Sequence[nn.Parameter]:
    # retrieve unique set of parameters
    params_dict = {}
    for _, module in named_modules:
        for param in module.parameters():
            if param not in params_dict:
                params_dict[param] = param
    return list(params_dict.values())


class OptimizerWrapper:
    """OptimizerWrapper class.

    This class wraps PyTorch optimizer to add additional steps such as gradient
    clipping.

    Args:
        params: List of torch parameters.
        optim: PyTorch optimizer.
        clip_grad_norm: Maximum norm value of gradients to clip.
    """

    _params: Sequence[nn.Parameter]
    _optim: Optimizer
    _clip_grad_norm: Optional[float]
    _lr_scheduler: Optional[LRScheduler]

    def __init__(
        self,
        params: Sequence[nn.Parameter],
        optim: Optimizer,
        clip_grad_norm: Optional[float] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        self._params = params
        self._optim = optim
        self._clip_grad_norm = clip_grad_norm
        self._lr_scheduler = lr_scheduler

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._optim.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Updates parameters.

        Args:
            grad_step: Total gradient step. This can be used for learning rate
                schedulers.
        """
        # clip gradients
        if self._clip_grad_norm:
            nn.utils.clip_grad_norm_(
                self._params, max_norm=self._clip_grad_norm
            )

        # update parameters
        self._optim.step()

        # schedule learning rate
        if self._lr_scheduler:
            self._lr_scheduler.step()

    @property
    def optim(self) -> Optimizer:
        return self._optim


@dataclasses.dataclass()
class OptimizerFactory(DynamicConfig):
    """A factory class that creates an optimizer object in a lazy way.

    The optimizers in algorithms can be configured through this factory class.
    """

    clip_grad_norm: Optional[float] = None
    lr_scheduler_factory: Optional[LRSchedulerFactory] = (
        make_lr_scheduler_field()
    )

    def create(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> OptimizerWrapper:
        """Returns an optimizer object.

        Args:
            named_modules (list): List of tuples of module names and modules.
            lr (float): Learning rate.

        Returns:
            OptimizerWrapper object.
        """
        named_modules = list(named_modules)
        params = _get_parameters_from_named_modules(named_modules)
        optim = self.create_optimizer(named_modules, lr)
        return OptimizerWrapper(
            params=params,
            optim=optim,
            clip_grad_norm=self.clip_grad_norm,
            lr_scheduler=(
                self.lr_scheduler_factory.create(optim)
                if self.lr_scheduler_factory
                else None
            ),
        )

    def create_optimizer(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> Optimizer:
        raise NotImplementedError


@dataclasses.dataclass()
class SGDFactory(OptimizerFactory):
    """An alias for SGD optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import SGDFactory

        factory = SGDFactory(weight_decay=1e-4)

    Args:
        clip_grad_norm: Maximum norm value of gradients to clip.
        lr_scheduler_factory: LRSchedulerFactory.
        momentum: momentum factor.
        dampening: dampening for momentum.
        weight_decay: weight decay (L2 penalty).
        nesterov: flag to enable Nesterov momentum.
    """

    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False

    def create_optimizer(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> Optimizer:
        return SGD(
            _get_parameters_from_named_modules(named_modules),
            lr=lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )

    @staticmethod
    def get_type() -> str:
        return "sgd"


@dataclasses.dataclass()
class AdamFactory(OptimizerFactory):
    """An alias for Adam optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import AdamFactory

        factory = AdamFactory(weight_decay=1e-4)

    Args:
        clip_grad_norm: Maximum norm value of gradients to clip.
        lr_scheduler_factory: LRSchedulerFactory.
        betas: coefficients used for computing running averages of
            gradient and its square.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        amsgrad: flag to use the AMSGrad variant of this algorithm.
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False

    def create_optimizer(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> Adam:
        return Adam(
            params=_get_parameters_from_named_modules(named_modules),
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

    @staticmethod
    def get_type() -> str:
        return "adam"


@dataclasses.dataclass()
class AdamWFactory(OptimizerFactory):
    """An alias for AdamW optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import AdamWFactory

        factory = AdamWFactory(weight_decay=1e-4)

    Args:
        clip_grad_norm: Maximum norm value of gradients to clip.
        lr_scheduler_factory: LRSchedulerFactory.
        betas: coefficients used for computing running averages of
            gradient and its square.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        amsgrad: flag to use the AMSGrad variant of this algorithm.
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False

    def create_optimizer(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> AdamW:
        return AdamW(
            _get_parameters_from_named_modules(named_modules),
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
            capturable=False,
            differentiable=False,
        )

    @staticmethod
    def get_type() -> str:
        return "adam_w"


@dataclasses.dataclass()
class RMSpropFactory(OptimizerFactory):
    """An alias for RMSprop optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import RMSpropFactory

        factory = RMSpropFactory(weight_decay=1e-4)

    Args:
        clip_grad_norm: Maximum norm value of gradients to clip.
        lr_scheduler_factory: LRSchedulerFactory.
        alpha: smoothing constant.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        momentum: momentum factor.
        centered: flag to compute the centered RMSProp, the gradient is
            normalized by an estimation of its variance.
    """

    alpha: float = 0.95
    eps: float = 1e-2
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = True

    def create_optimizer(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> RMSprop:
        return RMSprop(
            _get_parameters_from_named_modules(named_modules),
            lr=lr,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
        )

    @staticmethod
    def get_type() -> str:
        return "rmsprop"


@dataclasses.dataclass()
class GPTAdamWFactory(OptimizerFactory):
    """AdamW optimizer for Decision Transformer architectures.

    .. code-block:: python

        from d3rlpy.optimizers import GPTAdamWFactory

        factory = GPTAdamWFactory(weight_decay=1e-4)

    Args:
        clip_grad_norm: Maximum norm value of gradients to clip.
        lr_scheduler_factory: LRSchedulerFactory.
        betas: coefficients used for computing running averages of
            gradient and its square.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        amsgrad: flag to use the AMSGrad variant of this algorithm.
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False

    def create_optimizer(
        self, named_modules: Iterable[Tuple[str, nn.Module]], lr: float
    ) -> AdamW:
        named_modules = list(named_modules)
        params_dict = {}
        decay = set()
        no_decay = set()
        for module_name, module in named_modules:
            for param_name, param in module.named_parameters():
                full_name = (
                    f"{module_name}.{param_name}" if module_name else param_name
                )

                if full_name not in params_dict:
                    params_dict[full_name] = param

                if param_name.endswith("bias"):
                    no_decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(
                    module, (nn.Linear, nn.Conv2d)
                ):
                    decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(
                    module, (nn.LayerNorm, nn.Embedding)
                ):
                    no_decay.add(full_name)

        # add non-catched parameters to no_decay
        all_names = set(params_dict.keys())
        remainings = all_names.difference(decay | no_decay)
        no_decay.update(remainings)
        assert len(decay | no_decay) == len(
            _get_parameters_from_named_modules(named_modules)
        )
        assert len(decay & no_decay) == 0

        optim_groups = [
            {
                "params": [params_dict[name] for name in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    params_dict[name] for name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optim_groups,
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            amsgrad=self.amsgrad,
        )

    @staticmethod
    def get_type() -> str:
        return "gpt_adam_w"


register_optimizer_factory, make_optimizer_field = generate_config_registration(
    OptimizerFactory, lambda: AdamFactory()
)


register_optimizer_factory(SGDFactory)
register_optimizer_factory(AdamFactory)
register_optimizer_factory(AdamWFactory)
register_optimizer_factory(RMSpropFactory)
register_optimizer_factory(GPTAdamWFactory)
