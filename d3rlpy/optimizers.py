import copy
import torch.optim as optim

from torch.optim import SGD, Adam, RMSprop


class OptimizerFactory:
    """ A factory class that creates an optimizer object in a lazy way.

    The optimizers in algorithms can be configured through this factory class.

    .. code-block:: python

        from torch.optim Adam
        from d3rlpy.optimizers import OptimizerFactory
        from d3rlpy.algos import DQN

        factory = OptimizerFactory(Adam, eps=0.001)

        dqn = DQN(optim_factory=factory)

    Note:
        Please check more details at :doc:`references/optimizers`.

    Args:
        optim_cls (type or str): An optimizer class.
        kwargs (any): arbitrary keyword-arguments.

    Attributes:
        optim_cls (type): An optimizer class.
        optim_kwargs (dict): given parameters for an optimizer.

    """
    def __init__(self, optim_cls, **kwargs):
        if isinstance(optim_cls, str):
            self.optim_cls = getattr(optim, optim_cls)
        else:
            self.optim_cls = optim_cls
        self.optim_kwargs = kwargs

    def create(self, params, lr):
        """ Returns an optimizer object.

        Args:
            params (list): a list of PyTorch parameters.
            lr (float): learning rate.

        Returns:
            torch.optim.Optimizer: an optimizer object.

        """
        return self.optim_cls(params, lr=lr, **self.optim_kwargs)

    def get_params(self, deep=False):
        """ Returns optimizer parameters.

        """
        if deep:
            params = copy.deepcopy(self.optim_kwargs)
        else:
            params = self.optim_kwargs
        return {'optim_cls': self.optim_cls.__name__, **params}


class SGDFactory(OptimizerFactory):
    """ An alias for SGD optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import SGDFactory

        factory = SGDFactory(weight_decay=1e-4)

    Args:
        momentum (float): momentum factor.
        dampening (float): dampening for momentum.
        weight_decay (float): weight decay (L2 penalty).
        nesterov (bool): flag to enable Nesterov momentum.

    Attributes:
        optim_cls (type): ``torch.optim.SGD`` class.
        optim_kwargs (dict): given parameters for an optimizer.

    """
    def __init__(self,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False,
                 **kwargs):
        super().__init__(optim_cls=SGD,
                         momentum=momentum,
                         dampening=dampening,
                         weight_decay=weight_decay,
                         nesterov=nesterov)


class AdamFactory(OptimizerFactory):
    """ An alias for Adam optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import AdamFactory

        factory = AdamFactory(weight_decay=1e-4)

    Args:
        betas (tuple): coefficients used for computing running averages of
            gradient and its square.
        eps (float): term added to the denominator to improve numerical
            stability.
        weight_decay (float): weight decay (L2 penalty).
        amsgrad (bool): flag to use the AMSGrad variant of this algorithm.

    Attributes:
        optim_cls (type): ``torch.optim.Adam`` class.
        optim_kwargs (dict): given parameters for an optimizer.

    """
    def __init__(self,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 **kwargs):
        super().__init__(optim_cls=Adam,
                         betas=betas,
                         eps=eps,
                         weight_decay=weight_decay,
                         amsgrad=amsgrad)


class RMSpropFactory(OptimizerFactory):
    """ An alias for RMSprop optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import RMSpropFactory

        factory = RMSpropFactory(weight_decay=1e-4)

    Args:
        alpha (float): smoothing constant.
        eps (float): term added to the denominator to improve numerical
            stability.
        weight_decay (float): weight decay (L2 penalty).
        momentum (float): momentum factor.
        centered (bool): flag to compute the centered RMSProp, the gradient is
            normalized by an estimation of its variance.

    Attributes:
        optim_cls (type): ``torch.optim.RMSprop`` class.
        optim_kwargs (dict): given parameters for an optimizer.

    """
    def __init__(self,
                 alpha=0.99,
                 eps=1e-8,
                 weight_decay=0,
                 momentum=0,
                 centered=True,
                 **kwargs):
        super().__init__(optim_cls=RMSprop,
                         alpha=alpha,
                         eps=eps,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         centered=centered)
