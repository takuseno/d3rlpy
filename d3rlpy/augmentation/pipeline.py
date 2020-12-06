from abc import ABCMeta, abstractmethod


class AugmentationPipeline(metaclass=ABCMeta):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def append(self, augmentation):
        """ Append augmentation to pipeline.

        Args:
            augmentation (d3rlpy.augmentation.base.Augmentation): augmentation.

        """
        self.augmentations.append(augmentation)

    def get_augmentation_types(self):
        """ Returns augmentation types.

        Returns:
            list(str): list of augmentation types.

        """
        return [aug.get_type() for aug in self.augmentations]

    def get_augmentation_params(self):
        """ Returns augmentation parameters.

        Args:
            deep (bool): flag to deeply copy objects.

        Returns:
            list(dict): list of augmentation parameters.

        """
        return [aug.get_params() for aug in self.augmentations]

    @abstractmethod
    def get_params(self, deep=False):
        """ Returns pipeline parameters.

        Returns:
            dict: piple parameters.

        """
        pass

    def transform(self, x):
        """ Returns observation processed by all augmentations.

        Args:
            x (torch.Tensor): observation tensor.

        Returns:
            torch.Tensor: processed observation tensor.

        """
        if not self.augmentations:
            return x

        for augmentation in self.augmentations:
            x = augmentation.transform(x)

        return x

    @abstractmethod
    def process(self, func, inputs, targets):
        """ Runs a given function while augmenting inputs.

        Args:
            func (callable): function to compute.
            inputs (dict): inputs to the func.
            target (list(str)): list of argument names to augment.

        Returns:
            torch.Tensor: the computation result.

        """
        pass


class DrQPipeline(AugmentationPipeline):
    """ Data-reguralized Q augmentation pipeline.

    References:
        * `Kostrikov et al., Image Augmentation Is All You Need: Regularizing
          Deep Reinforcement Learning from Pixels.
          <https://arxiv.org/abs/2004.13649>`_

    Args:
        augmentations (list(d3rlpy.augmentation.base.Augmentation or str)):
            list of augmentations or augmentation types.
        n_mean (int): the number of computations to average

    Attributes:
        augmentations (list(d3rlpy.augmentation.base.Augmentation)):
            list of augmentations.
        n_mean (int): the number of computations to average

    """
    def __init__(self, augmentations=None, n_mean=1):
        if augmentations is None:
            augmentations = []
        super().__init__(augmentations)
        self.n_mean = n_mean

    def get_params(self, deep=False):
        return {'n_mean': self.n_mean}

    def process(self, func, inputs, targets):
        ret = 0.0
        for _ in range(self.n_mean):
            kwargs = dict(inputs)
            for target in targets:
                kwargs[target] = self.transform(kwargs[target])
            ret += func(**kwargs)
        return ret / self.n_mean
