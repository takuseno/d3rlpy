from abc import ABCMeta, abstractmethod


class Augmentation(metaclass=ABCMeta):
    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def get_type(self):
        pass

    @abstractmethod
    def get_params(self, deep=False):
        pass
