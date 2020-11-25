from abc import ABCMeta, abstractmethod


class Augmentation(metaclass=ABCMeta):

    TYPE = 'none'

    @abstractmethod
    def transform(self, x):
        pass

    def get_type(self):
        """ Returns augmentation type.

        Returns:
            str: augmentation type.

        """
        return self.TYPE

    @abstractmethod
    def get_params(self, deep=False):
        pass
