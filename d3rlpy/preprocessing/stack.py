import numpy as np


class StackedObservation:
    """ StackedObservation class.

    This class is used to stack images to handle temporal features.

    References:
        * `Mnih et al., Human-level control through deep reinforcement
          learning. <https://www.nature.com/articles/nature14236>`_

    Args:
        observation_shape (tuple): image observation shape.
        n_frames (int): the number of frames to stack.
        dtype (int): numpy data type.

    Attributes:
        image_channels (int): the number of channles of image.
        n_frames (int): the number of frames to stack.
        dtype (int): numpy data type.
        stack (numpy.ndarray): stacked observation.

    """
    def __init__(self, observation_shape, n_frames, dtype=np.uint8):
        self.image_channels = observation_shape[0]
        image_size = observation_shape[1:]
        self.n_frames = n_frames
        self.dtype = dtype
        stacked_shape = (self.image_channels * n_frames, *image_size)
        self.stack = np.zeros(stacked_shape, dtype=self.dtype)

    def append(self, image):
        """ Stack new image.

        Args:
            image (numpy.ndarray): image observation.

        """
        assert image.dtype == self.dtype
        self.stack = np.roll(self.stack, -self.image_channels, axis=0)
        self.stack[self.image_channels * (self.n_frames - 1):] = image.copy()

    def eval(self):
        """ Returns stacked observation.

        Returns:
            numpy.ndarray: stacked observation.

        """
        return self.stack

    def clear(self):
        """ Clear stacked observation by filling 0.
        """
        self.stack.fill(0)
