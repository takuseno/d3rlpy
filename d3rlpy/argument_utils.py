from .encoders import create_encoder_factory, EncoderFactory
from .q_functions import create_q_func_factory, QFunctionFactory
from .preprocessing.scalers import create_scaler, Scaler
from .augmentation import create_augmentation, AugmentationPipeline
from .augmentation.base import Augmentation
from .gpu import Device


def check_encoder(value):
    """ Checks value and returns EncoderFactory object.

    Returns:
        d3rlpy.encoders.EncoderFactory: encoder factory object.

    """
    if isinstance(value, EncoderFactory):
        return value
    if isinstance(value, str):
        return create_encoder_factory(value)
    raise ValueError('This argument must be str or EncoderFactory object.')


def check_q_func(value):
    """ Checks value and returns QFunctionFactory object.

    Returns:
        d3rlpy.q_functions.QFunctionFactory: Q function factory object.

    """
    if isinstance(value, QFunctionFactory):
        return value
    if isinstance(value, str):
        return create_q_func_factory(value)
    raise ValueError('This argument must be str or QFunctionFactory object.')


def check_scaler(value):
    """ Checks value and returns Scaler object.

    Returns:
        d3rlpy.preprocessing.scalers.Scaler: scaler object.

    """
    if isinstance(value, Scaler):
        return value
    if isinstance(value, str):
        return create_scaler(value)
    if value is None:
        return None
    raise ValueError('This argument must be str or Scaler object.')


def check_augmentation(value):
    """ Checks value and returns AugmentationPipeline object.

    Returns:
        d3rlpy.augmentation.AugmentationPipeline: augmentation pipeline object.

    """
    if isinstance(value, AugmentationPipeline):
        return value
    if isinstance(value, list):
        augmentations = []
        for v in value:
            if isinstance(v, str):
                v = create_augmentation(v)
            elif not isinstance(v, Augmentation):
                raise ValueError('str or Augmentation is expected.')
            augmentations.append(v)
        return AugmentationPipeline(augmentations)
    if value is None:
        return AugmentationPipeline([])
    raise ValueError('This argument must be list or AugmentationPipeline.')


def check_use_gpu(value):
    """ Checks value and returns Device object.

    Returns:
        d3rlpy.gpu.Device: device object.

    """
    # isinstance cannot tell difference between bool and int
    if type(value) == bool:
        if value:
            return Device(0)
        return None
    if type(value) == int:
        return Device(value)
    if isinstance(value, Device):
        return value
    if value is None:
        return None
    raise ValueError('This argument must be bool, int or Device.')
