import logging
import os
from functools import lru_cache
from typing import Union

import torch
from torch import Tensor

#
# Default settings
#
# todo read settings from file?
OPTIONS_DIR = os.path.join(os.path.expanduser('~'), f'{__package__}/options')
DEFAULT_TENSOR = 'torch.cuda.FloatTensor'
EPSILON = 1e-16
CHECKING_SHAPE = False

#
# Override settings from environmental variables
#
if os.environ.get('UNIT_TESTING', None) == 'True':
    UNIT_TESTING = True
    DEFAULT_TENSOR = 'torch.cuda.DoubleTensor'
    CHECKING_SHAPE = True
else:
    UNIT_TESTING = False

if os.environ.get('BENCHMARKING', None) == 'True':
    BENCHMARKING = True
    CHECKING_SHAPE = False
else:
    BENCHMARKING = False


#
# Configure logging
#

@lru_cache(maxsize=None)
def get_configured_logger(name: str, format: str = None):
    # todo make more formatting option available
    if UNIT_TESTING:
        format = format or '[%(levelname)s] %(name)s.%(funcName)s L.%(lineno)d - %(message)s'
    else:
        format = format or '[%(levelname)s] %(name)s - %(message)s'

    logger = logging.getLogger(name)
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


LOGGER = get_configured_logger(__name__)

#
# Logging
#
LOGGER.info(f'Setting default tensor type: {DEFAULT_TENSOR}')
LOGGER.info(f'Setting epsilon: {EPSILON}')
LOGGER.info(f'Input tensor shape checking: {CHECKING_SHAPE}')
LOGGER.info(f'Saving compiled options in: {OPTIONS_DIR}')
#LOGGER.info(f"Current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
if 'double' in DEFAULT_TENSOR.lower() and BENCHMARKING:
    LOGGER.warning("Using double tensors in benchmark mode.")


#
# Helper functions
#
def tc_type(t: Union[str, Tensor]) -> str:
    """Converts torch tensor type to TC type name
    """
    import torch
    if isinstance(t, str):
        # todo use regex and raise exception for not finding match
        return t.split('.')[-1].replace('Tensor', '').lower()
    elif torch.is_tensor(t):
        return tc_type(t.type())
    raise TypeError(f'Expecting a string or Tensor. Instead found: {type(t)}')


#
# Apply settings
#
torch.set_default_tensor_type(DEFAULT_TENSOR)
DEFAULT_TYPE = tc_type(DEFAULT_TENSOR)
if not os.path.exists(OPTIONS_DIR):
    os.makedirs(OPTIONS_DIR)

#
# Sanity check
#
assert torch.Tensor(1).is_cuda, \
    f'Default tensor type is not cuda: {torch.Tensor(1).type()}. Check tc_composer.settings.'

#
# Clean up
#
del LOGGER, torch, os, lru_cache
