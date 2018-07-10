import logging
import os

import torch
from functools import lru_cache
from torch import Tensor
from typing import Union
#
# Configure logging
#

@lru_cache(maxsize=None)
def get_configured_logger(name):
    logger = logging.getLogger(name)
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(('[%(levelname)s] %(name)s - %(message)s')))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

LOGGER = get_configured_logger(__name__)


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
    DEFAULT_TENSOR = 'torch.cuda.DoubleTensor'
    CHECKING_SHAPE = True

if os.environ.get('BENCHMARKING', None) == 'True':
    CHECKING_SHAPE = False
    if 'double' in DEFAULT_TENSOR.lower():
        LOGGER.warning("Using double tensors in benchmark mode.")

#
# Logging
#
LOGGER.info(f'Setting default tensor type: {DEFAULT_TENSOR}')
LOGGER.info(f'Setting epsilon: {EPSILON}')
LOGGER.info(f'Input tensor shape checking: {CHECKING_SHAPE}')
LOGGER.info(f'Saving compiled options in: {OPTIONS_DIR}')


#
# Helper functions
#
def tc_type(t: Union[str, Tensor]) -> str:
    """Converts torch tensor type to TC type name
    """
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
assert torch.Tensor(1).is_cuda, f'Default tensor type is not cuda: {torch.Tensor(1).type()}. Check tc_composer.settings.'


#
# Clean up
#
del LOGGER, torch, os, lru_cache

