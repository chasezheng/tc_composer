import logging
import os

import torch
from functools import lru_cache

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
TENSOR_TYPE = 'torch.cuda.FloatTensor'
TYPE_NAME = 'float'
EPSILON = 1e-16
CHECKING_SHAPE = True

#
# Override settings from environmental variables
#
if os.environ.get('UNIT_TESTING', None) == 'True':
    TENSOR_TYPE = 'torch.cuda.DoubleTensor'
    TYPE_NAME = 'double'

if os.environ.get('BENCHMARKING', None) == 'True':
    CHECKING_SHAPE = False
    if TYPE_NAME == 'double':
        LOGGER.warning("Using double tensors in benchmark mode.")

#
# Logging
#
LOGGER.info(f'Setting default tensor type: {TENSOR_TYPE}')
LOGGER.info(f'Setting epsilon: {EPSILON}')
LOGGER.info(f'Input tensor shape checking: {CHECKING_SHAPE}')
LOGGER.info(f'Saving compiled options in: {OPTIONS_DIR}')

#
# Apply settings
#
torch.set_default_tensor_type(TENSOR_TYPE)
if not os.path.exists(OPTIONS_DIR):
    os.makedirs(OPTIONS_DIR)

#
# Clean up
#
del LOGGER, torch, os, lru_cache

