import logging
import os

import torch

# todo read settings from file?
TENSOR_TYPE = 'torch.cuda.FloatTensor'
TYPE_NAME = 'float'
EPSILON = 1e-16


if os.environ.get('UNIT_TESTING', None) == 'True':
    TENSOR_TYPE = 'torch.cuda.DoubleTensor'
    TYPE_NAME = 'double'


# todo configure logging
logger = logging.getLogger(__name__)
logger.info(f'Setting default tensor type: {TENSOR_TYPE}')
logger.info(f'Setting epsilon: {EPSILON}')

torch.set_default_tensor_type(TENSOR_TYPE)
