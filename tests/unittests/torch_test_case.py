import logging
import unittest
from functools import lru_cache
from numbers import Number
from typing import Union

import numpy as np
import torch
from torch import DoubleTensor, FloatTensor, HalfTensor, LongTensor
from torch.autograd import Variable

from tc_composer.settings import EPSILON, TYPE_NAME

AnyNumeric = Union[DoubleTensor, FloatTensor, HalfTensor, LongTensor, np.ndarray, Number, Variable]


class TorchTestCase(unittest.TestCase):
    RTOL = 1e-10
    ATOL = EPSILON

    def __init__(self, methodName=None):
        super(TorchTestCase, self).__init__(methodName=methodName)
        if TYPE_NAME != 'double':
            self.logger.warning("Please set default type to `double` for testing. "
                                f"Instead found: {TYPE_NAME}")

    @staticmethod
    def to_numpy(n: AnyNumeric, clone: bool = False) -> np.ndarray:
        if isinstance(n, Variable):
            n = n.data
        if torch.is_tensor(n):
            n: np.ndarray = n.cpu().numpy()
        if isinstance(n, Number):
            n = np.asarray(n)
        elif clone:
            n = n.copy()
        return n

    @property
    @lru_cache(maxsize=None)
    def logger(self):
        logger = logging.getLogger(type(self).__module__)
        return logger

    def assert_allclose(self, actual: AnyNumeric, desired: AnyNumeric,
                        rtol: float = None, atol: float = None,
                        equal_nan: bool = True, err_msg: str = '', verbose: bool = True):
        rtol = rtol or self.RTOL
        atol = atol or self.ATOL

        actual, desired = self.to_numpy(actual), \
                          self.to_numpy(desired)

        if actual.shape == desired.shape:
            self.logger.info(f"Max Absolute difference: {(actual - desired).max()}")
            self.logger.info(
                f"Max relative difference: {np.divide((actual - desired), desired).max()}")

        np.testing.assert_allclose(actual, desired,
                                   rtol, atol,
                                   equal_nan, err_msg, verbose)
