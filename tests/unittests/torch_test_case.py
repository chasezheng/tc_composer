import logging
import unittest
from functools import lru_cache
from numbers import Number
from typing import Union, Sequence

import numpy as np
import torch
from torch.cuda import DoubleTensor, FloatTensor, HalfTensor, LongTensor
from torch.autograd import Variable

from tc_composer.settings import EPSILON, DEFAULT_TYPE, get_configured_logger

# Type definition
DataType = (DoubleTensor, FloatTensor, HalfTensor, LongTensor, np.ndarray, Number, Variable)
AnyNumeric = Union[(*DataType, *(Sequence[T] for T in DataType))]


class TorchTestCase(unittest.TestCase):
    RTOL = 1e-10
    ATOL = EPSILON

    def __init__(self, methodName=None):
        super(TorchTestCase, self).__init__(methodName=methodName)
        if DEFAULT_TYPE != 'double':
            self.logger.warning("Please set default type to `double` for testing. "
                                f"Instead found: {DEFAULT_TYPE}")

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
        if not isinstance(n, np.ndarray):
            raise TypeError(f'Found type: {type(n)}')
        return n

    @property
    @lru_cache(maxsize=None)
    def logger(self):
        return get_configured_logger(type(self).__name__)

    def assert_allclose(self, actual: AnyNumeric, desired: AnyNumeric,
                        rtol: float = None, atol: float = None,
                        equal_nan: bool = True, err_msg: str = '', verbose: bool = True):
        if isinstance(actual, (list, tuple)):
            for n, (a, d) in enumerate(zip(actual, desired)):
                try:
                    self.assert_allclose(actual=a, desired=d,
                                     rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg, verbose=verbose)
                except:
                    self.logger.error(f'At n={n}')
                    raise
            return

        rtol = rtol or self.RTOL
        atol = atol or self.ATOL

        actual, desired = self.to_numpy(actual), \
                          self.to_numpy(desired)

        if actual.shape == desired.shape:
            self.logger.info(f"Max Absolute difference: {(actual - desired).max()}")
            current_setting = np.seterr(divide='ignore', invalid='ignore')
            try:
                self.logger.info(
                    f"Max relative difference: {np.divide((actual - desired), desired).max()}")
            finally:
                np.seterr(**current_setting)

        np.testing.assert_allclose(actual, desired,
                                   rtol, atol,
                                   equal_nan, err_msg, verbose)
