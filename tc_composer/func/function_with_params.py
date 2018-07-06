import os
from abc import ABCMeta, abstractmethod
from functools import lru_cache
from typing import Sequence

import tensor_comprehensions as tc
from numba import cuda
import multiprocessing
from torch import Tensor

from .. import settings
from ..settings import CHECKING_SHAPE, OPTIONS_DIR


class FunctionWithParams(metaclass=ABCMeta):
    __slots__ = 'out_buffers', '__compilation_cache'

    def __init__(self, out_buffers: Sequence[Tensor] = ()):
        self.out_buffers: Sequence[Tensor] = out_buffers
        self.__compilation_cache: tc.CompilationCache = None

    def __call__(self, *inputs: Tensor, outputs: Sequence[Tensor] = ()):
        outputs = outputs or self.out_buffers

        if CHECKING_SHAPE:
            return self.__compilation_cache.run(self.entry_point, (*inputs, *self.params), outputs=outputs)
        else:
            return self.__compilation_cache.unchecked_run(self.entry_point, (*inputs, *self.params), outputs=outputs)

    @property
    def entry_point(self):
        names = tc.tclib.parse_defs(self.tc_def)
        assert len(names) == 1, 'Only one function def is allowed.'
        return names[0]

    @property
    @lru_cache(maxsize=None)
    def logger(self):
        return settings.get_configured_logger(self.entry_point)

    @property
    @abstractmethod
    def params(self) -> Sequence[Tensor]:
        pass

    @property
    @abstractmethod
    def tc_def(self) -> str:
        pass

    @property
    def option_file(self):
        fname = '_'.join((self.entry_point,
                          *cuda.get_current_device().name.decode('utf-8').split()))
        return os.path.join(OPTIONS_DIR, fname)

    def recompile(self, *inputs: Tensor, option: tc.MappingOptions = None) -> None:
        if self.__compilation_cache is None:
            self.__compilation_cache = tc.CompilationCache(self.tc_def)

        self.logger.info(f'''Compiling for input shape - {list(tuple(i.shape) for i in inputs)}.''')
        self.__compilation_cache.compile(
            self.entry_point, (*inputs, *self.params),
                              option or self.get_options(*inputs))

    def get_options(self, *inputs: Tensor) -> tc.MappingOptions:
        inputs_and_params = (*inputs, *self.params)
        cache = tc.MappingOptionsCache(self.option_file)
        loaded = cache.load(self.tc_def, self.entry_point, inputs_and_params, 1)
        if len(loaded) == 0:
            self.logger.warning(f'No option loaded from file for input shape - {list(tuple(i.shape) for i in inputs)}.')
            self.logger.warning('Initializing naive options.')
            return tc.MappingOptions('naive')
        else:
            self.logger.info(f'Option loaded from file for input shape - {list(tuple(i.shape) for i in inputs)}.')
            return loaded[0]

    def tune_options(
            self,
            inputs: Sequence[Tensor],
            start_option: tc.MappingOptions = None,
            tuner_config: tc.TunerConfig = None,
            save_result: bool = True
    ) -> tc.MappingOptions:
        inputs_and_params = (*inputs, *self.params)
        tuner_config = tuner_config or tc.TunerConfig().threads(multiprocessing.cpu_count())

        if start_option is None:
            self.logger.info(f'Loading start options from file - {self.option_file}')
            start_option = self.get_options(*inputs)

        tuner = tc.Tuner(self.tc_def, self.option_file if save_result else '')

        if save_result:
            self.logger.info(f'Appending results to {self.option_file}')

        return tuner.tune(self.entry_point, inputs_and_params, start_option, tuner_config)
