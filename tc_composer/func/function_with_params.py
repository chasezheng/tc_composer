import multiprocessing
import os
from abc import ABCMeta, abstractmethod
from functools import lru_cache
from itertools import chain
from typing import Sequence, Tuple

import tensor_comprehensions as tc
from numba import cuda
from torch import Tensor

from .. import settings
from ..settings import CHECKING_SHAPE, OPTIONS_DIR
from ..unique_name import TensorName


class FunctionWithParams(metaclass=ABCMeta):
    __slots__ = 'in_names', 'outs_to_keep', 'outs_to_discard', 'compilation_cache', '_entry_point'

    def __init__(self,
                 in_names: Sequence[TensorName],
                 outs_to_keep: Sequence[TensorName],
                 outs_to_discard: Sequence[TensorName] = (),
                 entry_point: str = None):
        self.in_names: Sequence[TensorName] = tuple(in_names)
        self.outs_to_keep: Sequence[TensorName] = tuple(outs_to_keep)
        self.outs_to_discard = tuple(outs_to_discard)
        self.compilation_cache: tc.CompilationCache = None
        self._entry_point = entry_point

    @property
    @lru_cache(maxsize=None)
    def __call__(self):
        if self.compilation_cache is None:
            raise Exception('Please call recompile first.')

        if CHECKING_SHAPE:
            run = self.compilation_cache.run
        else:
            run = self.compilation_cache.unchecked_run

        if self.outs_to_discard == ():
            def call(*inputs, outputs=(), mrun=run, params=self.params, entry_point=self.entry_point):
                return mrun(entry_point, (*inputs, *params), outputs)
        else:
            if len(self.outs_to_keep) == 1:
                def call(*inputs, outputs=(), mrun=run, params=self.params, entry_point=self.entry_point):
                    return mrun(entry_point, (*inputs, *params), outputs)[-1]
            else:
                def call(*inputs, outputs=(), mrun=run, params=self.params, entry_point=self.entry_point,
                         discard_first_outs=len(self.outs_to_discard)):
                    return mrun(entry_point, (*inputs, *params), outputs)[discard_first_outs:]

        return call

    def __lshift__(self, other):
        """Return `self << other`. self precedes other
        """
        if isinstance(self, Composition) and isinstance(other, Composition):
            return Composition(*self.funcs, *other.funcs)
        elif isinstance(other, Composition):
            return Composition(self, *other.funcs)
        elif isinstance(self, Composition):
            return Composition(*self.funcs, other)
        else:
            return Composition(self, other)

    def __rshift__(self, other):
        """Return `self >> other`. self follows other
        """
        return other.__lshift__(self)

    @property
    def entry_point(self):
        return self._entry_point or type(self).__name__

    @property
    @abstractmethod
    def def_body(self) -> str:
        pass

    @property
    @lru_cache(maxsize=None)
    def logger(self):
        return settings.get_configured_logger(self.entry_point)

    @property
    @abstractmethod
    def named_params(self) -> Sequence[Tuple[TensorName, Tensor]]:
        pass

    @property
    def option_file(self):
        fname = '_'.join((self.entry_point,
                          *cuda.get_current_device().name.decode('utf-8').split()))
        return os.path.join(OPTIONS_DIR, fname)

    @property
    def params(self) -> Sequence[Tensor]:
        """A Sequence of pairs of names and tensors
        """
        return tuple(p for _, p in self.named_params)

    @property
    def tc_def(self) -> str:
        input_and_param_names: Sequence[TensorName] = (*self.in_names, *(n for n, _ in self.named_params))

        arg_list = ',\n    '.join(n.arg for n in input_and_param_names)
        return_list = ', '.join(str(o) for o in (*self.outs_to_discard, *self.outs_to_keep))

        return (f"def {self.entry_point}(\n"
                f"    {arg_list}\n"
                f") -> ({return_list})\n"
                "{\n    " +
                self.def_body.replace('\n', '\n    ') +
                "\n}")

    @staticmethod
    def compose(*funcs: 'FunctionWithParams', entry_point: str = None) -> 'Composition':
        return Composition(*funcs, entry_point=entry_point)

    def get_options(self, *inputs: Tensor, error_if_empty: bool = False) -> tc.MappingOptions:
        inputs_and_params = (*inputs, *self.params)
        cache = tc.MappingOptionsCache(self.option_file)
        loaded = cache.load(self.tc_def, self.entry_point, inputs_and_params, 1)
        if len(loaded) == 0:
            msg = f'No option loaded from file for input shape - {list(tuple(i.shape) for i in inputs)}.'
            if error_if_empty:
                raise OptionNotFound(msg)
            self.logger.warning(msg)
            self.logger.warning('Initializing naive options.')
            return tc.MappingOptions('naive')
        else:
            self.logger.info(f'Option loaded from file for input shape - {list(tuple(i.shape) for i in inputs)}.')
            return loaded[0]

    def recompile(self, *inputs: Tensor, option: tc.MappingOptions = None) -> None:
        if self.compilation_cache is None:
            self.compilation_cache = tc.CompilationCache(self.tc_def)

        self.logger.info(f'''Compiling for input shape - {list(tuple(i.shape) for i in inputs)}.''')
        self.compilation_cache.compile(
            self.entry_point, (*inputs, *self.params),
            option or self.get_options(*inputs))

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


class Composition(FunctionWithParams):
    __slots__ = 'funcs',

    def __init__(self, *funcs: FunctionWithParams, entry_point: str = None):
        super(Composition, self).__init__(
            in_names=funcs[0].in_names,
            outs_to_keep=funcs[-1].outs_to_keep,
            outs_to_discard=tuple(
                chain(*(f.outs_to_discard for f in funcs), *(f.outs_to_keep for f in funcs[:-1]))),
            entry_point=entry_point
        )
        for n, f in enumerate(funcs[1:]):
            assert len(funcs[n].outs_to_keep) == len(f.in_names)  # todo err msg
            for o, i in zip(funcs[n].outs_to_keep, f.in_names):
                assert len(o.sizes) == len(i.sizes)  # todo err msg

        self.funcs = funcs

    @property
    @lru_cache(maxsize=None)
    def named_params(self):
        # The order doesn't matter
        return tuple(chain(*(f.named_params for f in self.funcs)))

    @property
    @lru_cache(maxsize=None)
    def def_body(self):
        def statement_yielder():
            yield self.funcs[0].def_body + '\n'
            for n, f in enumerate(self.funcs[1:]):
                saved = f.in_names
                try:
                    f.in_names = self.funcs[n].outs_to_keep
                    yield f.def_body + '\n'
                finally:
                    f.in_names = saved

        return '\n'.join(statement_yielder())


class OptionNotFound(Exception):
    pass
