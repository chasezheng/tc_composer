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
    __slots__ = 'compilation_cache', '_entry_point', 'funcs'

    def __init__(self, entry_point: str = None):
        self.compilation_cache: tc.CompilationCache = None
        self._entry_point = entry_point

    @property
    @lru_cache(maxsize=None)
    def __call__(self):
        if self.compilation_cache is None:
            raise Exception('Please call recompile first.')

        body, in_names, outs_to_keep, outs_to_discard = self.def_components()

        if CHECKING_SHAPE:
            run = self.compilation_cache.run
        else:
            run = self.compilation_cache.unchecked_run

        if outs_to_discard == ():
            def call(*inputs, outputs=(), mrun=run, params=self.params, entry_point=self.entry_point):
                return mrun(entry_point, (*inputs, *params), outputs)
        else:
            if len(outs_to_keep) == 1:
                def call(*inputs, outputs=(), mrun=run, params=self.params, entry_point=self.entry_point):
                    return mrun(entry_point, (*inputs, *params), outputs)[-1]
            else:
                def call(*inputs, outputs=(), mrun=run, params=self.params, entry_point=self.entry_point,
                         discard_first_outs=len(outs_to_discard)):
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
        if isinstance(self, Composition) and isinstance(other, Composition):
            return Composition(*other._funcs, *self.funcs)
        elif isinstance(other, Composition):
            return Composition(*other._funcs, self)
        elif isinstance(self, Composition):
            return Composition(other, *self.funcs)
        else:
            return Composition(other, self)

    def __add__(self, other):
        if isinstance(self, Branch) and isinstance(other, Branch):
            return Branch(*self.funcs, *other.funcs)
        elif isinstance(other, Branch):
            return Branch(self, *other.funcs)
        elif isinstance(self, Branch):
            return Branch(*self.funcs, other)
        else:
            return Branch(self, other)

    def __radd__(self, other):
        if isinstance(other, FunctionWithParams):
            return self.__add__(other)
        else:
            return self  # todo test

    @property
    def entry_point(self):
        return self._entry_point or type(self).__name__

    @abstractmethod
    def def_components(self, in_names: Sequence[TensorName] = None) \
            -> Tuple[str, Sequence[TensorName], Sequence[TensorName], Sequence[TensorName]]:
        """
        :param in_names: Names for inputs (optional)
        :return: statement_string, in_names, outputs_to_keep, outputs_to_discard
        """
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
        """A Sequence of tensors
        """
        return tuple(p for _, p in self.named_params)

    def tc_def(self, *inputs: Tensor) -> str:
        if inputs != ():
            in_names = tuple(TensorName.new_from(inp, prefix='input') for inp in inputs)
        else:
            in_names = None
        body, in_names, outs_to_keep, outs_to_discard = self.def_components(in_names)

        input_and_param_names: Sequence[TensorName] = (*in_names, *(n for n, _ in self.named_params))

        arg_list = ',\n    '.join(n.arg for n in input_and_param_names)
        return_list = ',\n    '.join(o.arg for o in (*outs_to_discard, *outs_to_keep))

        return (f"def {self.entry_point}(\n"
                f"    {arg_list}\n"
                f") -> (\n"
                f"    {return_list}\n"
                f")\n"
                "{\n    " +
                body.replace('\n', '\n    ') +
                "\n}")

    def get_options(self, *inputs: Tensor, error_if_empty: bool = False) -> tc.MappingOptions:
        inputs_and_params = (*inputs, *self.params)
        cache = tc.MappingOptionsCache(self.option_file)
        loaded = cache.load(self.tc_def(*inputs), self.entry_point, inputs_and_params, 1)
        if len(loaded) == 0:
            msg = f'No option loaded from file for input shape - {list(tuple(i.shape) for i in inputs)}.'
            if error_if_empty:
                raise OptionNotFound(msg)
            self.logger.warning(msg)
            self.logger.warning('Initializing naive options.')
            return tc.MappingOptions('naive')
        else:
            self.logger.info(
                f'Option loaded from file for input shape - {list(tuple(i.shape) for i in inputs)}.')
            return loaded[0]

    def recompile(self, *inputs: Tensor, option: tc.MappingOptions = None) -> None:
        if self.compilation_cache is None:
            self.compilation_cache = tc.CompilationCache(self.tc_def(*inputs))

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

        tuner = tc.Tuner(self.tc_def(*inputs), self.option_file if save_result else '')

        if save_result:
            self.logger.info(f'Appending results to {self.option_file}')

        return tuner.tune(self.entry_point, inputs_and_params, start_option, tuner_config)


class Composition(FunctionWithParams):
    __slots__ = '_funcs',

    def __init__(self, *funcs: FunctionWithParams, entry_point: str = None):
        super(Composition, self).__init__(entry_point=entry_point)
        assert len(set(funcs)) == len(funcs), 'Functions should be unique.'
        self._funcs: Sequence[FunctionWithParams] = funcs

    def def_components(self, in_names: Sequence[TensorName] = None):
        in_names = in_names or self._funcs[0].def_components()[1]

        def components_yielder():
            last_outs_to_keep = in_names

            for n, f in enumerate(self._funcs[:-1]):
                try:
                    body, _, outs_to_keep, outs_to_discard = f.def_components(in_names=last_outs_to_keep)
                    yield body, (), (), (*outs_to_keep, *outs_to_discard)
                    last_outs_to_keep = outs_to_keep
                except:
                    self.logger.error(f'n = {n}, f = {f.entry_point}')
                    raise

            yield self._funcs[-1].def_components(in_names=last_outs_to_keep)

        results = tuple(components_yielder())
        body = '\n\n'.join(s for s, _, _, _ in results)
        outs_to_keep = results[-1][-2]
        outs_to_discard = tuple(chain(*(n for _, _, _, n in results)))

        return body, in_names, outs_to_keep, outs_to_discard

    @property
    def funcs(self) -> Sequence[FunctionWithParams]:
        return tuple(self._funcs)

    @property
    @lru_cache(maxsize=None)
    def named_params(self):
        # The order doesn't matter
        return tuple(dict(chain(*(f.named_params for f in self._funcs))).items())


class Branch(FunctionWithParams):
    __slots__ = '_funcs',

    def __init__(self, *funcs: FunctionWithParams, entry_point: str = None):
        super(Branch, self).__init__(entry_point=entry_point)
        assert len(set(funcs)) == len(funcs), 'Functions should be unique.'
        self._funcs = funcs

    def def_components(self, in_names: Sequence[TensorName] = None):
        in_names = in_names or self._funcs[0].def_components()[1]

        def components_yielder():
            for n, f in enumerate(self._funcs):
                try:
                    yield f.def_components(in_names=in_names)
                except:
                    self.logger.error(f'n = {n}, f = {f.entry_point}')  # todo implement and use repr
                    raise

        results = tuple(components_yielder())
        body = '\n\n'.join(s for s, _, _, _ in results)
        outs_to_keep = tuple(chain(*(n for _, _, n, _ in results)))
        outs_to_discard = tuple(chain(*(n for _, _, _, n in results)))

        return body, in_names, outs_to_keep, outs_to_discard

    @property
    def funcs(self) -> Sequence[FunctionWithParams]:
        return tuple(self._funcs)

    @property
    def named_params(self):
        # The order doesn't matter
        return tuple(dict(chain(*(f.named_params for f in self._funcs))).items())


class OptionNotFound(Exception):
    pass
