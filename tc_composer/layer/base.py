from abc import ABCMeta, abstractmethod
from typing import Sequence, Mapping, Tuple

import tensor_comprehensions as tc
from torch.autograd import Variable
from torch.nn import Module, ModuleList

from .unique_id import UniqueId
from ..util import cached_property


class Layer(UniqueId, Module, metaclass=ABCMeta):
    def __init__(self, in_n: int, out_n: int, name: str = None):
        super(Layer, self).__init__(name=name)
        self.in_n = in_n
        self.out_n = out_n

    @property
    @abstractmethod
    def lang(self) -> str:
        pass

    @property
    def tc_compile_kwargs(self) -> Mapping:
        return {}

    @cached_property
    def tc_unit(self) -> tc.TcUnit:
        if self.training:
            return tc.TcUnit(self.lang, training=True, name=self.id, backward=self.id + '_grad',
                             **self.tc_compile_kwargs)
        else:
            return tc.TcUnit(self.lang, training=False, name=self.id, **self.tc_compile_kwargs)

    @abstractmethod
    def tc_call_args(self, *args, **kwargs) -> Tuple[Sequence, Mapping]:
        pass

    def forward(self, *args, **kwargs) -> Variable:
        args, kwargs = self.tc_call_args(*args, **kwargs)
        return self.tc_unit(*args, **kwargs)

    def train(self, mode=True) -> None:
        super(Layer, self).train(mode)
        if 'tc_unit' in self.__dict__:
            del self.__dict__['tc_unit']


class Composition(Module):
    __slots__ = '_modules',

    def __init__(self, *modules: Module):
        super(Composition, self).__init__()
        self._modules: ModuleList = ModuleList(modules)

    def __lshift__(self, other: Module):
        return Composition(*self._modules, other)

    def __rshift__(self, other: Module):
        return Composition(other, *self._modules)

    def __repr__(self):
        return self._modules.__repr__()

    def forward(self, *t):
        for m in self._modules:
            t = m(t)
        return t
