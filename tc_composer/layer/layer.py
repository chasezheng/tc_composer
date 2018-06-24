import typing
from abc import ABCMeta, abstractmethod
from collections import Counter
from typing import Mapping, Sequence

import tensor_comprehensions as tc
from torch.autograd import Variable
from torch.nn import Module

from ..util import cached_property


class Layer(Module, metaclass=ABCMeta):
    __NAMES: typing.Counter[str] = Counter()

    def __init__(self, name: str = None):
        super(Layer, self).__init__()
        name = name or type(self).__name__

        if name in Layer.__NAMES:
            Layer.__NAMES[name] += 1
            name = name + str(Layer.__NAMES[name])

        assert name not in Layer.__NAMES
        self.__id: str = name
        Layer.__NAMES[name] += 1

    def __str__(self):
        return self.__id.__str__()

    @property
    def id(self):
        return self.__id

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
    def forward(self, *args, outputs: Sequence[Variable] = None, **kwargs) -> None:
        pass

    def train(self, mode=True) -> None:
        super(Layer, self).train(mode)
        if 'tc_unit' in self.__dict__:
            del self.__dict__['tc_unit']
