from typing import MutableSequence

from torch.nn import Module


class Composition(Module):
    __slots__ = '_modules',

    def __init__(self, *modules: Module):
        super(Composition, self).__init__()
        self._modules: MutableSequence[Module] = list(modules)
        for n, m in enumerate(modules):
            self.add_module(f'm{n}', m)

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
