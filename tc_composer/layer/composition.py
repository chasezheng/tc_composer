from torch.nn import Module
from torch.nn import ModuleList

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
