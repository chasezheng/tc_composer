from collections import Counter
from typing import Union, Sequence

import torch
from torch import Tensor

from .settings import DEFAULT_TYPE, tc_type

INDEX_NAMES = ['b', 'i', 'j', 'k', 'l', 'm', 'n']


class UniqueName(str):      # todo add a lot of tests
    __slots__ = ()

    _COUNTS = Counter()

    def __new__(cls, prefix: str = None):
        assert prefix is None or isinstance(prefix, str)
        prefix = prefix or cls.__name__[:1]

        counts = UniqueName._COUNTS
        if prefix in counts:
            suffix = counts[prefix]
            while prefix + str(suffix) in counts:
                suffix += 1
            name = prefix + str(suffix)
        else:
            name = prefix

        assert name not in counts
        assert (name.lower() != name) or len(
            name) > 1, f'Lower-case single letters are reserved for index variables.'
        counts[name] += 1
        return super(UniqueName, cls).__new__(cls, name)

    def __init__(self, *args, **kwargs):
        super(UniqueName, self).__init__()


class Size(UniqueName):
    __slots__ = '_num',

    def __new__(cls, var: Union[int, str] = None) -> 'Size':
        if isinstance(var, str):
            obj = super(Size, cls).__new__(cls, prefix=var)
            obj._num: int = None
        else:
            obj = super(Size, cls).__new__(cls, prefix=None)
            obj._num: int = var
        return obj

    def __str__(self):
        if self._num is not None:
            return str(self._num)
        else:
            return self

    def __repr__(self):
        return f'Size({self})'

    @property
    def num(self):
        return self._num

    @num.setter
    def num(self, v):
        assert self._num is None or self._num == v, f"Trying to reset the value of {''.join(self)}={self._num} to {v}"
        self._num = v

    def add(self, *v: Union[str, int]) -> str:
        summant = 0
        for i in v:
            if isinstance(i, int):
                summant += i

        remaining = [i for i in v if not isinstance(i, int)]
        if self._num is not None:
            out = str(self._num + summant)
        else:
            out = self
            if summant > 0:
                out += f' + {summant}'
            elif summant < 0:
                out += f' - {-summant}'

        for i in remaining:
            out += f' + {i}'

        return out

    def sub(self, *v: Union[str, int]) -> str:  # todo tests
        sub_total = 0
        for i in v:
            if isinstance(i, int):
                sub_total -= i

        remaining = [i for i in v if not isinstance(i, int)]
        if self._num is not None:
            out = str(self._num + sub_total)
        else:
            out = self
            if sub_total > 0:
                out += f' - {sub_total}'
            elif sub_total < 0:
                out += f' + {-sub_total}'

        for i in remaining:
            out += f' - {i}'

        return out


class TensorName(UniqueName):
    __slots__ = '_sizes', '_type'

    def __new__(cls,
                dim: int,
                type: str = DEFAULT_TYPE,
                sizes: Sequence[Union[str, int]] = None,
                prefix: str = 'T') -> 'TensorName':
        obj = super(TensorName, cls).__new__(cls, prefix=prefix)
        assert type.lower() in ('double', 'float', 'long')
        if sizes is not None:
            assert not isinstance(sizes, str)
            assert len(sizes) == dim
        else:
            sizes = tuple('S' for _ in range(dim))

        obj._sizes: Sequence[Size] = tuple(v if isinstance(v, Size) else Size(v) for v in sizes)
        obj._type = type

        return obj

    @property
    def arg(self):
        return f"{self._type}({', '.join(str(s) for s in self._sizes)}) {self}"

    @property
    def dim(self):
        return len(self._sizes)

    @property
    def indices(self) -> Sequence[str]:
        indices = INDEX_NAMES[:self.dim]
        if len(indices) < self.dim:
            indices += tuple(indices[-1] + str(i) for i in range(self.dim - len(indices)))

        # Sanity checks
        assert len(indices) == self.dim
        assert len(set(indices)) == self.dim
        return indices

    @property
    def sizes(self) -> Sequence[Size]:
        return self._sizes

    @property
    def type(self):
        return self._type

    @staticmethod
    def new_from(tensor: Tensor, prefix: str = None):
        return TensorName(dim=tensor.dim(), type=tc_type(tensor), sizes=tensor.shape, prefix=prefix)

    @staticmethod
    def make_pair(sizes: Sequence[int], prefix: str = None):
        return TensorName(dim=len(sizes), type=DEFAULT_TYPE, sizes=sizes, prefix=prefix), \
               torch.rand(*sizes)
