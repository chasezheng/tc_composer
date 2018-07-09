from collections import Counter
from typing import MutableSequence, Union, Sequence

from torch import Tensor
import torch
from .settings import DEFAULT_TYPE, tc_type


class UniqueName:
    __slots__ = '_name',

    _COUNTS = Counter()

    def __init__(self, prefix: str = None):
        prefix = prefix or type(self).__name__[:1]

        counts = UniqueName._COUNTS
        if prefix in counts:
            suffix = counts[prefix]
            while prefix + str(suffix) in counts:
                suffix += 1
            name = prefix + str(suffix)
        else:
            name = prefix

        assert name not in counts
        self._name = name
        counts[name] += 1

    def __str__(self):
        return self._name

    def __hash__(self):
        return self._name.__hash__()


class TensorName(UniqueName):
    __slots__ = 'sizes', 'type', 'indices'

    def __init__(self,
                 dim: int,
                 type: str = DEFAULT_TYPE,
                 sizes: Sequence[Union[str, int]] = None,
                 prefix: str = 'T'):
        super(TensorName, self).__init__(prefix=prefix)
        if sizes is not None:
            assert not isinstance(sizes, str)
            assert len(sizes) == dim
        else:
            sizes = tuple('I' for _ in range(dim))

        self.sizes: MutableSequence[Union[int, UniqueName]] = list(n if isinstance(n, int) else UniqueName(n) for n in sizes)
        self.type = type

    @property
    def arg(self):
        sizes = tuple(str(v) for v in self.sizes)

        return f"{self.type}({','.join(sizes)}) {self._name}"

    @staticmethod
    def new_from(tensor: Tensor, prefix: str = None):
        return TensorName(dim=tensor.dim(), type=tc_type(tensor), sizes=tensor.shape, prefix=prefix)

    @staticmethod
    def make_pair(sizes: Sequence[int], prefix: str = None):
        return TensorName(dim=len(sizes), type=DEFAULT_TYPE, sizes=sizes, prefix=prefix), \
               torch.rand(*sizes)

