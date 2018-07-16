from typing import Sequence

import tensor_comprehensions as tc
import torch
from torch import Tensor, autograd


class Vectorizer:
    __slots__ = ()

    LEN = 44

    @classmethod
    def to_bool(cls, t: Tensor) -> bool:
        t = t.view(-1)
        assert len(t) == 1

        return all(torch.sigmoid(t) > .5)

    @classmethod
    def to_ints(cls, t: Tensor) -> Sequence[int]:
        p = torch.sigmoid(t)
        return torch.ceil((1 - p).log())

    @classmethod
    def to_class(cls, t: Tensor, classes: Sequence[str]) -> str:
        idx = t.argmax(dim=0).item()
        return classes[idx]

    @classmethod
    def to_mapping_options(cls, t: Tensor) -> tc.MappingOptions:
        assert t.dim == 1
        strats = ('Max', 'Min', 'Preserve3Coincident')

        with autograd.set_grad_enabled(False):
            option = tc.MappingOptions('naive') \
                .fixParametersBeforeScheduling(cls.to_bool(t[0])) \
                .intraTileScheduleFusionStrategy(cls.to_class(t[1:4], strats)) \
                .outerScheduleFusionStrategy(cls.to_class(t[4:7], strats)) \
                .mapToBlocks(cls.to_ints(t[7:17])) \
                .mapToThreads(cls.to_ints(t[17:27])) \
                .matchLibraryCalls(cls.to_bool(t[27])) \
                .maxSharedMemory(*cls.to_ints(t[28])) \
                .tile(cls.to_ints(t[29:39])) \
                .unroll(*cls.to_ints(t[39])) \
                .unrollCopyShared(cls.to_bool(t[40])) \
                .usePrivateMemory(cls.to_bool(t[41])) \
                .useReadOnlyCache(cls.to_bool(t[42])) \
                .useSharedMemory(cls.to_bool(t[43]))
        return option

    @classmethod
    def from_mapping_options(cls, option: tc.MappingOptions) -> Tensor:
        raise NotImplementedError
