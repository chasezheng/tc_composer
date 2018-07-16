from typing import MutableSequence

import torch
from torch import Tensor
from torch import nn


class Decorrelation(nn.Module):
    __slots__ = '_m',

    def __init__(self):
        super(Decorrelation, self).__init__()
        self._m: MutableSequence[Tensor] = []
        self.register_backward_hook(self._backward_hook())

    def _backward_hook(self, *args):
        if len(self._m) == 0:
            return
        else:
            for m in self._m:
                self.scaled_abs_det(m).backward()
                self._m.clear()
            return

    def forward(self, *input: Tensor):
        for n in range(min(len(self._m), len(input))):
            self._m[n] = self._m[n] + torch.matmul(input[n].view(-1, 1), input[n].view(1, -1))
        for n in range(len(self._m), len(input)):
            self._m[n] = torch.matmul(input[n].view(-1, 1), input[n].view(1, -1))
        return input

    @staticmethod
    def scaled_abs_det(m: Tensor) -> Tensor:
        with torch.set_grad_enabled(False):
            U, s, V = torch.svd(m.inverse())
        s = V.t().matmul(m.matmul(U)).diag()

        return s.div(s.mean()).prod(0)
