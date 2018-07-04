import torch
from torch.nn import Parameter

from .function_with_params import FunctionWithParams
from ..settings import TYPE_NAME
from torch import Tensor
from typing import Sequence, Union
import tensor_comprehensions as tc



class AffineTransform(FunctionWithParams):
    __slots__ = 'in_n', '_use_bias', 'weight', 'bias', '_params'

    def __init__(self, in_n: int, out_n: int, bias: bool = True):
        super(AffineTransform, self).__init__()
        self.in_n = in_n
        self._use_bias = bias

        self.weight = Parameter(torch.randn(out_n, in_n))
        self.bias = Parameter(torch.randn(out_n))
        if self._use_bias:
            self._params = (self.weight, self.bias)
        else:
            self._params = (self.weight,)

    def __call__(self, t: Tensor, outputs: Sequence[Tensor] = ()) -> Tensor:
        return super(AffineTransform, self).__call__(t.view(-1, self.in_n), outputs=outputs)

    @property
    def params(self):
        return self._params

    @property
    def tc_def(self) -> str:
        if self._use_bias:
            return (f"def affine_transform({TYPE_NAME}(batch_size, in_n) input,\n"
                    f"                      {TYPE_NAME}(out_n, in_n) weight,\n"
                    f"                      {TYPE_NAME}(out_n) bias) -> (output) {'{'}\n"
                    "    output(b, n) +=! input(b, i) * weight(n, i)\n" +
                    "    output(b, n) = output(b, n) + bias(n)\n"
                    "}")
        else:
            return (f"def linear_transform({TYPE_NAME}(batch_size, in_n) input,\n"
                    f"                      {TYPE_NAME}(out_n, in_n) weight) -> (output) {'{'}\n"
                    "    output(b, n) +=! input(b, i) * weight(n, i)\n"
                    "}")

    def recompile(self, inp: Tensor, option: tc.MappingOptions = None):
        super(AffineTransform, self).recompile(inp.view(-1, self.in_n), option=option)