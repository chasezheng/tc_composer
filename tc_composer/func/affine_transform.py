import torch
from functools import lru_cache

from .function_with_params import FunctionWithParams
from ..unique_name import TensorName


class AffineTransform(FunctionWithParams):
    __slots__ = '_use_bias', 'in_n', 'out_n'

    def __init__(self, in_n: int, out_n: int, bias: bool = True):
        in_name = TensorName(2, prefix='input')
        out_name = TensorName(2, prefix='output')
        in_name.sizes[1] = in_n
        out_name.sizes[0] = in_name.sizes[0]
        super(AffineTransform, self).__init__(
            in_names=[in_name], outs_to_keep=[out_name], entry_point='affine' if bias else 'linear'
        )

        self.in_n = in_n
        self.out_n = out_n
        self._use_bias = bias

    @property
    @lru_cache(maxsize=None)
    def named_params(self):
        if self._use_bias:
            return TensorName.make_pair(sizes=(self.out_n, self.in_n), prefix='weight'), \
                   TensorName.make_pair(sizes=(self.out_n,), prefix='bias')
        else:
            return TensorName.make_pair(sizes=(self.out_n, self.in_n), prefix='weight'),

    @property
    def def_body(self) -> str:
        input, = self.in_names
        output, = self.outs_to_keep
        weight, _ = self.named_params[0]

        if self._use_bias:
            bias, _ = self.named_params[1]

            return (f"{output}(b, n) +=! {input}(b, i) * {weight}(n, i)\n"
                    f"{output}(b, n) = {output}(b, n) + {bias}(n)")
        else:
            return f"{output}(b, n) +=! {input}(b, i) * {weight}(n, i)"
